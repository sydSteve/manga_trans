from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from torchvision import models

from .config import AppConfig
from .models import TextRegion
from .utils import expand_box


def _mask_bbox(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    if mask is None or mask.size == 0 or not np.any(mask > 0):
        return None
    coords = np.column_stack(np.where(mask > 0))
    y1 = int(coords[:, 0].min())
    y2 = int(coords[:, 0].max()) + 1
    x1 = int(coords[:, 1].min())
    x2 = int(coords[:, 1].max()) + 1
    return x1, y1, x2, y2


class YuzuMarkerStyleBackend:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model_path = Path(config.style.model_path)
        self.labels_path = Path(config.style.labels_path)
        self.labels_ex_path = Path(config.style.labels_ex_path)
        self.device = "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = None
        self.labels: list[dict[str, object]] = []
        self.model_loaded = False
        self.runtime_error = ""
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.runtime_error = "missing_model_file"
            return
        if not self.labels_path.exists():
            self.runtime_error = "missing_labels_file"
            return
        try:
            raw_labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
            raw_labels_ex = []
            if self.labels_ex_path.exists():
                raw_labels_ex = json.loads(self.labels_ex_path.read_text(encoding="utf-8"))

            state = load_file(str(self.model_path), device="cpu")
            prefix = "model._orig_mod.model."
            stripped = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
            num_classes = int(stripped["fc.weight"].shape[0])

            labels: list[dict[str, object]] = []
            for index in range(num_classes):
                base = raw_labels[index] if index < len(raw_labels) else {"path": f"unknown/{index}", "language": "unknown"}
                ext = raw_labels_ex[index] if index < len(raw_labels_ex) else {}
                merged = dict(base)
                merged.update(ext)
                labels.append(merged)
            self.labels = labels

            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(stripped, strict=True)
            model.eval()
            model.to(self.device)
            self.model = model
            self.model_loaded = True
            self.runtime_error = ""
        except Exception as exc:  # noqa: BLE001
            self.runtime_error = str(exc)
            self.model = None
            self.model_loaded = False

    def analyze(
        self,
        image_rgb: np.ndarray,
        region: TextRegion,
        mask: np.ndarray | None,
    ) -> dict[str, object]:
        if not self.model_loaded or self.model is None:
            return {}
        crop = self._prepare_crop(image_rgb, region, mask)
        if crop is None:
            return {}
        tensor = self._preprocess(crop).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_k = min(max(1, self.config.style.top_k), probs.shape[0])
        values, indices = torch.topk(probs, k=top_k)
        family_scores: dict[str, float] = {}
        weight_scores: dict[str, float] = {}
        best_path = ""
        best_language = "unknown"
        best_confidence = 0.0

        for score_tensor, index_tensor in zip(values, indices):
            index = int(index_tensor.item())
            score = float(score_tensor.item())
            label = self.labels[index] if index < len(self.labels) else {"path": f"unknown/{index}", "language": "unknown"}
            family_hint = self._infer_family_hint(label)
            weight_hint = self._infer_weight_hint(str(label.get("path", "")))
            family_scores[family_hint] = family_scores.get(family_hint, 0.0) + score
            weight_scores[weight_hint] = weight_scores.get(weight_hint, 0.0) + score
            if score > best_confidence:
                best_confidence = score
                best_path = str(label.get("path", ""))
                best_language = str(label.get("language", "unknown"))

        if best_confidence < self.config.style.min_confidence:
            return {}

        family_hint = max(family_scores.items(), key=lambda item: item[1])[0]
        weight_hint = max(weight_scores.items(), key=lambda item: item[1])[0]
        resolved_font_path = self._resolve_font_path(family_hint)
        return {
            "backend": "yuzumarker",
            "style_confidence": round(best_confidence, 4),
            "font_family": family_hint,
            "weight_hint": weight_hint,
            "font_path": resolved_font_path,
            "resolved_font_path": resolved_font_path,
            "orientation_hint": self._infer_orientation_hint(region, best_language),
            "source_label_path": best_path,
        }

    def _prepare_crop(
        self,
        image_rgb: np.ndarray,
        region: TextRegion,
        mask: np.ndarray | None,
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = expand_box(region.box, image_rgb.shape, 4)
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if mask is not None and mask.size > 0:
            crop_mask = mask[y1:y2, x1:x2]
            if crop_mask.ndim == 3:
                crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)
            mask_bin = (crop_mask > 0).astype(np.uint8)
            if np.any(mask_bin > 0):
                crop = crop.copy()
                crop[mask_bin == 0] = 255
        return crop

    @staticmethod
    def _preprocess(crop: np.ndarray) -> torch.Tensor:
        target = 224
        h, w = crop.shape[:2]
        scale = target / max(1, max(h, w))
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target, target, 3), 255, dtype=np.uint8)
        offset_y = (target - resized_h) // 2
        offset_x = (target - resized_w) // 2
        canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
        tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0)

    @staticmethod
    def _infer_family_hint(label: dict[str, object]) -> str:
        path = str(label.get("path", "")).lower()
        serif = bool(label.get("serif", False))
        if any(token in path for token in ["kai", "gyosho", "marker", "pop", "round", "maru", "hand", "brush", "sho"]):
            return "handwritten_or_display"
        if any(token in path for token in ["gothic", "sans", "hei"]):
            return "sans/gothic"
        if any(token in path for token in ["mincho", "serif", "song", "ming", "batang"]):
            return "serif/mincho"
        return "serif/mincho" if serif else "sans/gothic"

    @staticmethod
    def _infer_weight_hint(path: str) -> str:
        lowered = path.lower()
        if any(token in lowered for token in ["heavy", "black", "ultra", "extrabold", "bold", "w9", "w10", "w12", "w14"]):
            return "bold"
        if any(token in lowered for token in ["light", "thin", "extralight"]):
            return "light"
        return "regular"

    def _resolve_font_path(self, family_hint: str) -> str:
        candidates = self.config.style.font_family_map.get(family_hint, []) + self.config.render.font_paths
        for path in candidates:
            if Path(path).exists():
                return path
        return ""

    @staticmethod
    def _infer_orientation_hint(region: TextRegion, language_bias: str) -> str:
        x1, y1, x2, y2 = region.box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        if language_bias.lower() in {"ja", "cjk"} and height >= width * 1.3:
            return "vertical"
        return "auto"


class StyleService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.yuzu_backend = YuzuMarkerStyleBackend(config)

    def analyze(self, image_rgb: np.ndarray, region: TextRegion, mask: np.ndarray | None) -> dict[str, object]:
        style_box = _mask_bbox(region.render_mask)
        if style_box is None:
            style_box = region.bubble_bbox if region.bubble_bbox else region.box
        x1, y1, x2, y2 = style_box
        crop = image_rgb[y1:y2, x1:x2]
        default_style = {
            "backend": "heuristic",
            "text_color": list(self.config.style.default_text_color),
            "stroke_color": list(self.config.style.default_stroke_color),
            "stroke_width": self.config.style.default_stroke_width,
            "orientation_hint": "auto",
            "font_family": "sans/gothic",
            "weight_hint": "regular",
            "font_path": next((path for path in self.config.render.font_paths if Path(path).exists()), ""),
            "resolved_font_path": next((path for path in self.config.render.font_paths if Path(path).exists()), ""),
            "style_confidence": 0.0,
        }
        if crop.size == 0:
            region.style = default_style
            return default_style

        render_mask = None
        if region.render_mask is not None and region.render_mask.shape[:2] == image_rgb.shape[:2]:
            render_mask = region.render_mask[y1:y2, x1:x2]
            if not np.any(render_mask > 0):
                render_mask = None

        masked_crop = crop.copy()
        if render_mask is not None:
            masked_crop[render_mask == 0] = 255

        gray = cv2.cvtColor(masked_crop, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(masked_crop, cv2.COLOR_RGB2HSV)
        _, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, light_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if render_mask is not None:
            dark_mask = cv2.bitwise_and(dark_mask, render_mask)
            light_mask = cv2.bitwise_and(light_mask, render_mask)
        white_ratio = float(np.mean(gray >= 215))
        saturation = float(np.mean(hsv[:, :, 1].astype(np.float32)))

        style = dict(default_style)
        style["text_color"] = list(self._sample_color(crop, dark_mask, tuple(self.config.style.default_text_color)))
        style["stroke_color"] = list(self._sample_color(crop, light_mask, tuple(self.config.style.default_stroke_color)))
        style["stroke_width"] = self._estimate_stroke_width(dark_mask)

        yuzu_style = self.yuzu_backend.analyze(image_rgb, region, mask)
        if yuzu_style:
            style.update(yuzu_style)

        use_safe_monochrome = (
            region.region_type == "dialogue_bubble" and (region.bubble_bbox is not None or white_ratio >= 0.56) and saturation <= 48.0
        ) or (
            region.region_type == "narration_box" and white_ratio >= 0.66 and saturation <= 32.0
        )
        if use_safe_monochrome:
            style["text_color"] = list(self.config.style.default_text_color)
            style["stroke_color"] = list(self.config.style.default_stroke_color)
            style["stroke_width"] = max(1, min(int(style.get("stroke_width", 1)), 2))

        if not style.get("font_path"):
            fallback_font = next((path for path in self.config.render.font_paths if Path(path).exists()), "")
            style["font_path"] = fallback_font
            style["resolved_font_path"] = fallback_font

        region.style = style
        region.debug["style_backend"] = style.get("backend", "heuristic")
        region.debug["style_confidence"] = style.get("style_confidence", 0.0)
        region.debug["style_sample_box"] = [x1, y1, x2, y2]
        return style

    @staticmethod
    def _sample_color(crop: np.ndarray, mask: np.ndarray, default: tuple[int, int, int]) -> tuple[int, int, int]:
        pixels = crop[mask > 0]
        if pixels.size == 0:
            return default
        color = np.median(pixels, axis=0)
        return tuple(int(np.clip(channel, 0, 255)) for channel in color.tolist())

    @staticmethod
    def _estimate_stroke_width(mask: np.ndarray) -> int:
        if not np.any(mask > 0):
            return 1
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        value = int(round(float(np.percentile(distance[distance > 0], 70))))
        return max(1, min(value, 4))

    def runtime_summary(self) -> dict[str, object]:
        return {
            "backend_name": "yuzumarker" if self.yuzu_backend.model_loaded else "heuristic",
            "model_loaded": self.yuzu_backend.model_loaded,
            "model_path_hit": self.yuzu_backend.model_path.exists(),
            "load_error": self.yuzu_backend.runtime_error,
        }
