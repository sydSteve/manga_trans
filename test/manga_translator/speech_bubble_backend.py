from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from .models import Bubble
from .utils import clamp_box


@dataclass
class SpeechBubbleSegmentationStatus:
    configured_source: str
    enabled: bool
    ultralytics_available: bool
    torch_available: bool
    model_loaded: bool = False
    load_error_code: str = ""
    load_error_message: str = ""
    resolved_source: str = ""
    attempted_sources: list[str] = field(default_factory=list)
    last_infer_error_code: str = ""
    last_infer_error_message: str = ""
    last_detection_count: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "configured_source": self.configured_source,
            "enabled": self.enabled,
            "ultralytics_available": self.ultralytics_available,
            "torch_available": self.torch_available,
            "model_loaded": self.model_loaded,
            "load_error_code": self.load_error_code,
            "load_error_message": self.load_error_message,
            "resolved_source": self.resolved_source,
            "attempted_sources": list(self.attempted_sources),
            "last_infer_error_code": self.last_infer_error_code,
            "last_infer_error_message": self.last_infer_error_message,
            "last_detection_count": self.last_detection_count,
        }


class SpeechBubbleSegmentationBackend:
    def __init__(self, config):
        self.config = config
        bubble_cfg = config.bubble
        self.status = SpeechBubbleSegmentationStatus(
            configured_source=bubble_cfg.model_path,
            enabled=bool(bubble_cfg.enabled),
            ultralytics_available=find_spec("ultralytics") is not None,
            torch_available=find_spec("torch") is not None,
        )
        self.model = None
        self.torch = None

        if not self.status.enabled:
            self.status.load_error_code = "disabled"
            self.status.load_error_message = "bubble backend disabled"
            return

        if not self.status.ultralytics_available or not self.status.torch_available:
            missing = []
            if not self.status.ultralytics_available:
                missing.append("ultralytics")
            if not self.status.torch_available:
                missing.append("torch")
            self.status.load_error_code = "missing_dependencies"
            self.status.load_error_message = ",".join(missing)
            return

        self._load()

    def _candidate_sources(self) -> list[str]:
        configured = (self.config.bubble.model_path or "").strip()
        return [configured] if configured else []

    def _load(self) -> None:
        runtime_dir = Path(__file__).resolve().parents[2] / "debug" / "ultralytics"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_dir))

        import torch
        from ultralytics import YOLO

        self.torch = torch
        last_error_code = "missing_model"
        last_error_message = "No speech bubble model configured"

        for source in self._candidate_sources():
            source_path = Path(source)
            self.status.attempted_sources.append(str(source_path))
            if source_path.exists() and source_path.suffix.lower() != ".pt":
                last_error_code = "unsupported_model_format"
                last_error_message = (
                    f"Expected an Ultralytics .pt checkpoint, got {source_path.name}"
                )
                continue

            try:
                model = YOLO(str(source_path if source_path.exists() else source))
                self.model = model
                self.status.model_loaded = True
                self.status.load_error_code = ""
                self.status.load_error_message = ""
                self.status.resolved_source = str(source_path if source_path.exists() else source)
                return
            except Exception as exc:  # noqa: BLE001
                last_error_code, last_error_message = self._map_load_error(exc, source_path)

        self.status.load_error_code = last_error_code
        self.status.load_error_message = last_error_message

    @staticmethod
    def _map_load_error(exc: Exception, source_path: Path) -> tuple[str, str]:
        message = str(exc)
        if source_path and not source_path.exists():
            return "missing_model", f"Model not found: {source_path}"
        if "not a PyTorch model" in message or "unsupported" in message.lower():
            return "unsupported_model_format", message
        return "load_failed", message

    def detect(self, image_bgr: np.ndarray) -> list[Bubble]:
        self.status.last_infer_error_code = ""
        self.status.last_infer_error_message = ""
        self.status.last_detection_count = 0

        if self.model is None or self.torch is None:
            return []

        device = "cpu"
        if self.config.device == "cuda" and self.torch.cuda.is_available():
            device = "0"

        try:
            results = self.model.predict(
                source=image_bgr,
                conf=float(self.config.bubble.min_confidence),
                device=device,
                verbose=False,
            )
            if not results:
                return []

            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return []

            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            polygons = getattr(getattr(result, "masks", None), "xy", None) or []

            bubbles: list[Bubble] = []
            for index, (box, score) in enumerate(zip(xyxy, scores, strict=False)):
                bbox = tuple(int(round(float(num))) for num in box)
                polygon_points: list[list[int]] = []
                if index < len(polygons):
                    polygon = np.asarray(polygons[index], dtype=np.float32)
                    if polygon.ndim == 2 and polygon.shape[0] >= 3:
                        polygon_points = self._normalize_polygon(polygon, image_bgr.shape)
                        bbox = (
                            int(np.floor(np.min(polygon[:, 0]))),
                            int(np.floor(np.min(polygon[:, 1]))),
                            int(np.ceil(np.max(polygon[:, 0]))),
                            int(np.ceil(np.max(polygon[:, 1]))),
                        )

                bbox = clamp_box(bbox, image_bgr.shape)
                x1, y1, x2, y2 = bbox
                if x2 - x1 < 24 or y2 - y1 < 24:
                    continue
                if not polygon_points:
                    polygon_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                bubbles.append(
                    Bubble(
                        bbox=bbox,
                        polygon=polygon_points,
                        score=float(score),
                        source="speech_bubble_segmentation",
                    )
                )

            bubbles.sort(key=lambda bubble: bubble.score, reverse=True)
            self.status.last_detection_count = len(bubbles)
            return bubbles
        except Exception as exc:  # noqa: BLE001
            self.status.last_infer_error_code = "infer_failed"
            self.status.last_infer_error_message = str(exc)
            return []

    @staticmethod
    def _normalize_polygon(polygon: np.ndarray, image_shape: tuple[int, ...]) -> list[list[int]]:
        height, width = image_shape[:2]
        normalized: list[list[int]] = []
        seen: set[tuple[int, int]] = set()
        for point in polygon:
            x = int(np.clip(round(float(point[0])), 0, max(0, width - 1)))
            y = int(np.clip(round(float(point[1])), 0, max(0, height - 1)))
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            normalized.append([x, y])
        return normalized if len(normalized) >= 3 else []

    def runtime_summary(self) -> dict[str, object]:
        summary = self.status.as_dict()
        summary["backend_name"] = "speech_bubble_segmentation" if self.status.model_loaded else "edges"
        summary["model_path_hit"] = bool(self.status.resolved_source)
        return summary
