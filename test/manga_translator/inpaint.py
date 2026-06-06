from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import AppConfig
from .models import DocumentContext, TextRegion
from .utils import clamp_box, shrink_box


DEFAULT_OVERLAY_ALPHA = 0.82
DEFAULT_OVERLAY_PADDING = 12
DEFAULT_OVERLAY_EDGE_BLUR = 9
DEFAULT_OVERLAY_INSET_RATIO = 0.06


def _polygon_to_page_mask(shape: tuple[int, int], polygon: list[list[int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(polygon) < 3:
        return mask
    points = np.asarray(polygon, dtype=np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, max(0, shape[1] - 1))
    points[:, 1] = np.clip(points[:, 1], 0, max(0, shape[0] - 1))
    cv2.fillPoly(mask, [points], 255)
    return mask


def _mask_bbox(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    if mask is None or mask.size == 0 or not np.any(mask > 0):
        return None
    coords = np.column_stack(np.where(mask > 0))
    y1 = int(coords[:, 0].min())
    y2 = int(coords[:, 0].max()) + 1
    x1 = int(coords[:, 1].min())
    x2 = int(coords[:, 1].max()) + 1
    return x1, y1, x2, y2


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(1, box[2] - box[0]) * max(1, box[3] - box[1])


def _intersect_boxes(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _expand_box(box: tuple[int, int, int, int], shape: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return clamp_box((x1 - padding, y1 - padding, x2 + padding, y2 + padding), shape)


def _shrink_box_xy(box: tuple[int, int, int, int], inset_x: int, inset_y: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    inset_x = max(0, int(inset_x))
    inset_y = max(0, int(inset_y))
    return (
        x1 + inset_x,
        y1 + inset_y,
        max(x1 + inset_x + 1, x2 - inset_x),
        max(y1 + inset_y + 1, y2 - inset_y),
    )


def _merge_boxes(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _masked_coverage(mask: np.ndarray, container_mask: np.ndarray) -> float:
    inside = container_mask > 0
    if not np.any(inside):
        return 0.0
    return float(np.mean(mask[inside] > 0))


class OverlayInpaintBackend:
    def __init__(self, config: AppConfig):
        self.config = config
        self.overlay_alpha = DEFAULT_OVERLAY_ALPHA
        self.overlay_padding = DEFAULT_OVERLAY_PADDING
        self.overlay_edge_blur = DEFAULT_OVERLAY_EDGE_BLUR

    def _overlay_inset(self, bubble_box: tuple[int, int, int, int]) -> tuple[int, int]:
        width = max(1, bubble_box[2] - bubble_box[0])
        height = max(1, bubble_box[3] - bubble_box[1])
        inset_x = max(2, min(self.overlay_padding, int(round(width * DEFAULT_OVERLAY_INSET_RATIO))))
        inset_y = max(2, min(self.overlay_padding, int(round(height * DEFAULT_OVERLAY_INSET_RATIO))))

        if height >= width * 2.2:
            inset_x = max(1, inset_x // 2)
        elif width >= height * 2.2:
            inset_y = max(1, inset_y // 2)

        if min(width, height) <= 56:
            inset_x = min(inset_x, 2)
            inset_y = min(inset_y, 2)
        return inset_x, inset_y

    @staticmethod
    def _is_outside_text_region(region: TextRegion) -> bool:
        return bool(region.debug.get("outside_bubble", False))

    def bubble_inner_mask(self, page_shape: tuple[int, int], region: TextRegion) -> np.ndarray:
        if self._is_outside_text_region(region):
            mask = np.zeros(page_shape, dtype=np.uint8)
            x1, y1, x2, y2 = region.box
            mask[y1:y2, x1:x2] = 255
            return mask

        if region.bubble_polygon:
            bubble_mask = _polygon_to_page_mask(page_shape, region.bubble_polygon)
        elif region.bubble_bbox:
            bubble_mask = np.zeros(page_shape, dtype=np.uint8)
            x1, y1, x2, y2 = region.bubble_bbox
            bubble_mask[y1:y2, x1:x2] = 255
        else:
            bubble_mask = np.zeros(page_shape, dtype=np.uint8)
            x1, y1, x2, y2 = region.box
            bubble_mask[y1:y2, x1:x2] = 255

        base_box = region.bubble_bbox if region.bubble_bbox else region.box
        inner_box = shrink_box(base_box, page_shape, max(2, self.config.inpaint.bubble_inner_shrink))
        inner_mask = np.zeros_like(bubble_mask)
        x1, y1, x2, y2 = inner_box
        inner_mask[y1:y2, x1:x2] = 255
        return cv2.bitwise_and(bubble_mask, inner_mask)

    def choose_overlay_box(
        self,
        page_shape: tuple[int, int],
        region: TextRegion,
        bubble_mask: np.ndarray,
    ) -> dict[str, object]:
        bubble_box = _mask_bbox(bubble_mask) or (region.bubble_bbox if region.bubble_bbox else region.box)
        if self._is_outside_text_region(region):
            return {
                "box": bubble_box,
                "source": "outside_text_box",
                "matched_union_box": None,
                "basis": "region_box",
                "inner_inset": [0, 0],
                "source_detail": "outside_text_region_box",
            }

        matched_union_raw = region.debug.get("matched_text_union_box")
        matched_union_box: tuple[int, int, int, int] | None = None
        if isinstance(matched_union_raw, list) and len(matched_union_raw) == 4:
            try:
                matched_union_box = tuple(int(value) for value in matched_union_raw)
            except Exception:  # noqa: BLE001
                matched_union_box = None

        inset_x, inset_y = self._overlay_inset(bubble_box)
        overlay_box = _shrink_box_xy(bubble_box, inset_x, inset_y)
        source = "bubble_inner"
        source_detail = "bubble_inner_primary"

        if matched_union_box is not None:
            expanded_union = _expand_box(matched_union_box, page_shape, max(3, self.overlay_padding // 2))
            clipped_union = _intersect_boxes(expanded_union, bubble_box)
            if clipped_union is not None and _box_area(clipped_union) > 0:
                overlay_box = _intersect_boxes(_merge_boxes(overlay_box, clipped_union), bubble_box) or bubble_box
                source_detail = "bubble_inner_with_text_guard"

        if _box_area(overlay_box) <= 0:
            overlay_box = bubble_box
            source = "bubble_inner_fallback"
            source_detail = "bubble_inner_direct"

        return {
            "box": overlay_box,
            "source": source,
            "matched_union_box": matched_union_box,
            "basis": "bubble_inner",
            "inner_inset": [int(inset_x), int(inset_y)],
            "source_detail": source_detail,
        }

    def build_mask(self, page_shape: tuple[int, int], region: TextRegion) -> np.ndarray:
        bubble_mask = self.bubble_inner_mask(page_shape, region)
        if not np.any(bubble_mask > 0):
            region.render_mask = np.zeros(page_shape, dtype=np.uint8)
            region.mask_mode = "cover_write_mask_empty"
            region.debug["overlay_source"] = "bubble_mask_empty"
            region.debug["overlay_box_basis"] = "bubble_mask_empty"
            region.debug["overlay_source_detail"] = "bubble_mask_empty"
            region.debug["overlay_inner_inset"] = [0, 0]
            region.debug["overlay_box"] = list(region.box)
            region.debug["overlay_alpha"] = self.overlay_alpha
            region.debug["overlay_mask_ratio"] = 0.0
            region.debug["cover_write_source"] = "bubble_mask_empty"
            region.debug["cover_write_box_basis"] = "bubble_mask_empty"
            region.debug["cover_write_source_detail"] = "bubble_mask_empty"
            region.debug["cover_write_mask_ratio"] = 0.0
            region.debug["render_mask_ratio"] = 0.0
            region.debug["render_mask_bbox"] = list(region.box)
            region.debug["cover_write_box"] = list(region.box)
            return np.zeros(page_shape, dtype=np.uint8)

        overlay_choice = self.choose_overlay_box(page_shape, region, bubble_mask)
        overlay_box = tuple(int(value) for value in overlay_choice["box"])
        overlay_source = str(overlay_choice["source"])
        matched_union_box = overlay_choice.get("matched_union_box")
        mask = np.zeros(page_shape, dtype=np.uint8)
        x1, y1, x2, y2 = overlay_box
        mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, bubble_mask)

        region.render_mask = mask.copy()
        region.mask_mode = "cover_write_mask"
        region.debug["overlay_source"] = overlay_source
        region.debug["overlay_box_basis"] = str(overlay_choice.get("basis", "bubble_inner"))
        region.debug["overlay_source_detail"] = str(overlay_choice.get("source_detail", "bubble_inner_primary"))
        region.debug["overlay_inner_inset"] = list(overlay_choice.get("inner_inset", [0, 0]))
        region.debug["overlay_box"] = list(overlay_box)
        region.debug["overlay_alpha"] = self.overlay_alpha
        region.debug["overlay_mask_ratio"] = round(_masked_coverage(mask, bubble_mask), 4)
        region.debug["cover_write_source"] = overlay_source
        region.debug["cover_write_box_basis"] = str(overlay_choice.get("basis", "bubble_inner"))
        region.debug["cover_write_source_detail"] = str(overlay_choice.get("source_detail", "bubble_inner_primary"))
        region.debug["cover_write_mask_ratio"] = round(_masked_coverage(mask, bubble_mask), 4)
        region.debug["render_mask_ratio"] = region.debug["overlay_mask_ratio"]
        region.debug["render_mask_bbox"] = list(_mask_bbox(mask) or overlay_box)
        region.debug["matched_text_union_box"] = list(matched_union_box) if matched_union_box else None
        region.debug["cover_write_box"] = list(overlay_box)
        return mask

    def apply(
        self,
        image_rgb: np.ndarray,
        region: TextRegion,
        overlay_mask: np.ndarray,
        *,
        fallback_used: bool = False,
        error_message: str = "",
    ) -> np.ndarray:
        bubble_mask = self.bubble_inner_mask(image_rgb.shape[:2], region)
        if not np.any(overlay_mask > 0):
            overlay_mask = bubble_mask
            region.debug["overlay_source"] = "bubble_inner_fallback"
            region.debug["overlay_box_basis"] = region.debug.get("overlay_box_basis", "bubble_inner")
            region.debug["overlay_source_detail"] = "bubble_mask_apply_fallback"
            region.debug["overlay_box"] = list(_mask_bbox(bubble_mask) or region.box)

        overlay_mask = cv2.bitwise_and(overlay_mask, bubble_mask)
        overlay_box = _mask_bbox(overlay_mask)
        if overlay_box is None:
            overlay_box = _mask_bbox(bubble_mask) or region.box
        x1, y1, x2, y2 = overlay_box
        alpha_map = (overlay_mask.astype(np.float32) / 255.0) * self.overlay_alpha
        if self.overlay_edge_blur > 0:
            k = max(1, int(self.overlay_edge_blur))
            if k % 2 == 0:
                k += 1
            alpha_map = cv2.GaussianBlur(alpha_map, (k, k), 0)
            alpha_map = np.clip(alpha_map, 0.0, self.overlay_alpha)

        base = image_rgb.astype(np.float32)
        result = image_rgb.copy()
        roi_alpha = alpha_map[y1:y2, x1:x2]
        roi_base = base[y1:y2, x1:x2]
        roi_white = np.full_like(roi_base, 255, dtype=np.float32)
        roi_blended = roi_base * (1.0 - roi_alpha[:, :, None]) + roi_white * roi_alpha[:, :, None]
        result[y1:y2, x1:x2] = np.clip(roi_blended, 0, 255).astype(np.uint8)

        region.inpaint_mode = "translucent_white_overlay"
        region.debug["fallback_used"] = fallback_used
        region.debug["inpaint_error"] = error_message
        region.debug["inpaint_backend"] = "overlay_fallback" if fallback_used else "translucent_white_overlay"
        region.debug["inpaint_fallback_backend"] = "translucent_white_overlay" if fallback_used else ""
        region.debug["inpaint_model"] = "overlay_fallback" if fallback_used else "overlay"
        region.debug["inpaint_device"] = ""
        region.debug["overlay_box"] = list(overlay_box)
        region.debug["overlay_mask_ratio"] = round(_masked_coverage(overlay_mask, bubble_mask), 4)
        region.debug["cover_write_box"] = list(overlay_box)
        return result

    def runtime_summary(self) -> dict[str, object]:
        return {
            "backend_name": "translucent_white_overlay",
            "backend_ready": True,
            "overlay_alpha": DEFAULT_OVERLAY_ALPHA,
            "overlay_padding": DEFAULT_OVERLAY_PADDING,
            "overlay_edge_blur": DEFAULT_OVERLAY_EDGE_BLUR,
            "fallback_fill_mode": self.config.inpaint.failure_fill_mode,
        }


class LamaCleanerInpaintBackend:
    def __init__(self, config: AppConfig, fallback_backend: OverlayInpaintBackend):
        self.config = config
        self.fallback_backend = fallback_backend
        self.model_dir = Path(str(config.inpaint.model_dir or "")).resolve()
        self.requested_device = str(config.inpaint.device or config.device or "cuda").lower()
        self.manager_cache: dict[str, object] = {}
        self.manager_errors: dict[str, str] = {}
        self.manager_devices: dict[str, str] = {}

    @staticmethod
    def _is_outside_text_region(region: TextRegion) -> bool:
        return bool(region.debug.get("outside_bubble", False))

    def _select_model(self, region: TextRegion) -> str:
        if self._is_outside_text_region(region):
            return str(self.config.inpaint.outside_text_model or "ldm")
        return str(self.config.inpaint.bubble_model or "lama")

    def _ensure_torch_home(self) -> None:
        if self.model_dir:
            os.environ["TORCH_HOME"] = str(self.model_dir)

    def _resolve_device(self) -> tuple[torch.device | None, str, str]:
        if self.requested_device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda"), "cuda", ""
            return None, "", "cuda_unavailable"
        if self.requested_device == "cpu":
            return torch.device("cpu"), "cpu", ""
        if self.requested_device == "mps":
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return torch.device("mps"), "mps", ""
            return None, "", "mps_unavailable"
        return None, "", f"unsupported_device:{self.requested_device}"

    def _make_lama_config(self):
        from lama_cleaner.schema import Config as LamaCleanerConfig

        return LamaCleanerConfig(
            ldm_steps=int(self.config.inpaint.ldm_steps),
            ldm_sampler=str(self.config.inpaint.ldm_sampler),
            hd_strategy=str(self.config.inpaint.hd_strategy),
            hd_strategy_crop_margin=int(self.config.inpaint.hd_strategy_crop_margin),
            hd_strategy_crop_trigger_size=int(self.config.inpaint.hd_strategy_crop_trigger_size),
            hd_strategy_resize_limit=int(self.config.inpaint.hd_strategy_resize_limit),
        )

    def _get_manager(self, model_name: str) -> tuple[object | None, str, str]:
        if model_name in self.manager_cache:
            return self.manager_cache[model_name], "", self.manager_devices.get(model_name, "")
        if model_name in self.manager_errors:
            return None, self.manager_errors[model_name], self.manager_devices.get(model_name, "")

        device, resolved_device, device_error = self._resolve_device()
        if device is None:
            self.manager_errors[model_name] = device_error
            self.manager_devices[model_name] = resolved_device
            return None, device_error, resolved_device

        self._ensure_torch_home()
        try:
            from lama_cleaner.model_manager import ModelManager

            manager = ModelManager(name=model_name, device=device)
            self.manager_cache[model_name] = manager
            self.manager_devices[model_name] = resolved_device
            return manager, "", resolved_device
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            self.manager_errors[model_name] = error_message
            self.manager_devices[model_name] = resolved_device
            if resolved_device == "cuda":
                torch.cuda.empty_cache()
            return None, error_message, resolved_device

    def _apply_fallback(
        self,
        image_rgb: np.ndarray,
        region: TextRegion,
        mask: np.ndarray,
        *,
        error_message: str,
        requested_model: str,
        resolved_device: str,
    ) -> np.ndarray:
        result = self.fallback_backend.apply(
            image_rgb,
            region,
            mask,
            fallback_used=True,
            error_message=error_message,
        )
        region.debug["inpaint_model"] = "overlay_fallback"
        region.debug["inpaint_requested_model"] = requested_model
        region.debug["inpaint_requested_device"] = self.requested_device
        region.debug["inpaint_runtime_device"] = resolved_device
        region.debug["inpaint_model_dir"] = str(self.model_dir)
        region.debug["inpaint_region_scope"] = "outside_text" if self._is_outside_text_region(region) else "bubble_inner"
        region.debug["inpaint_error"] = error_message
        region.debug["fallback_used"] = True
        return result

    def apply(self, image_rgb: np.ndarray, region: TextRegion, mask: np.ndarray) -> np.ndarray:
        requested_model = self._select_model(region)
        region.debug["inpaint_requested_model"] = requested_model
        region.debug["inpaint_requested_device"] = self.requested_device
        region.debug["inpaint_model_dir"] = str(self.model_dir)
        region.debug["inpaint_region_scope"] = "outside_text" if self._is_outside_text_region(region) else "bubble_inner"

        if mask is None or mask.size == 0 or not np.any(mask > 0):
            return self._apply_fallback(
                image_rgb,
                region,
                np.zeros(image_rgb.shape[:2], dtype=np.uint8),
                error_message="empty_inpaint_mask",
                requested_model=requested_model,
                resolved_device="",
            )

        manager, load_error, resolved_device = self._get_manager(requested_model)
        if manager is None:
            return self._apply_fallback(
                image_rgb,
                region,
                mask,
                error_message=load_error or "lama_cleaner_manager_unavailable",
                requested_model=requested_model,
                resolved_device=resolved_device,
            )

        try:
            result_bgr = manager(image_rgb, mask, self._make_lama_config())
            if result_bgr is None or result_bgr.shape[:2] != image_rgb.shape[:2]:
                raise RuntimeError("lama_cleaner_invalid_result_shape")
            if result_bgr.dtype != np.uint8:
                result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            region.inpaint_mode = f"lama_cleaner_{requested_model}"
            region.debug["inpaint_backend"] = "lama_cleaner"
            region.debug["inpaint_fallback_backend"] = ""
            region.debug["inpaint_model"] = requested_model
            region.debug["inpaint_device"] = resolved_device
            region.debug["inpaint_runtime_device"] = resolved_device
            region.debug["inpaint_error"] = ""
            region.debug["fallback_used"] = False
            region.debug["cover_write_box"] = list(_mask_bbox(mask) or region.box)
            return result_rgb
        except Exception as exc:  # noqa: BLE001
            if resolved_device == "cuda":
                torch.cuda.empty_cache()
            return self._apply_fallback(
                image_rgb,
                region,
                mask,
                error_message=str(exc),
                requested_model=requested_model,
                resolved_device=resolved_device,
            )

    def runtime_summary(self) -> dict[str, object]:
        bubble_model = str(self.config.inpaint.bubble_model or "lama")
        outside_model = str(self.config.inpaint.outside_text_model or "ldm")
        return {
            "backend_name": "lama_cleaner",
            "backend_ready": True,
            "requested_device": self.requested_device,
            "cuda_available": torch.cuda.is_available(),
            "model_dir": str(self.model_dir),
            "torch_home": str(self.model_dir),
            "bubble_model": bubble_model,
            "outside_text_model": outside_model,
            "bubble_model_loaded": bubble_model in self.manager_cache,
            "outside_text_model_loaded": outside_model in self.manager_cache,
            "model_load_errors": dict(self.manager_errors),
            "loaded_model_devices": dict(self.manager_devices),
            "hd_strategy": str(self.config.inpaint.hd_strategy),
            "hd_strategy_crop_margin": int(self.config.inpaint.hd_strategy_crop_margin),
            "hd_strategy_crop_trigger_size": int(self.config.inpaint.hd_strategy_crop_trigger_size),
            "hd_strategy_resize_limit": int(self.config.inpaint.hd_strategy_resize_limit),
            "ldm_steps": int(self.config.inpaint.ldm_steps),
            "ldm_sampler": str(self.config.inpaint.ldm_sampler),
            "fallback_backend": str(self.config.inpaint.fallback_backend),
            "fallback_fill_mode": self.config.inpaint.failure_fill_mode,
        }


class InpaintService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.mask_backend = OverlayInpaintBackend(config)
        self.primary_backend = (
            LamaCleanerInpaintBackend(config, self.mask_backend)
            if str(config.inpaint.backend or "").lower() == "lama_cleaner"
            else self.mask_backend
        )

    def prepare_page(self, image_rgb: np.ndarray, doc: DocumentContext | None = None) -> None:
        if doc is not None:
            doc.debug["inpaint_mask_backend"] = self.runtime_summary()

    def build_mask(self, image_rgb: np.ndarray, region: TextRegion) -> np.ndarray:
        return self.mask_backend.build_mask(image_rgb.shape[:2], region)

    def apply(self, image_rgb: np.ndarray, region: TextRegion, mask: np.ndarray) -> np.ndarray:
        return self.primary_backend.apply(image_rgb, region, mask)

    def runtime_summary(self) -> dict[str, object]:
        if self.primary_backend is self.mask_backend:
            return self.mask_backend.runtime_summary()
        summary = dict(self.primary_backend.runtime_summary())
        summary["fallback_backend_summary"] = self.mask_backend.runtime_summary()
        return summary
