from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Box = tuple[int, int, int, int]
Polygon = list[list[int]]


@dataclass
class Bubble:
    bbox: Box
    polygon: Polygon = field(default_factory=list)
    score: float = 0.0
    source: str = "edge"


@dataclass
class LayoutRegion:
    box: Box
    label: str
    score: float = 0.0
    polygon: list[list[int]] = field(default_factory=list)
    reading_order: int = -1
    source: str = "layout"


@dataclass
class DocumentContext:
    image_shape: tuple[int, int] | None = None
    layout_regions: list[LayoutRegion] = field(default_factory=list)
    bubbles: list[Bubble] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextRegion:
    index: int
    box: Box
    polygon: list[list[int]] = field(default_factory=list)
    detector: str = "easyocr"
    detector_text_hint: str = ""
    easy_text: str = ""
    easy_conf: float = 0.0
    structure_score: float = 0.0
    region_type: str = "unknown"
    bubble_index: int = -1
    bubble_bbox: Box | None = None
    bubble_polygon: Polygon = field(default_factory=list)
    bubble_score: float = 0.0
    layout_label: str = ""
    reading_order: int = -1
    ocr_primary_text: str = ""
    ocr_fallback_text: str = ""
    ocr_text: str = ""
    translation: str = ""
    skip_reason: str = ""
    mask_mode: str = ""
    inpaint_mode: str = ""
    render_rect: Box | None = None
    render_mask: np.ndarray | None = None
    prefill_text: str = ""
    prefill_source: str = ""
    style: dict[str, Any] = field(default_factory=dict)
    fusion_sources: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "box": list(self.box),
            "polygon": self.polygon,
            "detector": self.detector,
            "detector_text_hint": self.detector_text_hint,
            "easy_text": self.easy_text,
            "easy_conf": round(float(self.easy_conf), 4),
            "structure_score": round(float(self.structure_score), 4),
            "region_type": self.region_type,
            "bubble_index": self.bubble_index,
            "bubble_bbox": list(self.bubble_bbox) if self.bubble_bbox else None,
            "bubble_polygon": self.bubble_polygon,
            "bubble_score": round(float(self.bubble_score), 4),
            "layout_label": self.layout_label,
            "reading_order": self.reading_order,
            "ocr_primary_text": self.ocr_primary_text,
            "ocr_fallback_text": self.ocr_fallback_text,
            "ocr_text": self.ocr_text,
            "translation": self.translation,
            "skip_reason": self.skip_reason,
            "mask_mode": self.mask_mode,
            "inpaint_mode": self.inpaint_mode,
            "render_rect": list(self.render_rect) if self.render_rect else None,
            "prefill_text": self.prefill_text,
            "prefill_source": self.prefill_source,
            "style": self.style,
            "fusion_sources": list(self.fusion_sources),
            **self.debug,
        }
