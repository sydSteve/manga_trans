from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import AppConfig
from .detection import TextDetector
from .inpaint import InpaintService
from .models import DocumentContext, TextRegion
from .ocr import OCRService
from .render import render_translation
from .style import StyleService
from .translation import DeepSeekBatchTranslator
from .utils import ensure_parent, list_images, normalize_text


RENDERABLE_OCR_VALIDITY = {"valid", "short_japanese"}
OUTSIDE_TEXT_SKIP_MAX_LENGTH = 8


class MangaTranslationPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = TextDetector(config)
        self.ocr = OCRService(config)
        self.translator = DeepSeekBatchTranslator(config)
        self.inpaint = InpaintService(config)
        self.styler = StyleService(config)
        self.last_region_type_counts: dict[str, int] = {}
        self.last_text_region_count = 0
        self.last_inside_bubble_text_region_count = 0
        self.last_bubble_region_count = 0
        self.last_outside_bubble_region_count = 0
        self.last_outside_bubble_translated_count = 0
        self.last_outside_bubble_short_skip_count = 0

    def process_directory(self, input_dir: Path, output_dir: Path, debug: bool = False) -> int:
        image_files = list_images(input_dir)
        success_count = 0
        debug_root = output_dir if debug and self.config.debug.enabled else None
        for image_path in image_files:
            relative_path = image_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            ensure_parent(output_path)
            if self.process_image(image_path, output_path, debug_root):
                success_count += 1
        return success_count

    def process_image(self, image_path: Path, output_path: Path, debug_root: Path | None) -> bool:
        pil_image = Image.open(image_path).convert("RGB")
        pil_image = ImageOps.exif_transpose(pil_image)
        image_rgb = np.array(pil_image)
        doc = DocumentContext(image_shape=image_rgb.shape[:2])

        text_regions = self.detector.detect(image_rgb, doc)
        self.inpaint.prepare_page(image_rgb, doc)
        regions = self._build_pipeline_regions(doc, text_regions)

        for region in regions:
            text = self.ocr.recognize(image_rgb, region)
            validity = str(region.debug.get("ocr_text_validity", ""))
            is_outside_text = bool(region.debug.get("outside_bubble", False))
            region.debug["bubble_translation_policy"] = "outside_text_pending" if is_outside_text else "bubble_pending"
            if normalize_text(text) and validity in RENDERABLE_OCR_VALIDITY:
                if is_outside_text:
                    text_count = len(normalize_text(text))
                    region.debug["short_text_count"] = text_count
                    if text_count <= OUTSIDE_TEXT_SKIP_MAX_LENGTH:
                        region.debug["short_text_skip"] = True
                        region.debug["bubble_translation_policy"] = "outside_text_short_skip"
                        region.skip_reason = region.skip_reason or "outside_text_too_short"
                        region.debug["outside_text_ocr_text"] = text
                        region.ocr_text = ""
                    else:
                        region.debug["short_text_skip"] = False
                        region.debug["bubble_translation_policy"] = "outside_text_primary"
                else:
                    region.debug["bubble_translation_policy"] = "bubble_primary"
            else:
                if is_outside_text:
                    region.debug["short_text_skip"] = False
                if not normalize_text(text):
                    region.skip_reason = region.skip_reason or "ocr_empty"
                else:
                    region.skip_reason = region.skip_reason or f"ocr_rejected:{validity or 'invalid'}"
                region.debug["bubble_ocr_text"] = text
                region.ocr_text = ""

        self.last_region_type_counts = dict(Counter(region.region_type for region in regions))
        self.last_text_region_count = len(text_regions)
        self.last_inside_bubble_text_region_count = sum(1 for region in text_regions if region.bubble_index >= 0)
        self.last_bubble_region_count = sum(1 for region in regions if not region.debug.get("outside_bubble", False))
        self.last_outside_bubble_region_count = sum(1 for region in regions if region.debug.get("outside_bubble", False))
        self.last_outside_bubble_short_skip_count = sum(
            1 for region in regions if region.debug.get("outside_bubble", False) and region.debug.get("short_text_skip", False)
        )
        self.translator.translate_regions(regions)
        self.last_outside_bubble_translated_count = sum(
            1
            for region in regions
            if region.debug.get("outside_bubble", False) and bool(normalize_text(region.translation))
        )

        working_rgb = image_rgb.copy()
        working_pil = Image.fromarray(working_rgb)
        for region in regions:
            if region.region_type not in {"dialogue_bubble", "narration_box"}:
                continue
            if not region.translation:
                continue

            mask = self.inpaint.build_mask(image_rgb, region)
            self.styler.analyze(image_rgb, region, mask)
            candidate_rgb = self.inpaint.apply(working_rgb, region, mask)
            candidate_pil = Image.fromarray(candidate_rgb)
            rendered = render_translation(candidate_pil, region, self.config)
            if rendered:
                working_pil = candidate_pil
                working_rgb = np.array(working_pil)
            elif not region.skip_reason:
                region.skip_reason = "render_failed_keep_original"

        ensure_parent(output_path)
        working_pil.save(output_path)

        if debug_root:
            self._write_debug(output_path, debug_root, image_path, doc, regions, text_regions)
        return True

    def _build_pipeline_regions(self, doc: DocumentContext, text_regions: list[TextRegion]) -> list[TextRegion]:
        bubble_regions = self._build_bubble_regions(doc, text_regions)
        outside_text_regions = self._build_outside_text_regions(text_regions)
        regions = bubble_regions + outside_text_regions
        for index, region in enumerate(regions):
            region.index = index
        doc.debug["outside_bubble_region_count"] = len(outside_text_regions)
        doc.debug["pipeline_region_count"] = len(regions)
        return regions

    def _build_bubble_regions(self, doc: DocumentContext, text_regions: list[TextRegion]) -> list[TextRegion]:
        matched_indices_by_bubble: dict[int, list[int]] = {}
        text_region_by_index: dict[int, TextRegion] = {}
        for region in text_regions:
            text_region_by_index[region.index] = region
            if region.bubble_index < 0:
                continue
            matched_indices_by_bubble.setdefault(region.bubble_index, []).append(region.index)

        bubble_regions: list[TextRegion] = []
        for bubble_index, bubble in enumerate(doc.bubbles):
            x1, y1, x2, y2 = bubble.bbox
            matched_indices = list(matched_indices_by_bubble.get(bubble_index, []))
            matched_union_box = self._union_text_boxes([text_region_by_index[index].box for index in matched_indices if index in text_region_by_index])
            bubble_region = TextRegion(
                index=bubble_index,
                box=bubble.bbox,
                polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                detector="bubble_primary",
                easy_conf=float(bubble.score),
                structure_score=float(bubble.score),
                region_type="dialogue_bubble",
                bubble_index=bubble_index,
                bubble_bbox=bubble.bbox,
                bubble_polygon=list(bubble.polygon),
                bubble_score=float(bubble.score),
                fusion_sources=["bubble_primary"],
                debug={
                    "translation_scope": "bubble",
                    "bubble_source": bubble.source,
                    "matched_text_region_indices": matched_indices,
                    "matched_text_union_box": list(matched_union_box) if matched_union_box else None,
                },
            )
            bubble_regions.append(bubble_region)
        doc.debug["text_region_count"] = len(text_regions)
        doc.debug["inside_bubble_text_region_count"] = sum(1 for region in text_regions if region.bubble_index >= 0)
        doc.debug["bubble_region_count"] = len(bubble_regions)
        return bubble_regions

    @staticmethod
    def _build_outside_text_regions(text_regions: list[TextRegion]) -> list[TextRegion]:
        outside_regions: list[TextRegion] = []
        for source_region in text_regions:
            if source_region.bubble_index >= 0:
                continue
            outside_region = TextRegion(
                index=source_region.index,
                box=source_region.box,
                polygon=list(source_region.polygon),
                detector=source_region.detector,
                detector_text_hint=source_region.detector_text_hint,
                easy_text=source_region.easy_text,
                easy_conf=float(source_region.easy_conf),
                structure_score=float(source_region.structure_score),
                region_type="narration_box",
                bubble_index=-1,
                bubble_bbox=None,
                bubble_polygon=[],
                bubble_score=0.0,
                layout_label=source_region.layout_label,
                reading_order=source_region.reading_order,
                fusion_sources=list(source_region.fusion_sources),
                debug={
                    **dict(source_region.debug),
                    "translation_scope": "outside_text",
                    "source_text_region_index": source_region.index,
                    "outside_bubble": True,
                    "render_policy": "outside_text_box",
                },
            )
            outside_regions.append(outside_region)
        return outside_regions

    @staticmethod
    def _union_text_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
        if not boxes:
            return None
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _write_debug(
        self,
        output_path: Path,
        debug_root: Path,
        image_path: Path,
        doc: DocumentContext,
        regions: list[TextRegion],
        text_regions: list[TextRegion],
    ) -> None:
        debug_dir = debug_root / self.config.debug.output_dirname
        debug_file = debug_dir / output_path.relative_to(debug_root)
        debug_file = debug_file.with_suffix(".json")
        ensure_parent(debug_file)
        debug_payload = {
            "source_image": str(image_path),
            "output_image": str(output_path),
            "bubble_region_count": sum(1 for region in regions if not region.debug.get("outside_bubble", False)),
            "outside_bubble_region_count": sum(1 for region in regions if region.debug.get("outside_bubble", False)),
            "text_region_count": len(text_regions),
            "inside_bubble_text_region_count": sum(1 for region in text_regions if region.bubble_index >= 0),
            "region_count": len(regions),
            "region_type_counts": self.last_region_type_counts,
            "runtime_summary": self.runtime_summary(),
            "detection_postprocess": dict(doc.debug.get("detection_postprocess", {})),
            "filtered_regions": list(doc.debug.get("filtered_regions", [])),
            "bubble_regions": [region.to_debug_dict() for region in regions if not region.debug.get("outside_bubble", False)],
            "outside_text_regions": [region.to_debug_dict() for region in regions if region.debug.get("outside_bubble", False)],
            "text_regions": [region.to_debug_dict() for region in text_regions],
            "regions": [region.to_debug_dict() for region in regions],
        }
        debug_file.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def runtime_summary(self) -> dict[str, object]:
        detector_summary = self.detector.runtime_summary()
        ocr_summary = self.ocr.runtime_summary()
        inpaint_summary = self.inpaint.runtime_summary()
        style_summary = self.styler.runtime_summary()
        translator_summary = self.translator.runtime_summary()
        return {
            "device": self.config.device,
            "detector_backend": detector_summary.get("backend_name", ""),
            "bubble_backend": detector_summary.get("bubble_backend", ""),
            "ocr_backend": ocr_summary.get("backend_name", ""),
            "inpaint_backend": inpaint_summary.get("backend_name", ""),
            "style_backend": style_summary.get("backend_name", ""),
            "translator_enabled": translator_summary.get("enabled", False),
            "translator_api_requested": translator_summary.get("api_requested", False),
            "translator_api_failure_count": translator_summary.get("api_failure_count", 0),
            "translator_fallback_to_ocr_count": translator_summary.get("fallback_to_ocr_count", 0),
            "translator_failure_reasons": translator_summary.get("failure_reasons", {}),
            "bubble_region_count": self.last_bubble_region_count,
            "outside_bubble_region_count": self.last_outside_bubble_region_count,
            "outside_bubble_translated_count": self.last_outside_bubble_translated_count,
            "outside_bubble_short_skip_count": self.last_outside_bubble_short_skip_count,
            "text_region_count": self.last_text_region_count,
            "inside_bubble_text_region_count": self.last_inside_bubble_text_region_count,
            "model_hits": {
                "detector": bool(detector_summary.get("model_path_hit", False)),
                "bubble": bool(detector_summary.get("bubble_model_path_hit", False)),
                "ocr": bool(ocr_summary.get("model_path_hit", False)),
                "inpaint": bool(inpaint_summary.get("backend_ready", False)),
                "style": bool(style_summary.get("model_path_hit", False)),
            },
            "region_type_counts": self.last_region_type_counts,
        }
