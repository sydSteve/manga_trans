from __future__ import annotations

import cv2
import easyocr
import numpy as np

from .comic_text_and_bubble_detector_backend import ComicTextAndBubbleDetectorBackend
from .config import AppConfig
from .models import Bubble, DocumentContext, TextRegion
from .speech_bubble_backend import SpeechBubbleSegmentationBackend
from .utils import box_center, clamp_box, intersection_ratio, normalize_text


LOW_SCORE_EASYOCR_THRESHOLD = 0.05
ADJACENT_OVERLAP_RATIO = 0.45
ADJACENT_GAP_MIN = 18
ADJACENT_GAP_SCALE = 0.75


def detect_bubbles_by_edges(image_bgr: np.ndarray) -> list[Bubble]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles: list[Bubble] = []
    height, width = gray.shape
    image_area = height * width
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1200 or area > image_area * 0.45:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 24 or h < 24:
            continue
        aspect = w / max(1.0, float(h))
        if not 0.18 <= aspect <= 5.5:
            continue
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        inner_pixels = gray[mask > 0]
        if inner_pixels.size < 50:
            continue
        whiteness = float(np.mean(inner_pixels))
        variance = float(np.var(inner_pixels))
        score = 0.0
        if whiteness >= 180:
            score += 0.5
        if variance <= 900:
            score += 0.35
        if 0.4 <= aspect <= 2.8:
            score += 0.2
        if score < 0.35:
            continue
        approx = cv2.approxPolyDP(contour, 0.012 * cv2.arcLength(contour, True), True)
        polygon = [[int(point[0][0]), int(point[0][1])] for point in approx] if len(approx) >= 3 else []
        if not polygon:
            polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        bubbles.append(Bubble(bbox=(x, y, x + w, y + h), polygon=polygon, score=score, source="edge"))

    bubbles.sort(key=lambda bubble: bubble.score, reverse=True)
    return bubbles


def _polygon_to_box(polygon: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _find_bubble(box: tuple[int, int, int, int], bubbles: list[Bubble]) -> tuple[int, Bubble | None, float]:
    best_index = -1
    best_bubble = None
    best_score = 0.0
    center = box_center(box)
    for index, bubble in enumerate(bubbles):
        x1, y1, x2, y2 = bubble.bbox
        inside = x1 <= center[0] <= x2 and y1 <= center[1] <= y2
        overlap = intersection_ratio(box, bubble.bbox)
        score = max(overlap, 0.35 if inside else 0.0) + bubble.score * 0.25
        if score > best_score:
            best_index = index
            best_bubble = bubble
            best_score = score
    if best_bubble is not None and not _is_valid_bubble_match(box, best_bubble):
        return -1, None, 0.0
    return best_index, best_bubble, best_score


def _is_valid_bubble_match(box: tuple[int, int, int, int], bubble: Bubble) -> bool:
    overlap = intersection_ratio(box, bubble.bbox)
    if overlap >= 0.2:
        return True

    cx, cy = box_center(box)
    x1, y1, x2, y2 = bubble.bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _box_similarity(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    return max(intersection_ratio(a, b), intersection_ratio(b, a))


def _interval_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _interval_gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    if a[1] < b[0]:
        return max(0, b[0] - a[1])
    if b[1] < a[0]:
        return max(0, a[0] - b[1])
    return 0


def _projection_overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    overlap = _interval_overlap(a, b)
    base = max(1, min(a[1] - a[0], b[1] - b[0]))
    return overlap / base


def _region_sort_key(region: TextRegion) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = region.box
    return y1, x1, y2, x2


def _region_trace_ids(region: TextRegion) -> list[int]:
    trace = region.debug.get("merged_from")
    if isinstance(trace, list) and trace:
        values = []
        for item in trace:
            try:
                values.append(int(item))
            except Exception:  # noqa: BLE001
                continue
        if values:
            return sorted(set(values))
    return [int(region.index)]


def _merge_unique_bubbles(
    primary: list[Bubble],
    extra: list[Bubble],
    duplicate_threshold: float = 0.75,
) -> list[Bubble]:
    merged: list[Bubble] = []
    for bubble in sorted(primary, key=lambda item: item.score, reverse=True):
        if any(_box_similarity(bubble.bbox, existing.bbox) >= duplicate_threshold for existing in merged):
            continue
        merged.append(bubble)

    for bubble in sorted(extra, key=lambda item: item.score, reverse=True):
        if any(_box_similarity(bubble.bbox, existing.bbox) >= duplicate_threshold for existing in merged):
            continue
        merged.append(bubble)

    merged.sort(key=lambda item: item.score, reverse=True)
    return merged


def _match_bubbles(regions: list[TextRegion], bubbles: list[Bubble]) -> None:
    for region in regions:
        bubble_index, bubble, bubble_score = _find_bubble(region.box, bubbles)
        region.bubble_index = bubble_index
        region.bubble_bbox = bubble.bbox if bubble else None
        region.bubble_polygon = list(bubble.polygon) if bubble else []
        region.bubble_score = bubble_score
        if bubble:
            region.debug["bubble_source"] = bubble.source


def _should_merge_adjacent_regions(a: TextRegion, b: TextRegion) -> bool:
    if a.bubble_index < 0 or a.bubble_index != b.bubble_index:
        return False

    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    horizontal_overlap = _projection_overlap_ratio((ax1, ax2), (bx1, bx2))
    vertical_overlap = _projection_overlap_ratio((ay1, ay2), (by1, by2))
    gap_x = _interval_gap((ax1, ax2), (bx1, bx2))
    gap_y = _interval_gap((ay1, ay2), (by1, by2))
    vertical_gap_limit = max(ADJACENT_GAP_MIN, int(min(ay2 - ay1, by2 - by1) * ADJACENT_GAP_SCALE))
    horizontal_gap_limit = max(ADJACENT_GAP_MIN, int(min(ax2 - ax1, bx2 - bx1) * ADJACENT_GAP_SCALE))

    if horizontal_overlap >= ADJACENT_OVERLAP_RATIO and gap_y <= vertical_gap_limit:
        return True
    if vertical_overlap >= ADJACENT_OVERLAP_RATIO and gap_x <= horizontal_gap_limit:
        return True
    return False


def _merge_region_group(regions: list[TextRegion]) -> TextRegion:
    ordered = sorted(regions, key=_region_sort_key)
    first = ordered[0]
    x1 = min(region.box[0] for region in ordered)
    y1 = min(region.box[1] for region in ordered)
    x2 = max(region.box[2] for region in ordered)
    y2 = max(region.box[3] for region in ordered)
    merged_from = sorted({item for region in ordered for item in _region_trace_ids(region)})
    detector_hint_parts = [
        normalize_text(region.detector_text_hint or region.easy_text) for region in ordered if normalize_text(region.detector_text_hint or region.easy_text)
    ]
    easy_text_parts = [normalize_text(region.easy_text) for region in ordered if normalize_text(region.easy_text)]
    detector = (
        "comic_text_and_bubble_detector"
        if any(region.detector == "comic_text_and_bubble_detector" for region in ordered)
        else first.detector
    )
    debug: dict[str, object] = {}
    for key in ("detector_backend", "structure_label", "bubble_source"):
        for region in ordered:
            if key in region.debug and region.debug.get(key) not in ("", None):
                debug[key] = region.debug[key]
                break
    debug["merged_from"] = merged_from
    debug["merged_child_count"] = len(merged_from)
    debug["merge_mode"] = "adjacent_same_bubble"

    merged = TextRegion(
        index=first.index,
        box=(x1, y1, x2, y2),
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        detector=detector,
        detector_text_hint=" ".join(detector_hint_parts),
        easy_text=" ".join(easy_text_parts),
        easy_conf=max(region.easy_conf for region in ordered),
        structure_score=max(region.structure_score for region in ordered),
        region_type=first.region_type,
        bubble_index=first.bubble_index,
        bubble_bbox=first.bubble_bbox,
        bubble_polygon=list(first.bubble_polygon),
        bubble_score=max(region.bubble_score for region in ordered),
        layout_label=next((region.layout_label for region in ordered if region.layout_label), first.layout_label),
        reading_order=next((region.reading_order for region in ordered if region.reading_order >= 0), first.reading_order),
        fusion_sources=sorted({source for region in ordered for source in (region.fusion_sources or [])}),
        debug=debug,
    )
    return merged


class TextDetector:
    def __init__(self, config: AppConfig):
        self.config = config
        self.reader = easyocr.Reader(config.ocr.language, gpu=config.device == "cuda")
        self.structure_backend = ComicTextAndBubbleDetectorBackend(config)
        self.bubble_backend = SpeechBubbleSegmentationBackend(config)
        self.last_bubble_runtime = self._compose_bubble_runtime([], [], [])

    def detect(self, image_rgb: np.ndarray, doc: DocumentContext | None = None) -> list[TextRegion]:
        image_bgr = image_rgb[:, :, ::-1].copy()
        bubbles = self._detect_bubbles(image_bgr)
        if doc is not None:
            doc.bubbles = bubbles
            doc.debug["bubble_runtime"] = dict(self.last_bubble_runtime)

        primary_regions = self._detect_with_structure_backend(image_rgb, bubbles)
        if primary_regions:
            if self.config.detector.enable_easyocr_fusion:
                primary_regions = self._merge_with_easyocr_hints(image_rgb, bubbles, primary_regions)
            primary_regions = self._finalize_regions(primary_regions, bubbles, doc)
            for region in primary_regions:
                region.debug["detector_backend"] = "comic_text_and_bubble_detector"
            return primary_regions

        fallback_regions = self._detect_with_easyocr(image_rgb, bubbles)
        fallback_regions = self._finalize_regions(fallback_regions, bubbles, doc)
        for region in fallback_regions:
            region.debug["detector_backend"] = "easyocr_fallback"
            region.debug["detector_error_code"] = self.structure_backend.status.load_error_code
            region.debug["detector_error_message"] = self.structure_backend.status.load_error_message
        return fallback_regions

    def _detect_with_structure_backend(self, image_rgb: np.ndarray, bubbles: list[Bubble]) -> list[TextRegion]:
        detections = self.structure_backend.detect(image_rgb)
        if not detections:
            return []

        text_keywords = tuple(keyword.lower() for keyword in self.config.detector.text_label_keywords)
        bubble_keywords = tuple(keyword.lower() for keyword in self.config.detector.bubble_label_keywords)
        regions: list[TextRegion] = []

        for detection in detections:
            label = str(detection.get("label", "")).lower()
            if any(keyword in label for keyword in bubble_keywords):
                continue
            if not any(keyword in label for keyword in text_keywords):
                continue

            box = clamp_box(tuple(int(num) for num in detection.get("box", [0, 0, 0, 0])), image_rgb.shape)
            x1, y1, x2, y2 = box
            region = TextRegion(
                index=len(regions),
                box=box,
                polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                detector="comic_text_and_bubble_detector",
                structure_score=float(detection.get("score", 0.0)),
                easy_conf=float(detection.get("score", 0.0)),
                fusion_sources=["comic_text_and_bubble_detector"],
                layout_label=label,
            )
            region.debug["structure_label"] = label
            regions.append(region)

        _match_bubbles(regions, bubbles)
        return regions

    def _build_easyocr_region(
        self,
        index: int,
        item: list,
        image_shape: tuple[int, int, int],
        bubbles: list[Bubble],
    ) -> TextRegion:
        polygon = [[int(point[0]), int(point[1])] for point in item[0]]
        box = clamp_box(_polygon_to_box(polygon), image_shape)
        easy_text = normalize_text(item[1] if len(item) > 1 else "")
        easy_conf = float(item[2] if len(item) > 2 else 0.0)
        bubble_index, bubble, bubble_score = _find_bubble(box, bubbles)
        return TextRegion(
            index=index,
            box=box,
            polygon=polygon,
            detector="easyocr",
            detector_text_hint=easy_text,
            easy_text=easy_text,
            easy_conf=easy_conf,
            structure_score=easy_conf,
            bubble_index=bubble_index,
            bubble_bbox=bubble.bbox if bubble else None,
            bubble_score=bubble_score,
            fusion_sources=["easyocr"],
        )

    def _detect_with_easyocr(self, image_rgb: np.ndarray, bubbles: list[Bubble]) -> list[TextRegion]:
        results = self.reader.readtext(
            image_rgb,
            text_threshold=0.5,
            low_text=0.2,
            link_threshold=0.35,
            paragraph=False,
            mag_ratio=1.4,
        )
        regions: list[TextRegion] = []
        for index, item in enumerate(results):
            region = self._build_easyocr_region(index, item, image_rgb.shape, bubbles)
            if self._should_keep_region(region):
                regions.append(region)
        return regions

    def _finalize_regions(
        self,
        regions: list[TextRegion],
        bubbles: list[Bubble],
        doc: DocumentContext | None,
    ) -> list[TextRegion]:
        input_count = len(regions)
        merged = self._merge_adjacent_regions(regions)
        _match_bubbles(merged, bubbles)
        kept, filtered = self._filter_low_score_regions(merged)
        for index, region in enumerate(kept):
            region.index = index

        if doc is not None:
            doc.debug["filtered_regions"] = filtered
            doc.debug["detection_postprocess"] = {
                "input_region_count": input_count,
                "merged_region_count": len(merged),
                "final_region_count": len(kept),
                "filtered_region_count": len(filtered),
                "low_score_threshold": LOW_SCORE_EASYOCR_THRESHOLD,
                "merge_mode": "adjacent_same_bubble",
            }
        return kept

    def _merge_adjacent_regions(self, regions: list[TextRegion]) -> list[TextRegion]:
        current = list(regions)
        while True:
            bubble_groups: dict[int, list[TextRegion]] = {}
            passthrough: list[TextRegion] = []
            for region in current:
                if region.bubble_index >= 0:
                    bubble_groups.setdefault(region.bubble_index, []).append(region)
                else:
                    passthrough.append(region)

            changed = False
            next_regions = list(passthrough)
            for _, group in bubble_groups.items():
                components = self._bubble_components(group)
                for component in components:
                    if len(component) > 1:
                        next_regions.append(_merge_region_group(component))
                        changed = True
                    else:
                        next_regions.extend(component)

            current = sorted(next_regions, key=_region_sort_key)
            if not changed:
                return current

    @staticmethod
    def _bubble_components(regions: list[TextRegion]) -> list[list[TextRegion]]:
        ordered = sorted(regions, key=_region_sort_key)
        visited = [False] * len(ordered)
        components: list[list[TextRegion]] = []

        for start in range(len(ordered)):
            if visited[start]:
                continue
            queue = [start]
            visited[start] = True
            component: list[TextRegion] = []
            while queue:
                index = queue.pop()
                component.append(ordered[index])
                for other_index in range(len(ordered)):
                    if visited[other_index]:
                        continue
                    if _should_merge_adjacent_regions(ordered[index], ordered[other_index]):
                        visited[other_index] = True
                        queue.append(other_index)
            components.append(sorted(component, key=_region_sort_key))
        return components

    @staticmethod
    def _filter_low_score_regions(regions: list[TextRegion]) -> tuple[list[TextRegion], list[dict[str, object]]]:
        kept: list[TextRegion] = []
        filtered: list[dict[str, object]] = []
        for region in regions:
            merged_child_count = int(region.debug.get("merged_child_count", 1) or 1)
            fusion_sources = set(region.fusion_sources or [])
            easyocr_only = region.detector == "easyocr" and "comic_text_and_bubble_detector" not in fusion_sources
            if easyocr_only and region.easy_conf < LOW_SCORE_EASYOCR_THRESHOLD and merged_child_count <= 1:
                filtered.append(
                    {
                        "index": region.index,
                        "box": list(region.box),
                        "easy_conf": round(float(region.easy_conf), 4),
                        "bubble_index": region.bubble_index,
                        "reason": "low_easyocr_score_unmerged",
                    }
                )
                continue
            kept.append(region)
        return kept, filtered

    def _merge_with_easyocr_hints(
        self,
        image_rgb: np.ndarray,
        bubbles: list[Bubble],
        base_regions: list[TextRegion],
    ) -> list[TextRegion]:
        easyocr_regions = self._detect_with_easyocr(image_rgb, bubbles)
        merged = list(base_regions)
        next_index = len(merged)

        for easy_region in easyocr_regions:
            best_match: TextRegion | None = None
            best_score = 0.0
            for region in merged:
                score = _box_similarity(region.box, easy_region.box)
                if score > best_score:
                    best_score = score
                    best_match = region
            if best_match is not None and best_score >= self.config.detector.fusion_iou_threshold:
                self._merge_region_hint(best_match, easy_region)
                continue

            easy_region.index = next_index
            next_index += 1
            easy_region.debug["fusion_role"] = "easyocr_extra"
            merged.append(easy_region)
        return merged

    @staticmethod
    def _merge_region_hint(base: TextRegion, hint: TextRegion) -> None:
        hint_text = normalize_text(hint.easy_text or hint.detector_text_hint)
        if hint_text and (not base.easy_text or len(hint_text) >= len(base.easy_text)):
            base.detector_text_hint = hint_text
            base.easy_text = hint_text
            base.easy_conf = max(base.easy_conf, hint.easy_conf)
        base.fusion_sources = sorted(set((base.fusion_sources or []) + ["easyocr"]))
        if hint_text:
            base.debug["fusion_easyocr_text"] = hint_text

    @staticmethod
    def _should_keep_region(region: TextRegion) -> bool:
        x1, y1, x2, y2 = region.box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        text_hint = normalize_text(region.easy_text or region.detector_text_hint)
        if width < 12 or height < 12 or area < 320:
            return False
        if area < 900 and len(text_hint) < 2 and region.bubble_score < 0.25:
            return False
        if min(width, height) < 22 and region.easy_conf < 0.45 and region.bubble_score < 0.25:
            return False
        return True

    def _detect_bubbles(self, image_bgr: np.ndarray) -> list[Bubble]:
        speech_enabled = bool(self.config.bubble.enabled) and self.config.bubble.backend == "speech_bubble_segmentation"
        speech_bubbles: list[Bubble] = []
        edge_bubbles: list[Bubble] = []

        if speech_enabled:
            speech_bubbles = self.bubble_backend.detect(image_bgr)

        if not speech_enabled or self.config.bubble.fallback == "edges":
            edge_bubbles = detect_bubbles_by_edges(image_bgr)

        used_fallback = False
        if speech_bubbles:
            final_bubbles = _merge_unique_bubbles(speech_bubbles, edge_bubbles)
        else:
            final_bubbles = edge_bubbles
            used_fallback = speech_enabled

        self.last_bubble_runtime = self._compose_bubble_runtime(
            speech_bubbles=speech_bubbles,
            edge_bubbles=edge_bubbles,
            final_bubbles=final_bubbles,
            used_fallback=used_fallback,
        )
        return final_bubbles

    def _compose_bubble_runtime(
        self,
        speech_bubbles: list[Bubble],
        edge_bubbles: list[Bubble],
        final_bubbles: list[Bubble],
        used_fallback: bool = False,
    ) -> dict[str, object]:
        base = self.bubble_backend.runtime_summary()
        using_speech = any(bubble.source == "speech_bubble_segmentation" for bubble in final_bubbles) or (
            not final_bubbles and bool(base.get("model_loaded", False))
        )
        return {
            **base,
            "backend_name": "speech_bubble_segmentation" if using_speech else "edges",
            "configured_backend": self.config.bubble.backend,
            "configured_fallback": self.config.bubble.fallback,
            "speech_bubble_count": len(speech_bubbles),
            "edge_bubble_count": len(edge_bubbles),
            "final_bubble_count": len(final_bubbles),
            "used_fallback": used_fallback,
            "used_enhancement": using_speech and any(bubble.source == "edge" for bubble in final_bubbles),
        }

    def runtime_summary(self) -> dict[str, object]:
        detector_loaded = self.structure_backend.status.detector_loaded
        return {
            "backend_name": "comic_text_and_bubble_detector" if detector_loaded else "easyocr_fallback",
            "easyocr_fusion_enabled": self.config.detector.enable_easyocr_fusion,
            "model_loaded": detector_loaded,
            "model_path_hit": bool(self.structure_backend.status.resolved_source),
            "load_error_code": self.structure_backend.status.load_error_code,
            "load_error_message": self.structure_backend.status.load_error_message,
            "bubble_backend": self.last_bubble_runtime.get("backend_name", "edges"),
            "bubble_model_loaded": bool(self.last_bubble_runtime.get("model_loaded", False)),
            "bubble_model_path_hit": bool(self.last_bubble_runtime.get("model_path_hit", False)),
            "bubble_runtime": dict(self.last_bubble_runtime),
        }
