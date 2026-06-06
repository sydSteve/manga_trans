from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import AppConfig
from .models import TextRegion
from .utils import bubble_matches_region


VERTICAL_PUNCT = {
    "\u3001",
    "\u3002",
    "\uff0c",
    "\uff0e",
    "\u30fb",
    "\u30fc",
    "\u2026",
    "\u30fb",
    "\u300c",
    "\u300d",
    "\u300e",
    "\u300f",
    "\uff08",
    "\uff09",
    "\uff01",
    "\uff1f",
}


def _is_overlay_mode(region: TextRegion) -> bool:
    return region.inpaint_mode == "translucent_white_overlay"


def _font_candidates(region: TextRegion, config: AppConfig) -> list[str]:
    candidates: list[str] = []
    resolved = str(region.style.get("font_path", "") or region.style.get("resolved_font_path", ""))
    if resolved:
        candidates.append(resolved)
    family = str(region.style.get("font_family", ""))
    if family and family in config.style.font_family_map:
        candidates.extend(config.style.font_family_map[family])
    candidates.extend(config.render.font_paths)
    seen = set()
    deduped = []
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _load_font(font_paths: list[str], size: int) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, str]:
    for path in font_paths:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size), path
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default(), ""


def _measure_char(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, char: str) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
    return max(1, right - left), max(1, bottom - top)


def _box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


def _shrink_box(box: tuple[int, int, int, int], amount: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (x1 + amount, y1 + amount, max(x1 + amount + 1, x2 - amount), max(y1 + amount + 1, y2 - amount))


def _mask_bbox(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    if mask is None or mask.size == 0 or not np.any(mask > 0):
        return None
    coords = np.column_stack(np.where(mask > 0))
    y1 = int(coords[:, 0].min())
    y2 = int(coords[:, 0].max()) + 1
    x1 = int(coords[:, 1].min())
    x2 = int(coords[:, 1].max()) + 1
    return x1, y1, x2, y2


def _render_boxes(region: TextRegion) -> list[tuple[str, tuple[int, int, int, int]]]:
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    cover_write_box = region.debug.get("cover_write_box")
    if isinstance(cover_write_box, list) and len(cover_write_box) == 4:
        try:
            box = tuple(int(value) for value in cover_write_box)
            boxes.append(("cover_write_box", box))
        except Exception:  # noqa: BLE001
            pass
    if region.region_type == "dialogue_bubble" and region.bubble_bbox and bubble_matches_region(region.box, region.bubble_bbox):
        bx1, by1, bx2, by2 = region.bubble_bbox
        inset = max(4, min(bx2 - bx1, by2 - by1) // 20)
        boxes.append(("bubble_inner", _shrink_box(region.bubble_bbox, inset)))
    boxes.append(("box", region.box))

    deduped: list[tuple[str, tuple[int, int, int, int]]] = []
    seen = set()
    for source, box in boxes:
        normalized = (min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3]))
        if normalized not in seen and _box_area(normalized) > 0:
            seen.add(normalized)
            deduped.append((source, normalized))
    return deduped


def _render_targets(region: TextRegion, image: Image.Image) -> list[tuple[str, tuple[int, int, int, int], np.ndarray | None]]:
    targets: list[tuple[str, tuple[int, int, int, int], np.ndarray | None]] = []
    h, w = image.height, image.width
    render_mask = region.render_mask
    use_mask_target = region.inpaint_mode != "translucent_white_overlay"
    if use_mask_target and render_mask is not None and render_mask.shape[:2] == (h, w):
        bbox = _mask_bbox(render_mask)
        if bbox is not None and _box_area(bbox) > 0:
            x1, y1, x2, y2 = bbox
            targets.append(("render_mask", bbox, render_mask[y1:y2, x1:x2].copy()))
    for source, box in _render_boxes(region):
        if any(existing_box == box for _, existing_box, _ in targets):
            continue
        targets.append((source, box, None))
    if region.inpaint_mode == "translucent_white_overlay" and targets:
        return targets[:1]
    return targets


def _layout_box_metrics(
    box: tuple[int, int, int, int],
    config: AppConfig,
    region: TextRegion,
) -> tuple[int, int, int]:
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    overlay_mode = _is_overlay_mode(region)
    margin_cap = max(1, min(box_width, box_height) // (14 if overlay_mode else 9))
    if overlay_mode and region.region_type == "dialogue_bubble":
        base_margin = max(1, config.render.margin - 1)
    else:
        base_margin = config.render.margin + (1 if region.region_type == "dialogue_bubble" else 0)
    margin = max(1, min(base_margin, margin_cap))
    return max(1, box_width - margin * 2), max(1, box_height - margin * 2), margin


def _spacing_candidates(
    font_size: int,
    config: AppConfig,
    region: TextRegion,
    box: tuple[int, int, int, int],
) -> tuple[list[int], list[int]]:
    box_width = max(1, box[2] - box[0])
    box_height = max(1, box[3] - box[1])
    overlay_mode = _is_overlay_mode(region)
    if overlay_mode and region.region_type == "dialogue_bubble":
        base_line = max(config.render.line_spacing + 4, font_size // 3)
        base_column = max(config.render.column_spacing + 2, font_size // 4)
    elif region.region_type == "dialogue_bubble":
        base_line = max(config.render.line_spacing + 2, font_size // 4)
        base_column = max(config.render.column_spacing + 1, font_size // 4)
    else:
        base_line = max(config.render.line_spacing + 1, font_size // 5)
        base_column = max(config.render.column_spacing, font_size // 5)

    line_offsets = (-3, -1, 2, 5) if overlay_mode else (-2, 0, 2, 4)
    column_offsets = (-1, 1, 3) if overlay_mode else (-1, 0, 2)
    line_values = {max(2, min(base_line + delta, max(2, box_height // 4))) for delta in line_offsets}
    column_values = {max(2, min(base_column + delta, max(2, box_width // 5))) for delta in column_offsets}
    return sorted(line_values), sorted(column_values)


def _fill_target(config: AppConfig, region: TextRegion) -> float:
    base = (
        config.render.dialogue_max_fill_ratio
        if region.region_type == "dialogue_bubble"
        else config.render.narration_max_fill_ratio
        if region.region_type == "narration_box"
        else config.render.max_fill_ratio
    )
    if _is_overlay_mode(region) and region.region_type == "dialogue_bubble":
        return max(base, 0.92)
    return base


def _column_height(column: list[tuple[str, int, int]], line_spacing: int) -> int:
    if not column:
        return 0
    return sum(char_h for _, _, char_h in column) + max(0, len(column) - 1) * line_spacing


def _build_vertical_columns(
    items: list[tuple[str, int, int]],
    capacities: list[int],
    line_spacing: int,
) -> list[list[tuple[str, int, int]]] | None:
    columns: list[list[tuple[str, int, int]]] = []
    index = 0
    remaining_columns = len(capacities)

    for capacity in capacities:
        if index >= len(items):
            return None
        remaining_items = items[index:]
        remaining_total_height = _column_height(remaining_items, line_spacing)
        target_height = min(capacity, max(1, int(round(remaining_total_height / max(1, remaining_columns)))))
        current_column: list[tuple[str, int, int]] = []
        current_height = 0

        while index < len(items):
            char, char_w, char_h = items[index]
            projected_height = current_height + (line_spacing if current_column else 0) + char_h
            if projected_height > capacity and current_column:
                break
            if projected_height > capacity and not current_column:
                return None

            remaining_after = len(items) - (index + 1)
            must_leave = remaining_columns - 1
            current_delta = abs(target_height - current_height)
            projected_delta = abs(target_height - projected_height)
            if (
                current_column
                and remaining_columns > 1
                and projected_height > target_height
                and projected_delta > current_delta
                and remaining_after >= must_leave
            ):
                break

            current_column.append((char, char_w, char_h))
            current_height = projected_height
            index += 1

        if not current_column:
            return None
        columns.append(current_column)
        remaining_columns -= 1

    if index != len(items):
        return None
    return columns


def _safe_mask(mask_crop: np.ndarray, font_size: int, margin: int) -> tuple[np.ndarray, float]:
    binary = (mask_crop > 0).astype(np.uint8)
    if not np.any(binary > 0):
        return binary * 255, 0.0

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    thresholds = [
        max(1.0, float(font_size) * 0.25 + float(margin) * 0.5),
        max(1.0, float(font_size) * 0.18 + float(margin) * 0.35),
        1.0,
    ]
    original_pixels = max(1, int(np.count_nonzero(binary)))
    for threshold in thresholds:
        safe = np.where(distance >= threshold, 255, 0).astype(np.uint8)
        if np.count_nonzero(safe) >= max(32, int(original_pixels * 0.18)):
            return safe, threshold
    return binary * 255, 0.0


def _longest_run(flags: np.ndarray) -> tuple[int, int, int]:
    best_start = 0
    best_end = 0
    best_len = 0
    current_start = -1
    for index, value in enumerate(flags.tolist()):
        if value and current_start < 0:
            current_start = index
        elif not value and current_start >= 0:
            run_len = index - current_start
            if run_len > best_len:
                best_start = current_start
                best_end = index
                best_len = run_len
            current_start = -1
    if current_start >= 0:
        run_len = len(flags) - current_start
        if run_len > best_len:
            best_start = current_start
            best_end = len(flags)
            best_len = run_len
    return best_start, best_end, best_len


def _column_window(mask_crop: np.ndarray, x1: int, x2: int) -> dict[str, float] | None:
    band = mask_crop[:, x1:x2]
    if band.size == 0 or x2 <= x1:
        return None
    occupancy = np.mean(band > 0, axis=1)
    for threshold in (0.7, 0.55, 0.4):
        run = occupancy >= threshold
        start, end, length = _longest_run(run)
        if length > 0:
            coverage = float(np.mean(occupancy[start:end])) if end > start else 0.0
            return {
                "x1": float(x1),
                "x2": float(x2),
                "y1": float(start),
                "y2": float(end),
                "height": float(length),
                "coverage": coverage,
            }
    return None


def _column_layout_geometry(
    inner_left: int,
    inner_width: int,
    column_width: int,
    column_spacing: int,
    column_count: int,
) -> tuple[int, list[int], list[int], float]:
    total_width = column_count * column_width + max(0, column_count - 1) * column_spacing
    group_left = inner_left + max(0, (inner_width - total_width) // 2)
    positions = [group_left + total_width - column_width - index * (column_width + column_spacing) for index in range(column_count)]
    group_center = group_left + total_width / 2.0
    offsets = [int(round((x + column_width / 2.0) - group_center)) for x in positions]
    return total_width, positions, offsets, group_center


def _best_mask_columns(mask_crop: np.ndarray, column_width: int, column_spacing: int, column_count: int) -> list[dict[str, float]] | None:
    height, width = mask_crop.shape[:2]
    total_width = column_count * column_width + max(0, column_count - 1) * column_spacing
    if total_width > width:
        return None

    best_specs: list[dict[str, float]] | None = None
    best_score = -1.0
    max_left = max(0, width - total_width)
    step = 1 if max_left <= 12 else 2

    for left in range(0, max_left + 1, step):
        specs: list[dict[str, float]] = []
        min_height = float("inf")
        coverage_sum = 0.0
        for col_index in range(column_count):
            band_left = left + total_width - (col_index + 1) * column_width - col_index * column_spacing
            band_right = band_left + column_width
            spec = _column_window(mask_crop, band_left, band_right)
            if spec is None:
                specs = []
                break
            min_height = min(min_height, spec["height"])
            coverage_sum += spec["coverage"]
            specs.append(spec)
        if not specs:
            continue
        width_ratio = total_width / max(1, width)
        height_ratio = min_height / max(1, height)
        group_center = left + total_width / 2.0
        center_distance = abs(group_center - (width / 2.0)) / max(1.0, width / 2.0)
        centered_ratio = max(0.0, 1.0 - center_distance)
        score = width_ratio * 1.7 + height_ratio * 2.2 + coverage_sum / max(1, column_count) + centered_ratio * 0.85
        if score > best_score:
            best_score = score
            offsets = [int(round(((spec["x1"] + spec["x2"]) / 2.0) - group_center)) for spec in specs]
            for spec, offset in zip(specs, offsets):
                spec["column_offset"] = float(offset)
            specs[0]["group_center"] = float(group_center)
            specs[0]["centered_ratio"] = float(centered_ratio)
            best_specs = specs

    return best_specs


def _layout_vertical_mask(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    mask_crop: np.ndarray,
    font: ImageFont.ImageFont,
    config: AppConfig,
    region: TextRegion,
) -> tuple[bool, list[tuple[str, int, int]], dict[str, float]]:
    x1, y1, x2, y2 = box
    font_size = max(1, int(getattr(font, "size", config.render.min_font_size)))
    width, height, margin = _layout_box_metrics(box, config, region)
    items = [(char, *_measure_char(draw, font, char)) for char in text]
    if not items:
        return False, [], {}

    safe_mask, safe_threshold = _safe_mask(mask_crop, font_size, margin)
    if not np.any(safe_mask > 0):
        return False, [], {}

    max_col_width = max(char_w for _, char_w, _ in items)
    line_spacings, column_spacings = _spacing_candidates(font_size, config, region, box)
    best_candidate: tuple[list[tuple[str, int, int]], dict[str, float]] | None = None
    best_score = -1.0
    target_fill = _fill_target(config, region)
    overlay_mode = _is_overlay_mode(region)

    for line_spacing in line_spacings:
        total_text_height = _column_height(items, line_spacing)
        for column_spacing in column_spacings:
            max_columns = min(10 if overlay_mode else 8, max(1, (width + column_spacing) // max(1, max_col_width + column_spacing)))
            min_columns = max(1, min(max_columns, (total_text_height + height - 1) // max(1, height)))
            for column_count in range(min_columns, max_columns + 1):
                specs = _best_mask_columns(safe_mask, max_col_width, column_spacing, column_count)
                if not specs:
                    continue
                capacities = [int(spec["height"]) for spec in specs]
                columns = _build_vertical_columns(items, capacities, line_spacing)
                if not columns:
                    continue

                used_heights = [_column_height(column, line_spacing) for column in columns]
                total_width = len(columns) * max_col_width + max(0, len(columns) - 1) * column_spacing
                width_ratio = total_width / max(1, width)
                capacity_fill = sum(used_heights) / max(1, sum(capacities))
                max_height_ratio = max(
                    used_height / max(1, capacity) for used_height, capacity in zip(used_heights, capacities)
                )
                balance_ratio = min(used_heights) / max(1, max(used_heights))
                area_ratio = min(1.0, width_ratio) * min(1.0, capacity_fill)
                safe_area_ratio = np.count_nonzero(safe_mask) / max(1, safe_mask.shape[0] * safe_mask.shape[1])
                group_center = float(specs[0].get("group_center", safe_mask.shape[1] / 2.0))
                centered_ratio = float(specs[0].get("centered_ratio", 0.0))
                column_offsets = [int(spec.get("column_offset", 0.0)) for spec in specs]

                placements: list[tuple[str, int, int]] = []
                for spec, column, used_height in zip(specs, columns, used_heights):
                    local_x = int(spec["x1"])
                    local_y = int(spec["y1"]) + max(0, int(spec["height"] - used_height) // 2)
                    for char, char_w, char_h in column:
                        offset_x = max(0, (max_col_width - char_w) // 2)
                        punct_x = 0
                        punct_y = 0
                        if char in VERTICAL_PUNCT:
                            punct_x = max(1, char_w // 5)
                            punct_y = -max(0, char_h // 8)
                        placements.append((char, x1 + local_x + offset_x + punct_x, y1 + local_y + punct_y))
                        local_y += char_h + line_spacing

                candidate_score = (
                    area_ratio * (4.9 if overlay_mode else 4.1)
                    + min(1.0, capacity_fill) * (2.45 if overlay_mode else 2.1)
                    + min(1.0, width_ratio) * (1.45 if overlay_mode else 1.2)
                    + balance_ratio * (0.6 if overlay_mode else 0.45)
                    + centered_ratio * 0.5
                    + min(1.0, safe_area_ratio) * 0.25
                    - abs(target_fill - min(1.0, capacity_fill)) * (0.25 if overlay_mode else 0.4)
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_candidate = (
                        placements,
                        {
                            "fill_ratio": min(1.0, capacity_fill),
                            "height_ratio": max_height_ratio,
                            "area_ratio": area_ratio,
                            "balance_ratio": balance_ratio,
                            "width_ratio": width_ratio,
                            "column_count": float(len(columns)),
                            "line_spacing": float(line_spacing),
                            "column_spacing": float(column_spacing),
                            "safe_threshold": float(safe_threshold),
                            "safe_area_ratio": float(safe_area_ratio),
                            "centered_ratio": float(centered_ratio),
                            "group_center": float(group_center),
                            "column_offsets": column_offsets,
                            "layout_score": float(candidate_score),
                        },
                    )

    if best_candidate is None:
        return False, [], {}
    return True, best_candidate[0], best_candidate[1]


def _layout_vertical_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    config: AppConfig,
    region: TextRegion,
) -> tuple[bool, list[tuple[str, int, int]], dict[str, float]]:
    x1, y1, x2, y2 = box
    font_size = max(1, int(getattr(font, "size", config.render.min_font_size)))
    width, height, margin = _layout_box_metrics(box, config, region)
    items = [(char, *_measure_char(draw, font, char)) for char in text]
    if not items:
        return False, [], {}

    max_col_width = max(char_w for _, char_w, _ in items)
    line_spacings, column_spacings = _spacing_candidates(font_size, config, region, box)
    best_candidate: tuple[list[tuple[str, int, int]], dict[str, float]] | None = None
    best_score = -1.0
    target_fill = _fill_target(config, region)
    overlay_mode = _is_overlay_mode(region)

    for line_spacing in line_spacings:
        total_text_height = _column_height(items, line_spacing)
        for column_spacing in column_spacings:
            min_columns = max(1, (total_text_height + height - 1) // max(1, height))
            max_columns = min(10 if overlay_mode else 8, max(1, (width + column_spacing) // max(1, max_col_width + column_spacing)))

            for column_count in range(min_columns, max_columns + 1):
                capacities = [height] * column_count
                columns = _build_vertical_columns(items, capacities, line_spacing)
                if not columns:
                    continue

                total_width = len(columns) * max_col_width + max(0, len(columns) - 1) * column_spacing
                if total_width > width:
                    continue

                used_heights = [_column_height(column, line_spacing) for column in columns]
                max_column_height = max(used_heights)
                min_column_height = min(used_heights)
                width_ratio = total_width / max(1, width)
                height_ratio = max_column_height / max(1, height)
                capacity_fill = sum(used_heights) / max(1, column_count * height)
                area_ratio = min(1.0, width_ratio) * min(1.0, capacity_fill)
                balance_ratio = min_column_height / max(1, max_column_height)
                inner_left = x1 + margin
                total_width, positions, column_offsets, group_center = _column_layout_geometry(
                    inner_left,
                    width,
                    max_col_width,
                    column_spacing,
                    len(columns),
                )
                centered_ratio = max(0.0, 1.0 - abs(group_center - (x1 + margin + width / 2.0)) / max(1.0, width / 2.0))

                placements: list[tuple[str, int, int]] = []
                for x, column in zip(positions, columns):
                    column_height = _column_height(column, line_spacing)
                    y = y1 + margin + max(0, (height - column_height) // 2)
                    for char, char_w, char_h in column:
                        offset_x = max(0, (max_col_width - char_w) // 2)
                        punct_x = 0
                        punct_y = 0
                        if char in VERTICAL_PUNCT:
                            punct_x = max(1, char_w // 5)
                            punct_y = -max(0, char_h // 8)
                        placements.append((char, x + offset_x + punct_x, y + punct_y))
                        y += char_h + line_spacing

                candidate_score = (
                    area_ratio * (4.7 if overlay_mode else 4.0)
                    + min(1.0, capacity_fill) * (2.25 if overlay_mode else 1.9)
                    + min(1.0, width_ratio) * (1.3 if overlay_mode else 1.0)
                    + balance_ratio * (0.55 if overlay_mode else 0.35)
                    + centered_ratio * 0.6
                    - abs(target_fill - min(1.0, capacity_fill)) * (0.24 if overlay_mode else 0.35)
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_candidate = (
                        placements,
                        {
                            "fill_ratio": min(1.0, capacity_fill),
                            "height_ratio": height_ratio,
                            "area_ratio": area_ratio,
                            "balance_ratio": balance_ratio,
                            "width_ratio": width_ratio,
                            "column_count": float(len(columns)),
                            "line_spacing": float(line_spacing),
                            "column_spacing": float(column_spacing),
                            "centered_ratio": float(centered_ratio),
                            "group_center": float(group_center),
                            "column_offsets": column_offsets,
                            "layout_score": float(candidate_score),
                        },
                    )

    if best_candidate is None:
        return False, [], {}
    return True, best_candidate[0], best_candidate[1]


def _render_score(metrics: dict[str, float], font_size: int, config: AppConfig, region: TextRegion) -> float:
    target_fill = _fill_target(config, region)
    overlay_mode = _is_overlay_mode(region)
    return (
        float(metrics.get("layout_score", 0.0))
        + float(metrics.get("area_ratio", 0.0)) * (1.95 if overlay_mode else 1.1)
        + min(1.0, float(metrics.get("width_ratio", 0.0))) * (0.55 if overlay_mode else 0.35)
        + float(metrics.get("balance_ratio", 0.0)) * (0.25 if overlay_mode else 0.0)
        + float(metrics.get("centered_ratio", 0.0)) * 0.4
        + min(1.0, font_size / max(1, config.render.max_font_size + max(0, config.render.font_size_offset) + (8 if overlay_mode else 0))) * (0.14 if overlay_mode else 0.08)
        - abs(target_fill - min(1.0, float(metrics.get("fill_ratio", 0.0)))) * (0.25 if overlay_mode else 0.45)
    )


def render_translation(image: Image.Image, region: TextRegion, config: AppConfig) -> bool:
    if not region.translation:
        return False

    draw = ImageDraw.Draw(image)
    fill = tuple(region.style.get("text_color", config.render.text_color))
    stroke_fill = tuple(region.style.get("stroke_color", config.render.stroke_color))
    font_candidates = _font_candidates(region, config)
    overflow_attempts = 0
    overlay_mode = _is_overlay_mode(region)
    size_step = 2 if overlay_mode else 1

    for box_source, box, mask_crop in _render_targets(region, image):
        box_width = max(1, box[2] - box[0])
        box_height = max(1, box[3] - box[1])
        adaptive_min_font = min(config.render.min_font_size, max(6, min(box_width, box_height) // 4))
        extra_font_headroom = max(0, min(12, min(box_width, box_height) // 8)) if overlay_mode else 0
        adaptive_max_font = max(
            adaptive_min_font,
            min(config.render.max_font_size + config.render.font_size_offset + extra_font_headroom, max(box_width, box_height)),
        )
        best_candidate: dict[str, object] | None = None
        best_score = -1.0

        for size in range(adaptive_max_font, adaptive_min_font - 1, -size_step):
            overflow_attempts += 1
            font, resolved_path = _load_font(font_candidates, size)
            stroke_width = int(region.style.get("stroke_width", config.render.stroke_width))
            if min(box_width, box_height) <= 72:
                stroke_width = max(1, int(round(stroke_width * config.render.small_box_stroke_scale)))

            if mask_crop is not None:
                ok, placements, metrics = _layout_vertical_mask(draw, region.translation, box, mask_crop, font, config, region)
            else:
                ok, placements, metrics = _layout_vertical_box(draw, region.translation, box, font, config, region)
            if not ok:
                continue

            candidate_score = _render_score(metrics, size, config, region)
            replace_candidate = candidate_score > best_score
            if best_candidate and not replace_candidate and overlay_mode:
                best_metrics = best_candidate["metrics"]
                close_score = candidate_score >= best_score - 0.08
                better_area = float(metrics.get("area_ratio", 0.0)) > float(best_metrics.get("area_ratio", 0.0)) + 0.03
                better_fill = float(metrics.get("fill_ratio", 0.0)) > float(best_metrics.get("fill_ratio", 0.0)) + 0.02
                if close_score and (better_area or better_fill):
                    replace_candidate = True
            if replace_candidate:
                best_score = candidate_score
                best_candidate = {
                    "placements": placements,
                    "metrics": metrics,
                    "font": font,
                    "resolved_path": resolved_path,
                    "stroke_width": stroke_width,
                    "font_size": size,
                    "source": box_source,
                }

        if best_candidate:
            for text, x, y in best_candidate["placements"]:
                draw.text(
                    (x, y),
                    text,
                    font=best_candidate["font"],
                    fill=fill,
                    stroke_width=best_candidate["stroke_width"],
                    stroke_fill=stroke_fill,
                )

            metrics = best_candidate["metrics"]
            region.debug["render_orientation"] = "vertical"
            region.debug["render_font_size"] = int(best_candidate["font_size"])
            region.debug["render_box_source"] = str(best_candidate["source"])
            region.debug["render_fill_ratio"] = round(float(metrics.get("fill_ratio", 0.0)), 4)
            region.debug["render_height_ratio"] = round(float(metrics.get("height_ratio", 0.0)), 4)
            region.debug["render_area_ratio"] = round(float(metrics.get("area_ratio", 0.0)), 4)
            region.debug["render_width_ratio"] = round(float(metrics.get("width_ratio", 0.0)), 4)
            region.debug["render_column_count"] = int(metrics.get("column_count", 1.0))
            region.debug["render_line_spacing"] = int(metrics.get("line_spacing", 0.0))
            region.debug["render_column_spacing"] = int(metrics.get("column_spacing", 0.0))
            region.debug["render_layout_score"] = round(float(metrics.get("layout_score", 0.0)), 4)
            region.debug["render_anchor_mode"] = "centered_columns"
            region.debug["render_centered"] = True
            region.debug["render_centered_ratio"] = round(float(metrics.get("centered_ratio", 0.0)), 4)
            region.debug["render_column_offsets"] = list(metrics.get("column_offsets", []))
            region.debug["render_target_fill"] = round(float(_fill_target(config, region)), 4)
            if "safe_threshold" in metrics:
                region.debug["render_safe_threshold"] = round(float(metrics["safe_threshold"]), 2)
            if "safe_area_ratio" in metrics:
                region.debug["render_safe_area_ratio"] = round(float(metrics["safe_area_ratio"]), 4)
            region.debug["render_overflow_attempts"] = overflow_attempts
            region.debug["resolved_font_path"] = str(best_candidate["resolved_path"])
            region.render_rect = box
            return True

    region.skip_reason = region.skip_reason or "text_overflow_skip"
    region.debug["render_overflow_attempts"] = overflow_attempts
    return False
