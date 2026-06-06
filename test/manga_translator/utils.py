from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def normalize_text(text: str) -> str:
    return "".join(str(text or "").replace("\r", "").split()).strip()


def clamp_box(box: tuple[int, int, int, int], shape: tuple[int, int, int] | tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def expand_box(box: tuple[int, int, int, int], shape: tuple[int, int, int] | tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return clamp_box((x1 - padding, y1 - padding, x2 + padding, y2 + padding), shape)


def shrink_box(box: tuple[int, int, int, int], shape: tuple[int, int, int] | tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return clamp_box((x1 + padding, y1 + padding, x2 - padding, y2 - padding), shape)


def intersection_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / a_area


def box_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def bubble_matches_region(region_box: tuple[int, int, int, int], bubble_box: tuple[int, int, int, int] | None) -> bool:
    if not bubble_box:
        return False
    overlap = intersection_ratio(region_box, bubble_box)
    if overlap >= 0.2:
        return True
    cx, cy = box_center(region_box)
    bx1, by1, bx2, by2 = bubble_box
    pad_x = max(6, (bx2 - bx1) // 12)
    pad_y = max(6, (by2 - by1) // 12)
    return (bx1 - pad_x) <= cx <= (bx2 + pad_x) and (by1 - pad_y) <= cy <= (by2 + pad_y)


def is_kana_char(ch: str) -> bool:
    return "\u3040" <= ch <= "\u30ff"


def is_katakana(ch: str) -> bool:
    return "\u30a0" <= ch <= "\u30ff"


def contains_kanji(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def kana_ratio(text: str) -> float:
    compact = normalize_text(text)
    if not compact:
        return 0.0
    return sum(1 for ch in compact if is_kana_char(ch)) / len(compact)


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        [path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()]
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def chunked(items: Iterable, size: int) -> list[list]:
    bucket = []
    chunks = []
    for item in items:
        bucket.append(item)
        if len(bucket) >= size:
            chunks.append(bucket)
            bucket = []
    if bucket:
        chunks.append(bucket)
    return chunks


def image_variance(gray: np.ndarray) -> float:
    return float(np.var(gray.astype(np.float32)))
