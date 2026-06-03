import os
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["OMP_NUM_THREADS"] = "4"        # 限制 OpenMP 线程
os.environ["MKL_NUM_THREADS"] = "4"        # 限制 MKL 线程
os.environ["OPENBLAS_NUM_THREADS"] = "4"   # 限制 OpenBLAS 线程
# 如果你有 GPU 并且想用，可以保留 CPU 线程数较少，让 GPU 干活
torch.set_num_threads(4)          # 限制 PyTorch 只用 4 个核心
import sys

# Windows GBK 控制台兼容：避免 〜(U+301C) 等字符导致 print 崩溃
_original_print = print
def _safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = [str(a).encode('gbk', errors='replace').decode('gbk') for a in args]
        _original_print(*safe_args, **kwargs)
print = _safe_print

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from manga_ocr import MangaOcr
import easyocr
import requests
import json
import re
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()
ENABLE_DEBUG_CLASSIFY = os.environ.get("ENABLE_DEBUG_CLASSIFY", "false").lower() == "true"
ENABLE_DEBUG_JSON = os.environ.get("ENABLE_DEBUG_JSON", "true").lower() == "true"
ENABLE_DEBUG_RENDER = os.environ.get("ENABLE_DEBUG_RENDER", "false").lower() == "true"
MIN_OCR_CONFIDENCE = float(os.environ.get("MIN_OCR_CONFIDENCE", "0.18"))
SFX_SKIP_MAX_LEN = int(os.environ.get("SFX_SKIP_MAX_LEN", "4"))
DIALOGUE_SCORE_THRESHOLD = float(os.environ.get("DIALOGUE_SCORE_THRESHOLD", "3.4"))
NARRATION_SCORE_THRESHOLD = float(os.environ.get("NARRATION_SCORE_THRESHOLD", "2.8"))
CLASSIFICATION_MARGIN = float(os.environ.get("CLASSIFICATION_MARGIN", "0.9"))
MANGA_OCR_FALLBACK_MIN_LEN = int(os.environ.get("MANGA_OCR_FALLBACK_MIN_LEN", "2"))
BUBBLE_MASK_PADDING = int(os.environ.get("BUBBLE_MASK_PADDING", "6"))
BUBBLE_BORDER_PROTECT = int(os.environ.get("BUBBLE_BORDER_PROTECT", "3"))
NARRATION_MASK_MIN_AREA = int(os.environ.get("NARRATION_MASK_MIN_AREA", "900"))
MASK_CONFIDENCE_THRESHOLD = float(os.environ.get("MASK_CONFIDENCE_THRESHOLD", "0.55"))
NARRATION_MAX_AREA_RATIO_TO_ANCHOR = float(os.environ.get("NARRATION_MAX_AREA_RATIO_TO_ANCHOR", "5.5"))
NARRATION_MAX_AREA_RATIO_TO_SEARCH = float(os.environ.get("NARRATION_MAX_AREA_RATIO_TO_SEARCH", "0.72"))
NARRATION_MIN_SHAPE_SCORE = float(os.environ.get("NARRATION_MIN_SHAPE_SCORE", "0.62"))
NARRATION_CORE_SHRINK = int(os.environ.get("NARRATION_CORE_SHRINK", "5"))
NARRATION_SMALL_FALLBACK_PADDING = int(os.environ.get("NARRATION_SMALL_FALLBACK_PADDING", "1"))
BUBBLE_CLASSIFICATION_MARGIN = float(os.environ.get("BUBBLE_CLASSIFICATION_MARGIN", "0.55"))
BUBBLE_SHORT_TEXT_MAX_LEN = int(os.environ.get("BUBBLE_SHORT_TEXT_MAX_LEN", "4"))
BUBBLE_RESCUE_MIN_CONFIDENCE = float(os.environ.get("BUBBLE_RESCUE_MIN_CONFIDENCE", "0.08"))
ENABLE_BUBBLE_TEXT_RESCUE = os.environ.get("ENABLE_BUBBLE_TEXT_RESCUE", "true").lower() == "true"
BUBBLE_OCR_FALLBACK_ALLOW_SHORT = os.environ.get("BUBBLE_OCR_FALLBACK_ALLOW_SHORT", "true").lower() == "true"
ENABLE_COLORED_BUBBLE_RESCUE = os.environ.get("ENABLE_COLORED_BUBBLE_RESCUE", "true").lower() == "true"
COLORED_BUBBLE_SCORE_THRESHOLD = float(os.environ.get("COLORED_BUBBLE_SCORE_THRESHOLD", "1.45"))
COLORED_BUBBLE_MARGIN = float(os.environ.get("COLORED_BUBBLE_MARGIN", "0.35"))
BUBBLE_OUTLINE_MIN_CONTRAST = float(os.environ.get("BUBBLE_OUTLINE_MIN_CONTRAST", "14.0"))
BUBBLE_INTERIOR_MAX_VARIANCE = float(os.environ.get("BUBBLE_INTERIOR_MAX_VARIANCE", "42.0"))
ENABLE_SPIKY_BUBBLE_RESCUE = os.environ.get("ENABLE_SPIKY_BUBBLE_RESCUE", "true").lower() == "true"
SPIKY_BUBBLE_SCORE_THRESHOLD = float(os.environ.get("SPIKY_BUBBLE_SCORE_THRESHOLD", "0.58"))
SPIKY_BUBBLE_MARGIN_BONUS = float(os.environ.get("SPIKY_BUBBLE_MARGIN_BONUS", "0.28"))
ENABLE_STRUCTURED_DIALOGUE_RESCUE = os.environ.get("ENABLE_STRUCTURED_DIALOGUE_RESCUE", "true").lower() == "true"
STRUCTURED_DIALOGUE_MIN_SCORE = float(os.environ.get("STRUCTURED_DIALOGUE_MIN_SCORE", "2.95"))
ENABLE_BUBBLE_CONSTRAINED_OCR_CROP = os.environ.get("ENABLE_BUBBLE_CONSTRAINED_OCR_CROP", "true").lower() == "true"
TRANSLATION_RETRY_FOR_STYLIZED_DIALOGUE = os.environ.get("TRANSLATION_RETRY_FOR_STYLIZED_DIALOGUE", "true").lower() == "true"
ENABLE_SMALL_DIALOGUE_RESCUE = os.environ.get("ENABLE_SMALL_DIALOGUE_RESCUE", "true").lower() == "true"
SMALL_DIALOGUE_MIN_SCORE = float(os.environ.get("SMALL_DIALOGUE_MIN_SCORE", "2.55"))
RENDER_CORE_SHRINK = int(os.environ.get("RENDER_CORE_SHRINK", "6"))
RENDER_MAX_FILL_RATIO = float(os.environ.get("RENDER_MAX_FILL_RATIO", "0.76"))
TEXT_FIT_MIN_SCALE = float(os.environ.get("TEXT_FIT_MIN_SCALE", "0.52"))
TEXT_OVERFLOW_SKIP_THRESHOLD = float(os.environ.get("TEXT_OVERFLOW_SKIP_THRESHOLD", "1.03"))
ENABLE_LARGE_REGION_SPLIT = os.environ.get("ENABLE_LARGE_REGION_SPLIT", "true").lower() == "true"
LARGE_REGION_SPLIT_MIN_AREA = int(os.environ.get("LARGE_REGION_SPLIT_MIN_AREA", "85000"))
LARGE_REGION_SPLIT_GAP_RATIO = float(os.environ.get("LARGE_REGION_SPLIT_GAP_RATIO", "1.35"))
ENABLE_SINGLE_LARGE_SPLIT = os.environ.get("ENABLE_SINGLE_LARGE_SPLIT", "true").lower() == "true"
SINGLE_LARGE_SPLIT_MIN_AREA = int(os.environ.get("SINGLE_LARGE_SPLIT_MIN_AREA", "42000"))
SINGLE_LARGE_SPLIT_MIN_SIDE = int(os.environ.get("SINGLE_LARGE_SPLIT_MIN_SIDE", "180"))
PROJECTION_VALLEY_RATIO = float(os.environ.get("PROJECTION_VALLEY_RATIO", "0.2"))
PROJECTION_VALLEY_MIN_WIDTH = int(os.environ.get("PROJECTION_VALLEY_MIN_WIDTH", "12"))
PROJECTION_VALLEY_MIN_RATIO = float(os.environ.get("PROJECTION_VALLEY_MIN_RATIO", "0.1"))
PROJECTION_CHILD_MIN_AREA = int(os.environ.get("PROJECTION_CHILD_MIN_AREA", "1800"))
SINGLE_LARGE_SPLIT_MAX_CUTS = int(os.environ.get("SINGLE_LARGE_SPLIT_MAX_CUTS", "2"))
ENABLE_SECONDARY_OCR_FILTER = os.environ.get("ENABLE_SECONDARY_OCR_FILTER", "true").lower() == "true"
OCR_ANOMALY_MIN_CONFIDENCE = float(os.environ.get("OCR_ANOMALY_MIN_CONFIDENCE", "0.32"))
TRANSLATE_SKIP_PATTERNS = tuple(
    item.strip() for item in os.environ.get(
        "TRANSLATE_SKIP_PATTERNS",
        "跳过,拟声词,动作字,意义不明,无法判断,不翻译"
    ).split(",") if item.strip()
)
DEBUG_OUTPUT_DIR = os.environ.get("DEBUG_OUTPUT_DIR", "debug_previews")
SHORT_DIALOGUE_WHITELIST = {"あ", "え", "う", "お", "ん", "えっ", "あっ", "うん", "はい", "へえ", "ええ"}
SHORT_TRANSLATION_MAP = {
    "あ": "啊",
    "え": "诶……",
    "えっ": "诶？",
    "うん": "嗯",
    "ん": "嗯",
    "はい": "好",
    "へえ": "哦……",
    "ええ": "嗯",
}
SFX_KEYWORDS = {
    "コン", "ポリ", "もじ", "バタン", "バタンッ", "ドン", "バン", "ガン", "ガタン",
    "ガチャ", "ガチャッ", "バサ", "ザワ", "ゴク", "カァ", "ぽっ", "ぽり", "ばたん"
}
JP_PUNCT_TRANSLATION = str.maketrans({
    ",": "，",
    "、": "，",
    ".": "。",
    "!": "！",
    "?": "？",
})
#================对话泡检测====================
DIALOGUE_PUNCT_MARKS = {"!", "?", "\uff01", "\uff1f", "\u2026", "\u30fc", "\u301c", "\uff5e", "\u3001", "\u3002"}
BUBBLE_REACTION_WORDS = {
    "\u3048", "\u3046\u3093", "\u306f\u3044", "\u3044\u3084", "\u3042", "\u3078\u3047", "\u307b\u3046",
    "\u305d\u3046", "\u3046\u3046\u3093", "\u3042\u308c", "\u306d", "\u306f\u3043"
}
HEART_MARKS = {"\u2661", "\u2665", "\u2764", "\U0001f496"}
BUBBLE_SHORT_TRANSLATION_MAP = {
    "\u3048": "诶……",
    "\u3046\u3093": "嗯。",
    "\u306f\u3044": "好。",
    "\u3044\u3084": "不是……",
    "\u305d\u3046": "是啊。",
    "\u305d\u3046\uff01": "是啊！",
    "\u3053\u3001\u3053\u308c\u306f": "这、这是……",
    "\u3053\u3001\u3053\u308c\u306f\u2026": "这、这是……",
}
STYLIZED_DIALOGUE_FALLBACK_MAP = {
    "\u7f8e\u5473\u3057\u304f\u306a\u30fc\u308c\u2661\u3082\u3048\u3082\u3048\u304d\u3085\u3093\u2661": "\u53d8\u5f97\u7f8e\u5473\u5427\u2661\u840c\u840c\u557e\u2661",
    "\u3082\u3048\u3082\u3048\u304d\u3085\u3093\u2661": "\u840c\u840c\u557e\u2661",
}

def compute_crop_bubble_features(crop_rgb):
    if crop_rgb is None or crop_rgb.size == 0:
        return {
            "outline_contrast": 0.0,
            "interior_uniformity": 0.0,
            "mean_saturation": 0.0,
            "colored_ratio": 0.0,
            "colored_bubble_like": False,
            "spiky_bubble_like": False,
            "spiky_edge_complexity": 0.0,
            "bubble_like_score": 0.0,
        }

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    h, w = gray.shape[:2]
    band = max(2, min(h, w) // 10)

    inner_mask = np.zeros_like(gray, dtype=bool)
    if h > band * 2 and w > band * 2:
        inner_mask[band:h - band, band:w - band] = True
    else:
        inner_mask[:, :] = True
    border_mask = ~inner_mask
    if not np.any(border_mask):
        border_mask = np.ones_like(gray, dtype=bool)

    inner_gray = gray[inner_mask].astype(np.float32)
    border_gray = gray[border_mask].astype(np.float32)
    inner_sat = hsv[:, :, 1][inner_mask].astype(np.float32)
    inner_val = hsv[:, :, 2][inner_mask].astype(np.float32)

    raw_variance = float(0.55 * np.std(inner_val) + 0.45 * np.std(inner_sat))
    interior_uniformity = max(0.0, 1.0 - min(1.0, raw_variance / max(1.0, BUBBLE_INTERIOR_MAX_VARIANCE)))
    outline_contrast = abs(float(np.mean(border_gray) - np.mean(inner_gray))) if len(border_gray) and len(inner_gray) else 0.0
    mean_saturation = float(np.mean(inner_sat)) if len(inner_sat) else 0.0
    colored_ratio = float(np.mean(inner_sat >= 18.0)) if len(inner_sat) else 0.0
    bright_ratio = float(np.mean(inner_gray >= 190)) if len(inner_gray) else 0.0
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0)) if edges.size else 0.0
    spiky_edge_complexity = max(0.0, min(1.0, (edge_density - 0.035) / 0.16))

    bubble_like_score = 0.0
    if colored_ratio >= 0.25:
        bubble_like_score += 0.55
    if interior_uniformity >= 0.24:
        bubble_like_score += 0.55
    if outline_contrast >= BUBBLE_OUTLINE_MIN_CONTRAST:
        bubble_like_score += 0.75
    elif outline_contrast >= BUBBLE_OUTLINE_MIN_CONTRAST * 0.65:
        bubble_like_score += 0.35
    if 14.0 <= mean_saturation <= 170.0:
        bubble_like_score += 0.25

    colored_bubble_like = (
        colored_ratio >= 0.30
        and interior_uniformity >= 0.22
        and outline_contrast >= BUBBLE_OUTLINE_MIN_CONTRAST * 0.65
    )
    spiky_bubble_like = (
        bright_ratio >= 0.55
        and mean_saturation <= 55.0
        and interior_uniformity >= 0.18
        and outline_contrast >= BUBBLE_OUTLINE_MIN_CONTRAST * 0.78
        and 0.08 <= edge_density <= 0.24
    )
    if spiky_bubble_like:
        bubble_like_score += 0.45
    return {
        "outline_contrast": outline_contrast,
        "interior_uniformity": interior_uniformity,
        "mean_saturation": mean_saturation,
        "colored_ratio": colored_ratio,
        "colored_bubble_like": colored_bubble_like,
        "spiky_bubble_like": spiky_bubble_like,
        "spiky_edge_complexity": spiky_edge_complexity,
        "bubble_like_score": bubble_like_score,
    }

def analyze_bubble_contour(image_bgr, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    crop_bgr = image_bgr[y:y + h, x:x + w]
    if crop_bgr.size == 0:
        return {
            "fill_ratio": 0.0,
            "outline_contrast": 0.0,
            "interior_uniformity": 0.0,
            "mean_saturation": 0.0,
            "colored_bubble_like": False,
            "spiky_bubble_like": False,
            "spiky_edge_complexity": 0.0,
            "bubble_like_score": 0.0,
        }

    local_cnt = cnt.reshape(-1, 2) - np.array([[x, y]])
    fill_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(fill_mask, [local_cnt.astype(np.int32)], 255)
    area = float(np.count_nonzero(fill_mask > 0))
    fill_ratio = area / max(1.0, w * h)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner_mask = cv2.erode(fill_mask, kernel, iterations=1)
    if np.count_nonzero(inner_mask > 0) < max(25, area * 0.18):
        inner_mask = fill_mask.copy()
    ring_mask = cv2.subtract(fill_mask, inner_mask)
    if np.count_nonzero(ring_mask > 0) < 10:
        ring_mask = fill_mask.copy()

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    inner_pixels = inner_mask > 0
    ring_pixels = ring_mask > 0
    inner_gray = gray[inner_pixels].astype(np.float32)
    ring_gray = gray[ring_pixels].astype(np.float32)
    inner_sat = hsv[:, :, 1][inner_pixels].astype(np.float32)
    inner_val = hsv[:, :, 2][inner_pixels].astype(np.float32)

    raw_variance = float(0.55 * np.std(inner_val) + 0.45 * np.std(inner_sat)) if len(inner_val) else BUBBLE_INTERIOR_MAX_VARIANCE
    interior_uniformity = max(0.0, 1.0 - min(1.0, raw_variance / max(1.0, BUBBLE_INTERIOR_MAX_VARIANCE)))
    outline_contrast = abs(float(np.mean(ring_gray) - np.mean(inner_gray))) if len(ring_gray) and len(inner_gray) else 0.0
    mean_saturation = float(np.mean(inner_sat)) if len(inner_sat) else 0.0
    bright_ratio = float(np.mean(inner_gray >= 190)) if len(inner_gray) else 0.0
    perimeter = float(cv2.arcLength(cnt, True))
    circularity = float((4.0 * np.pi * area) / max(1.0, perimeter * perimeter))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / max(1.0, hull_area))
    spiky_edge_complexity = max(0.0, perimeter / max(1.0, 2.0 * np.sqrt(np.pi * max(area, 1.0))) - 1.0)

    bubble_like_score = 0.0
    if fill_ratio >= 0.22:
        bubble_like_score += 0.45
    if fill_ratio >= 0.38:
        bubble_like_score += 0.35
    if interior_uniformity >= 0.22:
        bubble_like_score += 0.50
    if outline_contrast >= BUBBLE_OUTLINE_MIN_CONTRAST:
        bubble_like_score += 0.80
    elif outline_contrast >= BUBBLE_OUTLINE_MIN_CONTRAST * 0.65:
        bubble_like_score += 0.35
    if mean_saturation >= 18.0:
        bubble_like_score += 0.35

    colored_bubble_like = (
        mean_saturation >= 18.0
        and interior_uniformity >= 0.18
        and outline_contrast >= BUBBLE_OUTLINE_MIN_CONTRAST * 0.6
    )
    spiky_bubble_like = (
        bright_ratio >= 0.52
        and mean_saturation <= 45.0
        and fill_ratio >= 0.18
        and interior_uniformity >= 0.15
        and outline_contrast >= BUBBLE_OUTLINE_MIN_CONTRAST * 0.78
        and circularity <= 0.84
        and solidity >= 0.72
        and spiky_edge_complexity >= SPIKY_BUBBLE_SCORE_THRESHOLD
    )
    if spiky_bubble_like:
        bubble_like_score += 0.55
    return {
        "fill_ratio": fill_ratio,
        "outline_contrast": outline_contrast,
        "interior_uniformity": interior_uniformity,
        "mean_saturation": mean_saturation,
        "colored_bubble_like": colored_bubble_like,
        "spiky_bubble_like": spiky_bubble_like,
        "spiky_edge_complexity": round(float(spiky_edge_complexity), 4),
        "bubble_like_score": bubble_like_score,
    }

def rect_iou(rect_a, rect_b):
    inter = rect_intersection_area(rect_a, rect_b)
    if inter <= 0:
        return 0.0
    area_a = rect_area(rect_a)
    area_b = rect_area(rect_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def build_bubble_record(image_bgr, contour, margin=5, source="edge"):
    if contour is None or len(contour) < 3:
        return None
    cnt = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 24 or h < 24:
        return None
    metrics = analyze_bubble_contour(image_bgr, cnt)
    if metrics["fill_ratio"] < 0.12:
        return None
    extra_margin = 10 if metrics.get("spiky_bubble_like", False) else 0
    x1 = max(0, x - margin - extra_margin)
    y1 = max(0, y - margin - extra_margin)
    x2 = min(image_bgr.shape[1], x + w + margin + extra_margin)
    y2 = min(image_bgr.shape[0], y + h + margin + extra_margin)
    return {
        "bbox": (x1, y1, x2, y2),
        "area": area,
        "contour": cnt.reshape(-1, 2).tolist(),
        "fill_ratio": metrics["fill_ratio"],
        "outline_contrast": metrics["outline_contrast"],
        "interior_uniformity": metrics["interior_uniformity"],
        "mean_saturation": metrics["mean_saturation"],
        "colored_bubble_like": metrics["colored_bubble_like"],
        "spiky_bubble_like": metrics.get("spiky_bubble_like", False),
        "spiky_edge_complexity": metrics.get("spiky_edge_complexity", 0.0),
        "bubble_like_score": metrics["bubble_like_score"],
        "candidate_source": source,
    }

def add_unique_bubble_candidate(bubbles, candidate):
    if not candidate:
        return bubbles
    candidate_rect = get_bubble_bbox(candidate)
    candidate_score = float(candidate.get("bubble_like_score", 0.0))
    for idx, existing in enumerate(bubbles):
        existing_rect = get_bubble_bbox(existing)
        overlap = rect_intersection_area(candidate_rect, existing_rect) / max(
            1.0, min(rect_area(candidate_rect), rect_area(existing_rect))
        )
        if rect_iou(candidate_rect, existing_rect) >= 0.62 or overlap >= 0.78:
            existing_score = float(existing.get("bubble_like_score", 0.0))
            if candidate_score > existing_score + 0.08:
                bubbles[idx] = candidate
            return bubbles
    bubbles.append(candidate)
    return bubbles

def detect_anchor_bubble_candidate(image_bgr, anchor_rect):
    anchor_rect = clamp_rect(anchor_rect, image_bgr.shape)
    if rect_area(anchor_rect) <= 0:
        return None

    anchor_w = max(1, anchor_rect[2] - anchor_rect[0])
    anchor_h = max(1, anchor_rect[3] - anchor_rect[1])
    anchor_area = rect_area(anchor_rect)
    padding = max(22, min(120, int(max(anchor_w, anchor_h) * 0.75)))
    search_rect = expand_rect(anchor_rect, image_bgr.shape, padding=padding)
    x1, y1, x2, y2 = search_rect
    crop_bgr = image_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    bright_mask = (
        (gray >= 176)
        & (hsv[:, :, 1] <= 96)
        & (hsv[:, :, 2] >= 170)
    ).astype(np.uint8) * 255
    if np.count_nonzero(bright_mask > 0) < max(600, anchor_area * 0.30):
        return None

    kernel_size = max(5, min(11, ((min(crop_bgr.shape[:2]) // 18) | 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    working = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    working = cv2.morphologyEx(working, cv2.MORPH_OPEN, kernel, iterations=1)
    working = cv2.dilate(working, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(working, 8)
    local_anchor = (anchor_rect[0] - x1, anchor_rect[1] - y1, anchor_rect[2] - x1, anchor_rect[3] - y1)
    local_center = rect_center(local_anchor)
    search_area = rect_area(search_rect)
    best_candidate = None
    best_score = 0.0

    for label in range(1, num_labels):
        sx, sy, sw, sh, area = stats[label]
        if area < max(1400, anchor_area * 0.38):
            continue
        if area > min(search_area * 0.92, anchor_area * 10.5 + 12000):
            continue
        component_rect = (int(sx), int(sy), int(sx + sw), int(sy + sh))
        overlap = rect_intersection_area(component_rect, local_anchor) / max(1.0, anchor_area)
        center_hit = (
            0 <= local_center[0] < labels.shape[1]
            and 0 <= local_center[1] < labels.shape[0]
            and labels[local_center[1], local_center[0]] == label
        )
        if not center_hit and overlap < 0.22:
            continue

        component_mask = np.zeros((sh, sw), dtype=np.uint8)
        component_mask[labels[sy:sy + sh, sx:sx + sw] == label] = 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        local_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(local_cnt) < max(900, anchor_area * 0.30):
            continue

        full_cnt = local_cnt.reshape(-1, 2).astype(np.int32)
        full_cnt[:, 0] += x1 + sx
        full_cnt[:, 1] += y1 + sy
        candidate = build_bubble_record(image_bgr, full_cnt, margin=4, source="anchor_component")
        if not candidate:
            continue

        metrics_score = float(candidate.get("bubble_like_score", 0.0))
        fill_ratio = float(candidate.get("fill_ratio", 0.0))
        edge_complexity = float(candidate.get("spiky_edge_complexity", 0.0))
        white_like = (
            float(candidate.get("mean_saturation", 999.0)) <= 70.0
            and float(candidate.get("outline_contrast", 0.0)) >= BUBBLE_OUTLINE_MIN_CONTRAST * 0.58
        )
        panel_like_white_region = (
            white_like
            and not candidate.get("spiky_bubble_like", False)
            and not candidate.get("colored_bubble_like", False)
            and fill_ratio >= 0.84
            and edge_complexity >= 0.95
        )
        if panel_like_white_region:
            continue
        if not (
            metrics_score >= 0.42
            or candidate.get("spiky_bubble_like", False)
            or candidate.get("colored_bubble_like", False)
            or white_like
        ):
            continue

        candidate_score = metrics_score + overlap * 1.2 + (0.8 if center_hit else 0.0)
        if white_like:
            candidate_score += 0.25
        if 0.20 <= fill_ratio <= 0.94:
            candidate_score += 0.20
        if candidate.get("spiky_bubble_like", False):
            candidate_score += 0.30
        if candidate_score > best_score:
            best_score = candidate_score
            best_candidate = candidate

    return best_candidate

def augment_bubbles_from_regions(image_bgr, regions, bubbles):
    updated = list(bubbles)
    for region in regions:
        if region.get("bubble_index", -1) >= 0:
            continue
        rect = clamp_rect(box_to_rect(region["box"]), image_bgr.shape)
        if rect_area(rect) < 420:
            continue
        crop_bgr = image_bgr[rect[1]:rect[3], rect[0]:rect[2]]
        if crop_bgr.size == 0:
            continue
        seed_features = compute_crop_bubble_features(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        if (
            not normalize_rule_text(region.get("easy_text", ""))
            and seed_features.get("colored_ratio", 0.0) >= 0.28
            and not seed_features.get("colored_bubble_like", False)
            and seed_features.get("mean_saturation", 0.0) >= 24.0
        ):
            continue
        candidate = detect_anchor_bubble_candidate(image_bgr, rect)
        updated = add_unique_bubble_candidate(updated, candidate)
    return sorted(updated, key=lambda item: float(item.get("area", 0.0)))

def detect_bubbles_by_edges(image_bgr, canny_low=50, canny_high=150,
                            min_area=3000, max_area_ratio=0.12, margin=5):
    """
    利用边缘检测 + 轮廓查找，检测封闭的气泡区域（不依赖颜色）。
    返回气泡矩形列表 [(x1,y1,x2,y2), ...]
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    dark_lines = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )
    combined = cv2.bitwise_or(edges, dark_lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    area_limit = image_area * max(0.22, max_area_ratio)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > area_limit:
            continue
        candidate = build_bubble_record(image_bgr, cnt.reshape(-1, 2), margin=margin, source="edge")
        if not candidate:
            continue
        if candidate["bubble_like_score"] < 0.35 and candidate["fill_ratio"] < 0.20:
            continue
        bubbles = add_unique_bubble_candidate(bubbles, candidate)
    return sorted(bubbles, key=lambda item: float(item.get("area", 0.0)))
    # 边缘检测
    edges = cv2.Canny(gray, canny_low, canny_high)
    # 闭运算连接断裂边缘
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if area > image_area * max_area_ratio:
            continue
        # 外接矩形并向外扩展 margin
        x, y, w, h = cv2.boundingRect(cnt)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = x + w + margin
        y2 = y + h + margin
        bubbles.append({
            "bbox": (x1, y1, x2, y2),
            "area": float(area),
            "contour": cnt.reshape(-1, 2).tolist(),
        })
    return bubbles

def get_bubble_bbox(bubble):
    if isinstance(bubble, dict):
        return bubble.get("bbox", (0, 0, 0, 0))
    return bubble

def is_inside_any_bubble(bbox, bubbles):
    """判断文本框中心点是否在任一气泡矩形内"""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for bubble in bubbles:
        x1, y1, x2, y2 = get_bubble_bbox(bubble)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False

def contour_contains_point(contour, point):
    if contour is None or len(contour) < 3:
        return False
    pts = np.array(contour, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0

def rect_axis_sample_points(rect):
    x1, y1, x2, y2 = rect
    cx, cy = rect_center(rect)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    points = [(cx, cy)]
    if height >= width:
        points.extend([
            (cx, int(round(y1 + height * 0.28))),
            (cx, int(round(y1 + height * 0.72))),
        ])
    else:
        points.extend([
            (int(round(x1 + width * 0.28)), cy),
            (int(round(x1 + width * 0.72)), cy),
        ])
    return points

def compute_bubble_match_details(rect, bubble):
    x1, y1, x2, y2 = get_bubble_bbox(bubble)
    cx, cy = rect_center(rect)
    region_area = max(1.0, rect_area(rect))
    inter_w = max(0, min(rect[2], x2) - max(rect[0], x1))
    inter_h = max(0, min(rect[3], y2) - max(rect[1], y1))
    overlap = (inter_w * inter_h) / region_area
    center_inside = x1 <= cx <= x2 and y1 <= cy <= y2
    expanded_center_inside = (x1 - 8) <= cx <= (x2 + 8) and (y1 - 8) <= cy <= (y2 + 8)
    contour = bubble.get("contour") if isinstance(bubble, dict) else None
    contour_center_hit = contour_contains_point(contour, (cx, cy))
    axis_hits = sum(1 for point in rect_axis_sample_points(rect) if contour_contains_point(contour, point))
    bubble_mask_hit = contour_center_hit or axis_hits > 0

    if contour_center_hit:
        bubble_match_mode = "contour_center"
    elif axis_hits > 0:
        bubble_match_mode = "contour_axis"
    elif center_inside:
        bubble_match_mode = "bbox_center"
    elif overlap >= 0.08:
        bubble_match_mode = "bbox_overlap"
    elif expanded_center_inside:
        bubble_match_mode = "bbox_expanded"
    else:
        bubble_match_mode = ""

    bubble_like_score = float(bubble.get("bubble_like_score", 0.0)) if isinstance(bubble, dict) else 0.0
    score = overlap * 2.4
    score += 1.35 if contour_center_hit else 0.70 if axis_hits > 0 else 0.0
    score += 1.25 if center_inside else 0.55 if expanded_center_inside else 0.0
    score += min(0.95, bubble_like_score * 0.35)
    if bubble.get("spiky_bubble_like", False):
        score += 0.18
    return {
        "overlap": overlap,
        "center_inside": center_inside,
        "expanded_center_inside": expanded_center_inside,
        "contour_center_hit": contour_center_hit,
        "axis_hits": axis_hits,
        "bubble_mask_hit": bubble_mask_hit,
        "bubble_match_mode": bubble_match_mode,
        "score": score,
    }

def find_enclosing_bubble_index(bbox, bubbles):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    rect = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    region_area = max(1.0, (rect[2] - rect[0]) * (rect[3] - rect[1]))

    best_index = -1
    best_score = 0.0
    best_area = None
    for idx, bubble in enumerate(bubbles):
        details = compute_bubble_match_details(rect, bubble)
        if details["overlap"] < 0.08 and not details["expanded_center_inside"] and not details["bubble_mask_hit"]:
            continue

        score = details["score"]
        x1, y1, x2, y2 = get_bubble_bbox(bubble)
        area = max(1.0, (x2 - x1) * (y2 - y1))
        if best_index < 0 or score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and (best_area is None or area < best_area)):
            best_index = idx
            best_score = score
            best_area = area
    return best_index

    best_index = -1
    best_area = None
    for idx, bubble in enumerate(bubbles):
        x1, y1, x2, y2 = get_bubble_bbox(bubble)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = max(1, (x2 - x1) * (y2 - y1))
            if best_area is None or area < best_area:
                best_area = area
                best_index = idx
    return best_index

def normalize_rule_text(text):
    return "".join(str(text).split()).strip()

def is_hiragana(ch):
    return "\u3040" <= ch <= "\u309f"

def is_katakana(ch):
    return "\u30a0" <= ch <= "\u30ff"

def is_kana_char(ch):
    return is_hiragana(ch) or is_katakana(ch)

def is_kana_only(text):
    return bool(text) and all(is_kana_char(ch) for ch in text)

def contains_kanji(text):
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)

def looks_like_japanese_sentence(text):
    compact = normalize_rule_text(text)
    if not compact:
        return False
    if any(mark in compact for mark in "。！？…"):
        return True
    if contains_kanji(compact) and len(compact) >= 4:
        return True
    return len(compact) >= 6 and not is_kana_only(compact)

def box_to_rect(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def polygon_area(box):
    pts = np.array(box, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)

def rect_to_box(rect):
    x1, y1, x2, y2 = rect
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def rect_area(rect):
    x1, y1, x2, y2 = rect
    return max(0, x2 - x1) * max(0, y2 - y1)

def rect_center(rect):
    x1, y1, x2, y2 = rect
    return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)

def clamp_rect(rect, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = rect
    x1 = max(0, min(w, int(round(x1))))
    y1 = max(0, min(h, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    return x1, y1, x2, y2

def shrink_rect(rect, image_shape, padding=2):
    x1, y1, x2, y2 = rect
    return clamp_rect((x1 + padding, y1 + padding, x2 - padding, y2 - padding), image_shape)

def rect_intersection_area(rect_a, rect_b):
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    return inter_w * inter_h

def expand_rect(rect, image_shape, padding=6):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = rect
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(w, x2 + padding),
        min(h, y2 + padding),
    )

def rect_to_mask(image_shape, rect):
    x1, y1, x2, y2 = clamp_rect(rect, image_shape)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask

def clip_mask_to_rect(mask, rect):
    x1, y1, x2, y2 = rect
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped

def contour_to_mask(image_shape, contour):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if contour is None or len(contour) < 3:
        return mask
    pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def mask_to_rect(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

def smooth_mask(mask, kernel_size=5):
    if kernel_size <= 1:
        return mask
    kernel_size = max(3, kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    return smoothed

def grow_mask(mask, pixels):
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    return cv2.dilate(mask, kernel, iterations=1)

def shrink_mask(mask, pixels):
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    return cv2.erode(mask, kernel, iterations=1)

def mask_contains_point(mask, point):
    x, y = point
    h, w = mask.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return False
    return bool(mask[y, x] > 0)

def crop_by_rect(image_rgb, rect):
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop

def has_white_background(crop_rgb, bright_threshold=205, mean_threshold=185, bright_ratio_threshold=0.55):
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    bright_ratio = np.mean(gray >= bright_threshold)
    soft_bright_ratio = np.mean(gray >= 188)
    mean_brightness = float(np.mean(gray))
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    upper_quartile = float(np.percentile(gray, 75))
    return (
        (
            bright_ratio >= bright_ratio_threshold
            and mean_brightness >= mean_threshold
            and mean_saturation <= 42
        )
        or (
            soft_bright_ratio >= 0.66
            and upper_quartile >= 210
            and mean_saturation <= 48
        )
    )

def compute_mask_confidence(mask, anchor_rect):
    if mask is None or not np.any(mask > 0):
        return 0.0
    x1, y1, x2, y2 = anchor_rect
    anchor_area = max(1, rect_area(anchor_rect))
    covered = np.count_nonzero(mask[y1:y2, x1:x2] > 0) / anchor_area
    center_hit = 1.0 if mask_contains_point(mask, rect_center(anchor_rect)) else 0.0
    mask_rect = mask_to_rect(mask)
    if mask_rect is None:
        return 0.0
    rect_overlap = rect_intersection_area(mask_rect, anchor_rect) / anchor_area
    return float(min(1.0, 0.45 * covered + 0.35 * center_hit + 0.20 * rect_overlap))

def sample_fill_color(image_rgb, mask, fallback=(250, 250, 250)):
    pixels = image_rgb[mask > 0]
    if pixels.size == 0:
        return fallback
    brightness = np.mean(pixels, axis=1)
    threshold = np.percentile(brightness, 65)
    bright_pixels = pixels[brightness >= threshold]
    sample = bright_pixels if bright_pixels.size else pixels
    median = np.median(sample, axis=0)
    median = np.clip(median, 235, 255).astype(np.uint8)
    return int(median[0]), int(median[1]), int(median[2])

def build_masked_crop(image_rgb, rect, mask, fallback=(245, 245, 245)):
    crop = crop_by_rect(image_rgb, rect)
    if crop is None:
        return None
    x1, y1, x2, y2 = rect
    local_mask = mask[y1:y2, x1:x2]
    if local_mask.size == 0:
        return crop.copy()
    masked = crop.copy()
    pixels = masked[local_mask > 0]
    if pixels.size:
        fill_color = np.median(pixels, axis=0).astype(np.uint8)
    else:
        fill_color = np.array(fallback, dtype=np.uint8)
    masked[local_mask == 0] = fill_color
    return masked

def prepare_ocr_crop_pil(crop_rgb):
    crop_pil = Image.fromarray(crop_rgb)
    if crop_pil.width > 400:
        ratio = 400.0 / crop_pil.width
        new_h = max(1, int(crop_pil.height * ratio))
        crop_pil = crop_pil.resize((400, new_h), Image.Resampling.LANCZOS)
    return crop_pil

def choose_bubble_crop_for_ocr(rgb_image, anchor_rect, bubble):
    original_crop = crop_by_rect(rgb_image, anchor_rect)
    meta = {
        "ocr_crop_mode": "original",
        "ocr_crop_rect": list(anchor_rect),
        "ocr_crop_from_bubble_mask": False,
    }
    if original_crop is None or not ENABLE_BUBBLE_CONSTRAINED_OCR_CROP:
        return original_crop, meta

    if bubble is not None:
        contour = bubble.get("contour")
        full_mask = contour_to_mask(rgb_image.shape, contour)
    else:
        gray = cv2.cvtColor(original_crop, cv2.COLOR_RGB2GRAY)
        ink_local = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
        ink_local = cv2.morphologyEx(ink_local, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        ink_local = cv2.morphologyEx(ink_local, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        full_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = anchor_rect
        full_mask[y1:y2, x1:x2] = ink_local[:y2 - y1, :x2 - x1]

    if not np.any(full_mask > 0):
        return original_crop, meta

    anchor_w = max(1, anchor_rect[2] - anchor_rect[0])
    anchor_h = max(1, anchor_rect[3] - anchor_rect[1])
    local_rect = expand_rect(anchor_rect, rgb_image.shape, padding=max(8, min(20, max(anchor_w, anchor_h) // 3)))
    local_mask = clip_mask_to_rect(full_mask, local_rect)
    if not np.any(local_mask > 0):
        return original_crop, meta

    local_mask = smooth_mask(local_mask, kernel_size=3)
    local_mask = shrink_mask(local_mask, 1)
    if not np.any(local_mask > 0):
        return original_crop, meta

    constrained_rect = compute_content_rect_from_mask(
        local_mask,
        anchor_rect,
        rgb_image.shape,
        search_paddings=(4, 10, 16),
        margin=1,
    )
    constrained_rect = expand_rect(constrained_rect, rgb_image.shape, padding=3)
    if rect_area(constrained_rect) < rect_area(anchor_rect) * 0.42:
        constrained_rect = expand_rect(anchor_rect, rgb_image.shape, padding=4)

    constrained_crop = build_masked_crop(rgb_image, constrained_rect, full_mask)
    if constrained_crop is None or constrained_crop.size == 0:
        return original_crop, meta

    meta.update({
        "ocr_crop_mode": "bubble_constrained" if bubble is not None else "ink_constrained",
        "ocr_crop_rect": list(constrained_rect),
        "ocr_crop_from_bubble_mask": bubble is not None,
    })
    return constrained_crop, meta

def extract_content_rect(mask, fallback_rect, image_shape, margin=6):
    working = mask.copy()
    if margin > 0:
        working = shrink_mask(working, margin)
        if not np.any(working > 0):
            working = mask.copy()

    row_counts = np.count_nonzero(working > 0, axis=1)
    col_counts = np.count_nonzero(working > 0, axis=0)
    row_idx = np.where(row_counts >= max(1, row_counts.max() * 0.45))[0] if row_counts.max() > 0 else np.array([])
    col_idx = np.where(col_counts >= max(1, col_counts.max() * 0.45))[0] if col_counts.max() > 0 else np.array([])

    if len(col_idx) >= 2 and len(row_idx) >= 2:
        rect = (int(col_idx[0]), int(row_idx[0]), int(col_idx[-1]) + 1, int(row_idx[-1]) + 1)
    else:
        rect = mask_to_rect(working)

    if rect is None:
        return shrink_rect(fallback_rect, image_shape, padding=2)
    rect = clamp_rect(rect, image_shape)
    if rect_area(rect) <= 0:
        return shrink_rect(fallback_rect, image_shape, padding=2)
    return rect

def compute_content_rect_from_mask(mask, anchor_rect, image_shape, search_paddings, margin=6):
    for padding in search_paddings:
        search_rect = expand_rect(anchor_rect, image_shape, padding=padding)
        x1, y1, x2, y2 = search_rect
        local_mask = np.zeros_like(mask)
        local_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        rect = extract_content_rect(local_mask, search_rect, image_shape, margin=margin)
        if rect_area(rect) >= rect_area(anchor_rect) * 0.85:
            return rect
    return extract_content_rect(mask, anchor_rect, image_shape, margin=margin)

def build_core_content_rect(content_rect, anchor_rect, mask_rect, image_shape, region_type, render_mode):
    content_rect = clamp_rect(content_rect, image_shape)
    mask_rect = clamp_rect(mask_rect or content_rect, image_shape)
    if rect_area(content_rect) <= 0:
        return content_rect

    cx, cy = rect_center(anchor_rect)
    content_w = max(1, content_rect[2] - content_rect[0])
    content_h = max(1, content_rect[3] - content_rect[1])
    anchor_w = max(1, anchor_rect[2] - anchor_rect[0])
    anchor_h = max(1, anchor_rect[3] - anchor_rect[1])
    mask_w = max(1, mask_rect[2] - mask_rect[0])
    mask_h = max(1, mask_rect[3] - mask_rect[1])

    if region_type == "narration_box":
        width_ratio = min(RENDER_MAX_FILL_RATIO, 0.70)
        height_ratio = 0.84 if content_h >= content_w else 0.76
    elif render_mode == "bubble_mask":
        width_ratio = RENDER_MAX_FILL_RATIO
        height_ratio = 0.88
    else:
        width_ratio = min(0.86, RENDER_MAX_FILL_RATIO + 0.08)
        height_ratio = 0.90

    target_w = min(content_w, max(anchor_w + 2 * RENDER_CORE_SHRINK, int(content_w * width_ratio)))
    target_h = min(content_h, max(anchor_h + 2 * RENDER_CORE_SHRINK, int(content_h * height_ratio)))

    if content_w > content_h * 1.15:
        target_w = min(target_w, max(anchor_w + 2, int(content_w * 0.66)))
    if content_h > content_w * 1.6:
        target_h = min(content_h, max(anchor_h + 2 * RENDER_CORE_SHRINK, int(content_h * 0.92)))

    target_w = min(target_w, mask_w)
    target_h = min(target_h, mask_h)
    core_rect = build_centered_rect((cx, cy), target_w, target_h, mask_rect, image_shape)
    if rect_area(core_rect) <= 0:
        return content_rect
    return core_rect

def build_core_render_mask(render_mask, core_rect, image_shape, region_type, render_mode):
    if render_mask is None:
        return rect_to_mask(image_shape, core_rect)

    clip_rect = expand_rect(core_rect, image_shape, padding=max(1, RENDER_CORE_SHRINK // 2))
    core_mask = clip_mask_to_rect(render_mask, clip_rect)
    if not np.any(core_mask > 0):
        core_mask = rect_to_mask(image_shape, clip_rect)
    if render_mode == "bubble_mask":
        core_mask = smooth_mask(core_mask, kernel_size=3)
        core_mask = shrink_mask(core_mask, 1)
    elif region_type == "narration_box":
        core_mask = smooth_mask(core_mask, kernel_size=3)
    if not np.any(core_mask > 0):
        return rect_to_mask(image_shape, core_rect)
    return core_mask

def refine_fill_mask_to_layout(core_mask, content_rect, layout_meta, image_shape, vertical=True):
    if core_mask is None or not np.any(core_mask > 0):
        return core_mask
    total_width = int(layout_meta.get("total_width", 0))
    font_size = int(layout_meta.get("font_size", 0))
    if total_width <= 0 or rect_area(content_rect) <= 0:
        return core_mask

    rect_w = max(1, content_rect[2] - content_rect[0])
    rect_h = max(1, content_rect[3] - content_rect[1])
    cx, cy = rect_center(content_rect)
    if vertical:
        target_w = min(rect_w, max(total_width + 10, int(rect_w * 0.42)))
        target_h = min(rect_h, max(int(rect_h * 0.78), font_size * 3))
    else:
        target_w = min(rect_w, max(total_width + 12, int(rect_w * 0.68)))
        target_h = min(rect_h, max(font_size * 2 + 10, int(rect_h * 0.48)))

    layout_rect = build_centered_rect((cx, cy), target_w, target_h, content_rect, image_shape)
    refined_mask = clip_mask_to_rect(core_mask, expand_rect(layout_rect, image_shape, padding=2))
    if not np.any(refined_mask > 0):
        return core_mask
    return smooth_mask(refined_mask, kernel_size=3)

def build_small_fallback_rect(anchor_rect, image_shape, padding=None):
    if padding is None:
        padding = NARRATION_SMALL_FALLBACK_PADDING
    return expand_rect(anchor_rect, image_shape, padding=padding)

def build_centered_rect(center, width, height, bounds_rect, image_shape):
    cx, cy = center
    bx1, by1, bx2, by2 = bounds_rect
    width = max(1, int(round(width)))
    height = max(1, int(round(height)))
    x1 = cx - width // 2
    y1 = cy - height // 2
    x2 = x1 + width
    y2 = y1 + height

    if x1 < bx1:
        x2 += bx1 - x1
        x1 = bx1
    if x2 > bx2:
        x1 -= x2 - bx2
        x2 = bx2
    if y1 < by1:
        y2 += by1 - y1
        y1 = by1
    if y2 > by2:
        y1 -= y2 - by2
        y2 = by2

    return clamp_rect((max(bx1, x1), max(by1, y1), min(bx2, x2), min(by2, y2)), image_shape)

def compute_narration_content_rect(mask, anchor_rect, image_shape):
    base_rect = compute_content_rect_from_mask(mask, anchor_rect, image_shape, search_paddings=(4, 8, 14), margin=4)
    mask_rect = mask_to_rect(mask)
    if mask_rect is None:
        return shrink_rect(anchor_rect, image_shape, padding=1)

    mx1, my1, mx2, my2 = mask_rect
    ax1, ay1, ax2, ay2 = anchor_rect
    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    mw = max(1, mx2 - mx1)
    mh = max(1, my2 - my1)
    center = rect_center(anchor_rect)

    if mh >= mw:
        target_w = min(mw, max(aw + 2 * NARRATION_CORE_SHRINK, int(mw * 0.62)))
        target_h = min(mh, max(ah + 2 * NARRATION_CORE_SHRINK, int(mh * 0.9)))
    else:
        target_w = min(mw, max(aw + 2 * NARRATION_CORE_SHRINK, int(mw * 0.9)))
        target_h = min(mh, max(ah + 2 * NARRATION_CORE_SHRINK, int(mh * 0.62)))

    refined = build_centered_rect(center, target_w, target_h, mask_rect, image_shape)
    if rect_area(refined) < rect_area(anchor_rect) * 0.8:
        return base_rect
    return refined

def regularize_narration_mask(local_mask, search_rect, anchor_rect, image_shape, rect_fill):
    sx1, sy1, _, _ = search_rect
    rel_anchor = (
        anchor_rect[0] - sx1,
        anchor_rect[1] - sy1,
        anchor_rect[2] - sx1,
        anchor_rect[3] - sy1,
    )
    anchor_w = max(1, rel_anchor[2] - rel_anchor[0])
    anchor_h = max(1, rel_anchor[3] - rel_anchor[1])

    if rect_fill >= 0.74:
        ys, xs = np.where(local_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return local_mask
        rect_mask = np.zeros_like(local_mask)
        rect_mask[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1] = 255
        local_mask = rect_mask
    else:
        core_rect = (
            max(0, rel_anchor[0] - max(4, anchor_w // 4)),
            max(0, rel_anchor[1] - max(4, anchor_h // 5)),
            min(local_mask.shape[1], rel_anchor[2] + max(4, anchor_w // 4)),
            min(local_mask.shape[0], rel_anchor[3] + max(4, anchor_h // 5)),
        )
        local_mask = clip_mask_to_rect(local_mask, core_rect)

    local_mask = smooth_mask(local_mask, kernel_size=3)
    local_mask = shrink_mask(local_mask, max(1, NARRATION_CORE_SHRINK // 2))
    if not np.any(local_mask > 0):
        return local_mask

    final_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = search_rect
    final_mask[y1:y2, x1:x2] = local_mask[:y2 - y1, :x2 - x1]
    return final_mask

def build_bubble_render_mask(image_shape, bubble):
    mask = contour_to_mask(image_shape, bubble.get("contour"))
    if not np.any(mask > 0):
        return mask
    mask = smooth_mask(mask, kernel_size=5)
    mask = grow_mask(mask, BUBBLE_MASK_PADDING)
    mask = smooth_mask(mask, kernel_size=5)
    mask = shrink_mask(mask, BUBBLE_BORDER_PROTECT)
    return smooth_mask(mask, kernel_size=3)

def detect_narration_render_mask(image_rgb, anchor_rect):
    anchor_w = max(1, anchor_rect[2] - anchor_rect[0])
    anchor_h = max(1, anchor_rect[3] - anchor_rect[1])
    search_padding = max(8, min(18, max(anchor_w, anchor_h) // 3))
    search_rect = expand_rect(anchor_rect, image_rgb.shape, padding=search_padding)
    crop = crop_by_rect(image_rgb, search_rect)
    debug_meta = {
        "search_rect": list(search_rect),
        "candidate_count": 0,
        "candidate_area_ratio": 0.0,
        "shape_score": 0.0,
        "rejection_reason": "",
        "fallback_mode": "",
    }
    if crop is None:
        debug_meta["rejection_reason"] = "empty_search_crop"
        return None, 0.0, None, debug_meta

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    white_mask = np.where((gray >= 205) & (hsv[:, :, 1] <= 55), 255, 0).astype(np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        debug_meta["rejection_reason"] = "no_white_candidates"
        return None, 0.0, None, debug_meta

    sx1, sy1, sx2, sy2 = search_rect
    rel_anchor = (
        anchor_rect[0] - sx1,
        anchor_rect[1] - sy1,
        anchor_rect[2] - sx1,
        anchor_rect[3] - sy1,
    )
    rel_cx, rel_cy = rect_center(rel_anchor)
    anchor_area = max(1, rect_area(anchor_rect))
    search_area = max(1, rect_area(search_rect))
    valid_candidates = 0
    best = None
    best_score = -1.0
    rejection_reasons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < NARRATION_MASK_MIN_AREA:
            rejection_reasons.append("candidate_too_small")
            continue

        local_mask = np.zeros_like(white_mask)
        cv2.fillPoly(local_mask, [cnt], 255)
        x, y, w, h = cv2.boundingRect(cnt)
        candidate_rect = (sx1 + x, sy1 + y, sx1 + x + w, sy1 + y + h)
        coverage = np.count_nonzero(local_mask[rel_anchor[1]:rel_anchor[3], rel_anchor[0]:rel_anchor[2]] > 0) / anchor_area
        center_hit = 1.0 if (0 <= rel_cx < local_mask.shape[1] and 0 <= rel_cy < local_mask.shape[0] and local_mask[rel_cy, rel_cx] > 0) else 0.0
        anchor_overlap = rect_intersection_area(candidate_rect, anchor_rect) / anchor_area
        area_ratio_anchor = float(area / anchor_area)
        area_ratio_search = float(area / search_area)
        rect_fill = float(area / max(1, w * h))
        aspect = (w / h) if h else 999.0
        aspect_score = 1.0 if 0.18 <= aspect <= 5.6 else 0.45
        edge_touch = 1.0 if (x <= 1 or y <= 1 or x + w >= local_mask.shape[1] - 1 or y + h >= local_mask.shape[0] - 1) else 0.0
        separability = max(0.0, 1.0 - area_ratio_search)
        shape_score = min(
            1.0,
            coverage * 0.30 +
            center_hit * 0.18 +
            anchor_overlap * 0.16 +
            rect_fill * 0.18 +
            aspect_score * 0.10 +
            separability * 0.08 -
            edge_touch * 0.18
        )

        if center_hit < 1.0:
            rejection_reasons.append("center_miss")
            continue
        if anchor_overlap < 0.45:
            rejection_reasons.append("anchor_overlap_low")
            continue
        if area_ratio_anchor > NARRATION_MAX_AREA_RATIO_TO_ANCHOR:
            rejection_reasons.append("area_ratio_anchor_too_large")
            continue
        if area_ratio_search > NARRATION_MAX_AREA_RATIO_TO_SEARCH:
            rejection_reasons.append("area_ratio_search_too_large")
            continue

        valid_candidates += 1
        if shape_score > best_score:
            best_score = shape_score
            best = {
                "local_mask": local_mask,
                "candidate_rect": candidate_rect,
                "coverage": coverage,
                "rect_fill": rect_fill,
                "center_hit": center_hit,
                "area_ratio_anchor": area_ratio_anchor,
                "shape_score": shape_score,
            }

    debug_meta["candidate_count"] = valid_candidates

    if best is None:
        debug_meta["rejection_reason"] = rejection_reasons[0] if rejection_reasons else "no_valid_candidate"
        return None, 0.0, None, debug_meta

    debug_meta["candidate_area_ratio"] = round(best["area_ratio_anchor"], 4)
    debug_meta["shape_score"] = round(best["shape_score"], 4)

    if best["shape_score"] < NARRATION_MIN_SHAPE_SCORE:
        debug_meta["rejection_reason"] = "shape_score_too_low"
        return None, 0.0, best["candidate_rect"], debug_meta

    mask = regularize_narration_mask(
        best["local_mask"],
        search_rect,
        anchor_rect,
        image_rgb.shape,
        best["rect_fill"],
    )
    mask_rect = mask_to_rect(mask)
    if mask_rect is None or rect_area(mask_rect) <= 0:
        debug_meta["rejection_reason"] = "mask_regularize_failed"
        return None, 0.0, best["candidate_rect"], debug_meta

    confidence = min(
        1.0,
        best["coverage"] * 0.36 +
        best["center_hit"] * 0.14 +
        best["rect_fill"] * 0.12 +
        best["shape_score"] * 0.38
    )
    return mask, float(confidence), mask_rect, debug_meta

def build_render_target(region, region_type, info, rgb_image, bubbles):
    anchor_rect = info["rect"]
    image_shape = rgb_image.shape
    inside_bubble = info["inside_bubble"]
    render_mode = "rect_fallback"
    render_mask = None
    core_mask = None
    bubble_bbox = None
    mask_confidence = 0.0
    fallback_used = True
    search_rect = None
    candidate_count = 0
    candidate_area_ratio = 0.0
    shape_score = 0.0
    rejection_reason = ""
    fallback_mode = "rect_fallback"

    if region_type == "dialogue_bubble" and region.get("bubble_index", -1) >= 0:
        bubble = bubbles[region["bubble_index"]]
        render_mask = build_bubble_render_mask(image_shape, bubble)
        mask_confidence = compute_mask_confidence(render_mask, anchor_rect)
        bubble_bbox = list(get_bubble_bbox(bubble))
        if mask_confidence >= MASK_CONFIDENCE_THRESHOLD:
            render_mode = "bubble_mask"
            fallback_used = False
        else:
            render_mask = None

    if region_type == "narration_box" and render_mask is None:
        narration_mask, narration_conf, narration_rect, narration_meta = detect_narration_render_mask(rgb_image, anchor_rect)
        search_rect = narration_meta.get("search_rect")
        candidate_count = narration_meta.get("candidate_count", 0)
        candidate_area_ratio = narration_meta.get("candidate_area_ratio", 0.0)
        shape_score = narration_meta.get("shape_score", 0.0)
        rejection_reason = narration_meta.get("rejection_reason", "")
        if narration_mask is not None and narration_conf >= MASK_CONFIDENCE_THRESHOLD:
            render_mask = narration_mask
            render_mode = "narration_mask"
            mask_confidence = narration_conf
            fallback_used = False
            bubble_bbox = list(narration_rect) if narration_rect is not None else None
            fallback_mode = ""

    if render_mask is None:
        if region_type == "narration_box":
            render_rect = build_small_fallback_rect(anchor_rect, image_shape)
            fallback_mode = "narration_small_rect"
        else:
            render_rect = expand_rect(anchor_rect, image_shape, padding=4 if inside_bubble else 2)
        render_mask = rect_to_mask(image_shape, render_rect)
        bubble_bbox = list(render_rect)
    else:
        local_rect = expand_rect(
            anchor_rect,
            image_shape,
            padding=18 if render_mode == "bubble_mask" else 10,
        )
        local_mask = clip_mask_to_rect(render_mask, local_rect)
        if np.any(local_mask > 0):
            render_mask = local_mask

    if render_mode == "narration_mask":
        content_rect = compute_narration_content_rect(render_mask, anchor_rect, image_shape)
    else:
        content_rect = compute_content_rect_from_mask(
            render_mask,
            anchor_rect,
            image_shape,
            search_paddings=(8, 16, 26) if render_mode == "bubble_mask" else (5, 12, 20),
            margin=8 if render_mode != "rect_fallback" else 5,
        )
    mask_rect = mask_to_rect(render_mask) or content_rect
    core_content_rect = build_core_content_rect(
        content_rect,
        anchor_rect,
        mask_rect,
        image_shape,
        region_type,
        render_mode,
    )
    core_mask = build_core_render_mask(render_mask, core_content_rect, image_shape, region_type, render_mode)
    if core_mask is None or not np.any(core_mask > 0):
        core_mask = rect_to_mask(image_shape, core_content_rect)
    fill_color = sample_fill_color(rgb_image, core_mask if np.any(core_mask > 0) else render_mask)
    fill_alpha = 200 if render_mode == "bubble_mask" else 224 if region_type == "narration_box" else 216
    render_shrink_applied = rect_area(core_content_rect) + 1 < rect_area(content_rect)
    core_mask_area_ratio = rect_area(mask_to_rect(core_mask) or core_content_rect) / max(1.0, rect_area(mask_rect))

    return {
        "render_mode": render_mode,
        "render_mask": render_mask,
        "core_mask": core_mask,
        "content_rect": content_rect,
        "core_content_rect": core_content_rect,
        "mask_confidence": round(float(mask_confidence), 4),
        "fill_color": list(fill_color),
        "fill_alpha": fill_alpha,
        "fallback_used": fallback_used,
        "bubble_bbox": bubble_bbox,
        "search_rect": search_rect,
        "candidate_count": candidate_count,
        "candidate_area_ratio": candidate_area_ratio,
        "shape_score": shape_score,
        "rejection_reason": rejection_reason,
        "fallback_mode": fallback_mode,
        "core_mask_area_ratio": round(float(core_mask_area_ratio), 4),
        "render_shrink_applied": render_shrink_applied,
    }

def apply_mask_fill(pil_img, render_mask, fill_color, fill_alpha=220):
    if render_mask is None or not np.any(render_mask > 0):
        return pil_img.copy()
    base = pil_img.convert("RGBA")
    overlay = np.zeros((render_mask.shape[0], render_mask.shape[1], 4), dtype=np.uint8)
    alpha = render_mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.15, sigmaY=1.15)
    alpha = np.clip(alpha * float(fill_alpha), 0, 255).astype(np.uint8)
    overlay[:, :, 0] = fill_color[0]
    overlay[:, :, 1] = fill_color[1]
    overlay[:, :, 2] = fill_color[2]
    overlay[:, :, 3] = alpha
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    return Image.alpha_composite(base, overlay_img).convert("RGB")

def is_punctuation_only(text):
    punctuation_chars = set("…...。、「」『』【】（）()！？!?，,、・ー-～〜 ")
    return bool(text) and all(ch in punctuation_chars for ch in text)

def strip_text_noise(text):
    return normalize_rule_text(text).strip(" \t\r\n.,!?~")

def detect_sfx_traits(text):
    compact = normalize_rule_text(text)
    stripped = compact.strip("…...。、「」『』【】（）()！？!?，,、・ー-～〜")
    traits = []
    if not stripped:
        return traits
    if stripped in SFX_KEYWORDS:
        traits.append("sound_effect_dictionary_hit")
    if len(stripped) <= SFX_SKIP_MAX_LEN and is_kana_only(stripped):
        if all(is_katakana(ch) for ch in stripped):
            traits.append("katakana_short")
        if len(set(stripped)) <= 2 and any(is_katakana(ch) for ch in stripped):
            traits.append("repeated_syllable")
        if stripped[-1] in {"ッ", "ー"}:
            traits.append("sokuon_or_long_vowel_tail")
        if any(ch in stripped for ch in {"ッ", "ー"}) and any(is_katakana(ch) for ch in stripped):
            traits.append("sokuon_or_long_vowel_tail")
    return list(dict.fromkeys(traits))

def looks_like_sfx(text):
    return bool(detect_sfx_traits(text))

def is_short_dialogue_candidate(text, inside_bubble):
    stripped = normalize_rule_text(text).strip("…...。、「」『』【】（）()！？!?，,、・ー-～〜")
    if not stripped:
        return False
    if stripped in SHORT_DIALOGUE_WHITELIST:
        return True
    if len(stripped) > 8:
        return False
    has_dialogue_punct = any(mark in text for mark in "！？!?…")
    if inside_bubble and has_dialogue_punct:
        return True
    if inside_bubble and any(mark in text for mark in {"ー", "〜", "～"}) and not looks_like_sfx(stripped):
        return True
    if inside_bubble and stripped.endswith(("か", "よ", "ね", "の", "だ")) and not looks_like_sfx(stripped):
        return True
    if inside_bubble and stripped.endswith("?") and not looks_like_sfx(stripped):
        return True
    return False

"""
def has_dialogue_marks(text):
    compact = normalize_rule_text(text)
    return any(mark in compact for mark in DIALOGUE_PUNCT_MARKS) or "..." in compact

def bubble_reaction_hint(text):
    stripped = normalize_rule_text(text).strip("鈥?..銆傘€併€屻€嶃€庛€忋€愩€戯紙锛?)锛侊紵!?锛?銆併兓銉?锝炪€?)
    return stripped in BUBBLE_REACTION_WORDS

def looks_like_sfx(text, inside_bubble=False):
    compact = normalize_rule_text(text)
    stripped = compact.strip("鈥?..銆傘€併€屻€嶃€庛€忋€愩€戯紙锛?)锛侊紵!?锛?銆併兓銉?锝炪€?)
    traits = detect_sfx_traits(compact)
    if not inside_bubble:
        return bool(traits)
    if not stripped or not traits:
        return False
    if contains_kanji(stripped) or has_dialogue_marks(compact) or bubble_reaction_hint(stripped):
        return False
    if "sound_effect_dictionary_hit" in traits:
        return True
    return (
        len(stripped) <= BUBBLE_SHORT_TEXT_MAX_LEN
        and len(traits) >= 2
        and all(is_katakana(ch) for ch in stripped if is_kana_char(ch))
    )

def is_short_dialogue_candidate(text, inside_bubble):
    compact = normalize_rule_text(text)
    stripped = compact.strip("鈥?..銆傘€併€屻€嶃€庛€忋€愩€戯紙锛?)锛侊紵!?锛?銆併兓銉?锝炪€?)
    if not stripped:
        return False
    if stripped in SHORT_DIALOGUE_WHITELIST or bubble_reaction_hint(stripped):
        return True
    if len(stripped) > 8:
        return False
    if inside_bubble and has_dialogue_marks(compact):
        return True
    if inside_bubble and compact.startswith(("\u3053\u3001", "\u3042\u3001", "\u3048\u3001")):
        return True
    if inside_bubble and any(mark in compact for mark in {"\u30fc", "\u301c", "\uff5e"}) and not looks_like_sfx(stripped, inside_bubble=True):
        return True
    if inside_bubble and stripped.endswith(("\u304b", "\u3088", "\u306d", "\u306e", "\u3060")) and not looks_like_sfx(stripped, inside_bubble=True):
        return True
    if inside_bubble and len(stripped) <= BUBBLE_SHORT_TEXT_MAX_LEN and contains_kanji(stripped):
        return True
    if inside_bubble and len(stripped) <= 2 and all(is_hiragana(ch) for ch in stripped):
        return not looks_like_sfx(stripped, inside_bubble=True)
    return False

def score_bubble_text_candidate(text, confidence=0.0, source="manga_ocr"):
    compact = normalize_rule_text(text)
    stripped = compact.strip("鈥?..銆傘€併€屻€嶃€庛€忋€愩€戯紙锛?)锛侊紵!?锛?銆併兓銉?锝炪€?)
    if not stripped:
        return 0.0, []

    score = 0.0
    reasons = []
    if has_dialogue_marks(compact):
        score += 1.4
        reasons.append("dialogue_marks")
    if bubble_reaction_hint(stripped):
        score += 1.3
        reasons.append("reaction_hint")
    if is_short_dialogue_candidate(compact, True):
        score += 1.1
        reasons.append("short_dialogue_pattern")
    if contains_kanji(stripped):
        score += 0.7
        reasons.append("contains_kanji")
    if source == "manga_ocr":
        score += 0.35
        reasons.append("manga_ocr_bias")
    if confidence >= BUBBLE_RESCUE_MIN_CONFIDENCE:
        score += 0.6
        reasons.append("confidence_ok")
    elif source == "easyocr":
        score -= 0.4
        reasons.append("easyocr_low_conf")
    if len(stripped) <= BUBBLE_SHORT_TEXT_MAX_LEN:
        score += 0.25
        reasons.append("short_len")
    if looks_like_sfx(stripped, inside_bubble=True):
        score -= 2.2
        reasons.append("bubble_sfx_like")
    return score, reasons

def choose_bubble_ocr_candidate(manga_text, easy_text, easy_conf, region_type):
    manga_ok, manga_reason = is_valid_ocr_text(
        manga_text,
        inside_bubble=True,
        region_type=region_type,
        source="manga_ocr",
        confidence=1.0
    )
    easy_ok, easy_reason = is_valid_ocr_text(
        easy_text,
        inside_bubble=True,
        region_type=region_type,
        source="easyocr",
        confidence=easy_conf
    )
    manga_score, manga_reasons = score_bubble_text_candidate(manga_text, 1.0, source="manga_ocr")
    easy_score, easy_reasons = score_bubble_text_candidate(easy_text, easy_conf, source="easyocr")
    manga_compact = normalize_rule_text(manga_text)
    easy_compact = normalize_rule_text(easy_text)

    if manga_ok and easy_ok:
        if (
            BUBBLE_OCR_FALLBACK_ALLOW_SHORT
            and len(easy_compact) >= len(manga_compact) + 1
            and easy_score >= manga_score + 0.45
            and not looks_like_sfx(easy_compact, inside_bubble=True)
        ):
            return easy_compact, "easyocr_bubble_override", "easy_candidate_more_complete", easy_score
        return manga_compact, "manga_ocr", "manga_valid", manga_score

    if manga_ok:
        return manga_compact, "manga_ocr", manga_reason, manga_score

    if easy_ok and (len(easy_compact) >= MANGA_OCR_FALLBACK_MIN_LEN or BUBBLE_OCR_FALLBACK_ALLOW_SHORT):
        return easy_compact, "easyocr_fallback", easy_reason, easy_score

    if ENABLE_BUBBLE_TEXT_RESCUE:
        candidates = [
            (manga_compact, manga_score, "bubble_text_rescue", "manga:" + ",".join(manga_reasons or [manga_reason])),
            (easy_compact, easy_score, "bubble_text_rescue", "easy:" + ",".join(easy_reasons or [easy_reason])),
        ]
        candidates = [item for item in candidates if item[0] and not looks_like_sfx(item[0], inside_bubble=True)]
        if candidates:
            best_text, best_score, choice, reason = max(candidates, key=lambda item: item[1])
            if best_score >= 1.45:
                return best_text, choice, reason, best_score

    return "", "", f"{manga_reason}/{easy_reason}", max(manga_score, easy_score)

def count_japanese_chars(text):
    return sum(1 for ch in normalize_rule_text(text) if is_kana_char(ch) or contains_kanji(ch))

def detect_ocr_anomaly(text, source_used, confidence, inside_bubble, region_type):
    compact = normalize_rule_text(text)
    stripped = strip_text_noise(compact)
    if not stripped:
        return True, "empty_text"
    if is_rescuable_short_dialogue(compact, inside_bubble, region_type):
        return False, ""
    if is_short_dialogue_candidate(compact, inside_bubble):
        return False, ""
    if looks_like_japanese_sentence(compact):
        return False, ""

    jp_chars = count_japanese_chars(stripped)
    ascii_letters = sum(1 for ch in stripped if ch.isascii() and ch.isalpha())
    ascii_digits = sum(1 for ch in stripped if ch.isascii() and ch.isdigit())
    digit_chars = sum(1 for ch in stripped if ch.isdigit())
    symbol_chars = sum(1 for ch in stripped if not ch.isalnum() and not is_kana_char(ch) and not contains_kanji(ch))
    jp_ratio = jp_chars / max(1, len(stripped))

    if source_used.startswith("easyocr") and confidence < OCR_ANOMALY_MIN_CONFIDENCE:
        if len(stripped) == 1 and not bubble_reaction_hint(stripped):
            return True, "easyocr_single_char_low_conf"
        if len(stripped) <= 2 and contains_kanji(stripped) and not inside_bubble:
            return True, "easyocr_short_kanji_low_conf"
        if len(stripped) <= 3 and jp_ratio < 0.7:
            return True, "easyocr_short_nonjp_low_conf"

    if ascii_letters + ascii_digits >= max(2, len(stripped) - 1) and confidence < 0.75:
        return True, "ascii_dominant_noise"
    if digit_chars + symbol_chars >= max(2, len(stripped) - 1) and jp_ratio < 0.35:
        return True, "digit_symbol_noise"
    if symbol_chars >= max(2, len(stripped) // 2) and confidence < 0.8:
        return True, "symbol_dominant_noise"
    if len(stripped) == 1 and contains_kanji(stripped) and source_used.startswith("easyocr") and confidence < 0.55:
        return True, "isolated_kanji_noise"
    if not inside_bubble and len(stripped) <= 3 and jp_ratio < 0.6 and confidence < 0.6:
        return True, "short_non_dialogue_noise"
    return False, ""

def refine_ocr_candidate_with_secondary_filter(original_text, source_used, rescue_reason,
                                               manga_text, easy_text, easy_conf,
                                               inside_bubble, region_type):
    if not ENABLE_SECONDARY_OCR_FILTER or not original_text:
        return original_text, source_used, rescue_reason, ""

    current_conf = 1.0 if source_used in {"manga_ocr", "bubble_text_rescue"} else float(easy_conf)
    anomalous, anomaly_reason = detect_ocr_anomaly(
        original_text,
        source_used,
        current_conf,
        inside_bubble,
        region_type,
    )
    if not anomalous:
        return original_text, source_used, rescue_reason, ""

    alternatives = []
    manga_compact = normalize_rule_text(manga_text)
    easy_compact = normalize_rule_text(easy_text)
    if manga_compact and manga_compact != normalize_rule_text(original_text):
        alternatives.append((manga_compact, "manga_ocr_secondary", rescue_reason, 1.0))
    if easy_compact and easy_compact != normalize_rule_text(original_text):
        alternatives.append((easy_compact, "easyocr_secondary", rescue_reason, float(easy_conf)))

    for alt_text, alt_source, alt_reason, alt_conf in alternatives:
        alt_bad, _ = detect_ocr_anomaly(alt_text, alt_source, alt_conf, inside_bubble, region_type)
        alt_ok, _ = is_valid_ocr_text(
            alt_text,
            inside_bubble=inside_bubble,
            region_type=region_type,
            source="manga_ocr" if alt_source.startswith("manga") else "easyocr",
            confidence=alt_conf,
        )
        if alt_ok and not alt_bad:
            return alt_text, alt_source, (alt_reason + "|secondary_swap").strip("|"), ""

    return "", source_used, rescue_reason, anomaly_reason

"""

def has_dialogue_marks(text):
    compact = normalize_rule_text(text)
    return any(mark in compact for mark in DIALOGUE_PUNCT_MARKS) or "..." in compact

def bubble_reaction_hint(text):
    stripped = strip_text_noise(text)
    return stripped in BUBBLE_REACTION_WORDS

def looks_like_sfx(text, inside_bubble=False):
    compact = normalize_rule_text(text)
    stripped = strip_text_noise(compact)
    traits = detect_sfx_traits(compact)
    if not inside_bubble:
        return bool(traits)
    if not stripped or not traits:
        return False
    if contains_kanji(stripped) or has_dialogue_marks(compact) or bubble_reaction_hint(stripped):
        return False
    if "sound_effect_dictionary_hit" in traits:
        return True
    return (
        len(stripped) <= BUBBLE_SHORT_TEXT_MAX_LEN
        and len(traits) >= 2
        and all(is_katakana(ch) for ch in stripped if is_kana_char(ch))
    )

def is_short_dialogue_candidate(text, inside_bubble):
    compact = normalize_rule_text(text)
    stripped = strip_text_noise(compact)
    if not stripped:
        return False
    if stripped in SHORT_DIALOGUE_WHITELIST or bubble_reaction_hint(stripped):
        return True
    if len(stripped) > 8:
        return False
    if inside_bubble and has_dialogue_marks(compact):
        return True
    if inside_bubble and compact.startswith(("\u3053\u3001", "\u3042\u3001", "\u3048\u3001")):
        return True
    if inside_bubble and any(mark in compact for mark in {"\u30fc", "\u301c", "\uff5e"}) and not looks_like_sfx(stripped, inside_bubble=True):
        return True
    if inside_bubble and stripped.endswith(("\u304b", "\u3088", "\u306d", "\u306e", "\u3060")) and not looks_like_sfx(stripped, inside_bubble=True):
        return True
    if inside_bubble and len(stripped) <= BUBBLE_SHORT_TEXT_MAX_LEN and contains_kanji(stripped):
        return True
    if inside_bubble and len(stripped) <= 2 and all(is_hiragana(ch) for ch in stripped):
        return not looks_like_sfx(stripped, inside_bubble=True)
    return False

def is_rescuable_short_dialogue(text, inside_bubble, region_type):
    compact = normalize_rule_text(text)
    stripped = strip_text_noise(compact)
    if is_short_dialogue_candidate(compact, inside_bubble):
        return True
    if region_type != "narration_box" or not stripped:
        return False
    if len(stripped) > max(BUBBLE_SHORT_TEXT_MAX_LEN + 2, 6):
        return False
    if looks_like_sfx(stripped, inside_bubble=True):
        return False
    if has_dialogue_marks(compact) or bubble_reaction_hint(stripped):
        return True
    if compact.startswith(("\u3053\u3001", "\u3042\u3001", "\u3048\u3001")):
        return True
    if any(mark in compact for mark in {"\u30fc", "\u301c", "\uff5e"}):
        return True
    if stripped.endswith(("\u304b", "\u3088", "\u306d", "\u306e", "\u3060")):
        return True
    if len(stripped) <= 2 and all(is_hiragana(ch) for ch in stripped):
        return True
    return False

def score_bubble_text_candidate(text, confidence=0.0, source="manga_ocr"):
    compact = normalize_rule_text(text)
    stripped = strip_text_noise(compact)
    if not stripped:
        return 0.0, []

    score = 0.0
    reasons = []
    if has_dialogue_marks(compact):
        score += 1.4
        reasons.append("dialogue_marks")
    if bubble_reaction_hint(stripped):
        score += 1.3
        reasons.append("reaction_hint")
    if is_short_dialogue_candidate(compact, True):
        score += 1.1
        reasons.append("short_dialogue_pattern")
    if contains_kanji(stripped):
        score += 0.7
        reasons.append("contains_kanji")
    if source == "manga_ocr":
        score += 0.35
        reasons.append("manga_ocr_bias")
    if confidence >= BUBBLE_RESCUE_MIN_CONFIDENCE:
        score += 0.6
        reasons.append("confidence_ok")
    elif source == "easyocr":
        score -= 0.4
        reasons.append("easyocr_low_conf")
    if len(stripped) <= BUBBLE_SHORT_TEXT_MAX_LEN:
        score += 0.25
        reasons.append("short_len")
    if looks_like_sfx(stripped, inside_bubble=True):
        score -= 2.2
        reasons.append("bubble_sfx_like")
    return score, reasons

def contains_heart_mark(text):
    compact = normalize_rule_text(text)
    return any(mark in compact for mark in HEART_MARKS)

def normalize_stylized_dialogue_key(text):
    compact = normalize_rule_text(text)
    return (
        compact
        .replace("\u2665", "\u2661")
        .replace("\u2764", "\u2661")
        .replace("\U0001f496", "\u2661")
    )

def is_stylized_dialogue(text, region_type="dialogue_bubble", inside_bubble=False):
    compact = normalize_stylized_dialogue_key(text)
    stripped = strip_text_noise(compact)
    if not stripped or region_type == "sfx_or_noise":
        return False
    if looks_like_sfx(stripped, inside_bubble=inside_bubble):
        return False
    if contains_heart_mark(compact):
        return True
    if "\u3082\u3048\u3082\u3048\u304d\u3085\u3093" in compact:
        return True
    if compact.startswith("\u7f8e\u5473\u3057\u304f\u306a\u30fc\u308c"):
        return True
    if len(stripped) >= 6 and any(mark in compact for mark in {"\u30fc", "\u301c", "\uff5e", "\u2026", "!", "?", "\uff01", "\uff1f"}):
        return True
    if region_type in {"dialogue_bubble", "narration_box"} and len(stripped) >= 5 and has_dialogue_marks(compact):
        return True
    return False

def build_stylized_dialogue_fallback(text):
    compact = normalize_stylized_dialogue_key(text)
    if compact in STYLIZED_DIALOGUE_FALLBACK_MAP:
        return STYLIZED_DIALOGUE_FALLBACK_MAP[compact]

    parts = []
    heart = "\u2661" if contains_heart_mark(compact) else ""
    if compact.startswith("\u7f8e\u5473\u3057\u304f\u306a\u30fc\u308c"):
        parts.append("\u53d8\u5f97\u7f8e\u5473\u5427" + heart)
    if "\u3082\u3048\u3082\u3048\u304d\u3085\u3093" in compact:
        parts.append("\u840c\u840c\u557e" + heart)
    return "".join(parts)

def choose_bubble_ocr_candidate(manga_text, easy_text, easy_conf, region_type):
    manga_ok, manga_reason = is_valid_ocr_text(
        manga_text,
        inside_bubble=True,
        region_type=region_type,
        source="manga_ocr",
        confidence=1.0
    )
    easy_ok, easy_reason = is_valid_ocr_text(
        easy_text,
        inside_bubble=True,
        region_type=region_type,
        source="easyocr",
        confidence=easy_conf
    )
    manga_score, manga_reasons = score_bubble_text_candidate(manga_text, 1.0, source="manga_ocr")
    easy_score, easy_reasons = score_bubble_text_candidate(easy_text, easy_conf, source="easyocr")
    manga_compact = normalize_rule_text(manga_text)
    easy_compact = normalize_rule_text(easy_text)

    if manga_ok and easy_ok:
        if (
            BUBBLE_OCR_FALLBACK_ALLOW_SHORT
            and len(easy_compact) >= len(manga_compact) + 1
            and easy_score >= manga_score + 0.45
            and not looks_like_sfx(easy_compact, inside_bubble=True)
        ):
            return easy_compact, "easyocr_bubble_override", "easy_candidate_more_complete", easy_score
        return manga_compact, "manga_ocr", "manga_valid", manga_score

    if manga_ok:
        return manga_compact, "manga_ocr", manga_reason, manga_score

    if easy_ok and (len(easy_compact) >= MANGA_OCR_FALLBACK_MIN_LEN or BUBBLE_OCR_FALLBACK_ALLOW_SHORT):
        return easy_compact, "easyocr_fallback", easy_reason, easy_score

    if ENABLE_BUBBLE_TEXT_RESCUE:
        candidates = [
            (manga_compact, manga_score, "bubble_text_rescue", "manga:" + ",".join(manga_reasons or [manga_reason])),
            (easy_compact, easy_score, "bubble_text_rescue", "easy:" + ",".join(easy_reasons or [easy_reason])),
        ]
        candidates = [item for item in candidates if item[0] and not looks_like_sfx(item[0], inside_bubble=True)]
        if candidates:
            best_text, best_score, choice, reason = max(candidates, key=lambda item: item[1])
            if best_score >= 1.45:
                return best_text, choice, reason, best_score

    return "", "", f"{manga_reason}/{easy_reason}", max(manga_score, easy_score)

def should_skip_short_text(text, inside_bubble, region_type, easy_conf):
    compact = normalize_rule_text(text)
    stripped = compact.strip("…...。、「」『』【】（）()！？!?，,、・ー-～〜")

    if not stripped:
        return True
    if is_punctuation_only(compact):
        return True
    if region_type == "sfx_or_noise":
        return True
    if is_rescuable_short_dialogue(compact, inside_bubble, region_type):
        return False
    if inside_bubble and is_bubble_shape_rescuable_text(compact):
        return False
    if not inside_bubble and looks_like_sfx(stripped):
        return True
    if not inside_bubble and len(stripped) <= SFX_SKIP_MAX_LEN and easy_conf < MIN_OCR_CONFIDENCE and looks_like_sfx(stripped):
        return True
    if not inside_bubble and len(stripped) <= 2 and stripped not in SHORT_DIALOGUE_WHITELIST and not contains_kanji(stripped):
        return True
    if not inside_bubble and len(stripped) <= 3 and is_kana_only(stripped):
        return True
    return False

def is_valid_ocr_text(text, inside_bubble, region_type, source="manga_ocr", confidence=1.0):
    compact = normalize_rule_text(text)
    stripped = compact.strip("…...。、「」『』【】（）()！？!?，,、・ー-～〜")
    if not stripped:
        return False, "empty_text"
    if is_punctuation_only(compact):
        return False, "punctuation_only"
    if region_type == "sfx_or_noise":
        return False, "classified_as_sfx"
    if source == "easyocr" and confidence < MIN_OCR_CONFIDENCE and len(stripped) <= SFX_SKIP_MAX_LEN:
        return False, "easyocr_low_conf_short"
    if source == "easyocr" and len(stripped) == 1 and not inside_bubble and not contains_kanji(stripped):
        return False, "easyocr_single_char_noise"
    if not inside_bubble and looks_like_sfx(stripped):
        return False, "non_bubble_sfx_shape"
    if len(stripped) == 1 and not inside_bubble and stripped not in SHORT_DIALOGUE_WHITELIST and not contains_kanji(stripped):
        return False, "single_char_non_dialogue"
    if len(stripped) <= SFX_SKIP_MAX_LEN and is_kana_only(stripped) and not is_short_dialogue_candidate(compact, inside_bubble):
        if not inside_bubble or source == "easyocr":
            return False, "short_kana_fragment"
    return True, "valid"

def bubble_reaction_hint(text):
    stripped = strip_text_noise(text)
    return stripped in BUBBLE_REACTION_WORDS

def looks_like_sfx(text, inside_bubble=False):
    compact = normalize_rule_text(text)
    stripped = strip_text_noise(compact)
    traits = detect_sfx_traits(compact)
    if not inside_bubble:
        return bool(traits)
    if not stripped or not traits:
        return False
    if contains_kanji(stripped) or has_dialogue_marks(compact) or bubble_reaction_hint(stripped):
        return False
    if "sound_effect_dictionary_hit" in traits:
        return True
    return (
        len(stripped) <= BUBBLE_SHORT_TEXT_MAX_LEN
        and len(traits) >= 2
        and all(is_katakana(ch) for ch in stripped if is_kana_char(ch))
    )

def is_short_dialogue_candidate(text, inside_bubble):
    compact = normalize_rule_text(text)
    stripped = strip_text_noise(compact)
    if not stripped:
        return False
    if stripped in SHORT_DIALOGUE_WHITELIST or bubble_reaction_hint(stripped):
        return True
    if len(stripped) > 8:
        return False
    if inside_bubble and has_dialogue_marks(compact):
        return True
    if inside_bubble and compact.startswith(("\u3053\u3001", "\u3042\u3001", "\u3048\u3001")):
        return True
    if inside_bubble and any(mark in compact for mark in {"\u30fc", "\u301c", "\uff5e"}) and not looks_like_sfx(stripped, inside_bubble=True):
        return True
    if inside_bubble and stripped.endswith(("\u304b", "\u3088", "\u306d", "\u306e", "\u3060")) and not looks_like_sfx(stripped, inside_bubble=True):
        return True
    if inside_bubble and len(stripped) <= BUBBLE_SHORT_TEXT_MAX_LEN and contains_kanji(stripped):
        return True
    if inside_bubble and len(stripped) <= 2 and all(is_hiragana(ch) for ch in stripped):
        return not looks_like_sfx(stripped, inside_bubble=True)
    return False

def score_bubble_text_candidate(text, confidence=0.0, source="manga_ocr"):
    compact = normalize_rule_text(text)
    stripped = strip_text_noise(compact)
    if not stripped:
        return 0.0, []

    score = 0.0
    reasons = []
    if has_dialogue_marks(compact):
        score += 1.4
        reasons.append("dialogue_marks")
    if bubble_reaction_hint(stripped):
        score += 1.3
        reasons.append("reaction_hint")
    if is_short_dialogue_candidate(compact, True):
        score += 1.1
        reasons.append("short_dialogue_pattern")
    if contains_kanji(stripped):
        score += 0.7
        reasons.append("contains_kanji")
    if source == "manga_ocr":
        score += 0.35
        reasons.append("manga_ocr_bias")
    if confidence >= BUBBLE_RESCUE_MIN_CONFIDENCE:
        score += 0.6
        reasons.append("confidence_ok")
    elif source == "easyocr":
        score -= 0.4
        reasons.append("easyocr_low_conf")
    if len(stripped) <= BUBBLE_SHORT_TEXT_MAX_LEN:
        score += 0.25
        reasons.append("short_len")
    if looks_like_sfx(stripped, inside_bubble=True):
        score -= 2.2
        reasons.append("bubble_sfx_like")
    return score, reasons

def is_bubble_shape_rescuable_text(text):
    compact = normalize_rule_text(text)
    stripped = strip_text_noise(compact)
    if len(stripped) < 2:
        return False
    if looks_like_sfx(stripped, inside_bubble=True):
        return False
    jp_chars = count_japanese_chars(stripped)
    return (
        has_dialogue_marks(compact)
        or contains_kanji(stripped)
        or jp_chars >= 2
        or (len(stripped) >= 3 and not stripped.isascii())
    )

def is_valid_ocr_text(text, inside_bubble, region_type, source="manga_ocr", confidence=1.0, bubble_shape_rescue=False):
    compact = normalize_rule_text(text)
    stripped = strip_text_noise(compact)
    if not stripped:
        return False, "empty_text"
    if is_punctuation_only(compact):
        return False, "punctuation_only"
    if region_type == "sfx_or_noise":
        return False, "classified_as_sfx"
    if inside_bubble and looks_like_sfx(stripped, inside_bubble=True):
        return False, "bubble_sfx_shape"
    if len(stripped) <= BUBBLE_SHORT_TEXT_MAX_LEN and is_rescuable_short_dialogue(compact, inside_bubble, region_type):
        if is_short_dialogue_candidate(compact, True) or region_type == "narration_box":
            return True, "bubble_short_dialogue"
    if inside_bubble and len(stripped) <= BUBBLE_SHORT_TEXT_MAX_LEN:
        if source == "manga_ocr" and ENABLE_BUBBLE_TEXT_RESCUE:
            return True, "bubble_manga_short_rescue"
        if (
            source == "easyocr"
            and ENABLE_BUBBLE_TEXT_RESCUE
            and confidence >= BUBBLE_RESCUE_MIN_CONFIDENCE
            and not looks_like_sfx(stripped, inside_bubble=True)
        ):
            return True, "bubble_easyocr_short_rescue"
    if source == "easyocr" and confidence < MIN_OCR_CONFIDENCE and len(stripped) <= SFX_SKIP_MAX_LEN:
        return False, "easyocr_low_conf_short"
    if source == "easyocr" and len(stripped) == 1 and not inside_bubble and not contains_kanji(stripped):
        return False, "easyocr_single_char_noise"
    if not inside_bubble and looks_like_sfx(stripped):
        return False, "non_bubble_sfx_shape"
    if len(stripped) == 1 and not inside_bubble and stripped not in SHORT_DIALOGUE_WHITELIST and not contains_kanji(stripped):
        return False, "single_char_non_dialogue"
    if len(stripped) <= SFX_SKIP_MAX_LEN and is_kana_only(stripped) and not is_rescuable_short_dialogue(compact, inside_bubble, region_type):
        if not inside_bubble or source == "easyocr":
            if bubble_shape_rescue and inside_bubble and is_bubble_shape_rescuable_text(compact):
                return True, "bubble_shape_rescue"
            return False, "short_kana_fragment"
    return True, "valid"

def choose_bubble_ocr_candidate(manga_text, easy_text, easy_conf, region_type, bubble_shape_rescue=False):
    manga_ok, manga_reason = is_valid_ocr_text(
        manga_text,
        inside_bubble=True,
        region_type=region_type,
        source="manga_ocr",
        confidence=1.0,
        bubble_shape_rescue=bubble_shape_rescue,
    )
    easy_ok, easy_reason = is_valid_ocr_text(
        easy_text,
        inside_bubble=True,
        region_type=region_type,
        source="easyocr",
        confidence=easy_conf,
        bubble_shape_rescue=bubble_shape_rescue,
    )
    manga_score, manga_reasons = score_bubble_text_candidate(manga_text, 1.0, source="manga_ocr")
    easy_score, easy_reasons = score_bubble_text_candidate(easy_text, easy_conf, source="easyocr")
    manga_compact = normalize_rule_text(manga_text)
    easy_compact = normalize_rule_text(easy_text)

    if manga_ok and easy_ok:
        if (
            BUBBLE_OCR_FALLBACK_ALLOW_SHORT
            and len(easy_compact) >= len(manga_compact) + 1
            and easy_score >= manga_score + 0.45
            and not looks_like_sfx(easy_compact, inside_bubble=True)
        ):
            return easy_compact, "easyocr_bubble_override", "easy_candidate_more_complete", easy_score
        return manga_compact, "manga_ocr", "manga_valid", manga_score

    if manga_ok:
        return manga_compact, "manga_ocr", manga_reason, manga_score

    if easy_ok and (len(easy_compact) >= MANGA_OCR_FALLBACK_MIN_LEN or BUBBLE_OCR_FALLBACK_ALLOW_SHORT):
        return easy_compact, "easyocr_fallback", easy_reason, easy_score

    if ENABLE_BUBBLE_TEXT_RESCUE:
        candidates = [
            (manga_compact, manga_score, "bubble_text_rescue", "manga:" + ",".join(manga_reasons or [manga_reason])),
            (easy_compact, easy_score, "bubble_text_rescue", "easy:" + ",".join(easy_reasons or [easy_reason])),
        ]
        candidates = [item for item in candidates if item[0] and not looks_like_sfx(item[0], inside_bubble=True)]
        if candidates:
            best_text, best_score, choice, reason = max(candidates, key=lambda item: item[1])
            if best_score >= 1.45:
                return best_text, choice, reason, best_score

    if bubble_shape_rescue:
        candidates = []
        for text, score, choice, reason in [
            (manga_compact, manga_score, "bubble_shape_rescue", "manga_shape:" + manga_reason),
            (easy_compact, easy_score, "bubble_shape_rescue", "easy_shape:" + easy_reason),
        ]:
            if text and is_bubble_shape_rescuable_text(text):
                candidates.append((text, score + 0.35, choice, reason))
        if candidates:
            best_text, best_score, choice, reason = max(candidates, key=lambda item: item[1])
            if best_score >= 0.85:
                return best_text, choice, reason, best_score

    return "", "", f"{manga_reason}/{easy_reason}", max(manga_score, easy_score)

def count_japanese_chars(text):
    return sum(1 for ch in normalize_rule_text(text) if is_kana_char(ch) or contains_kanji(ch))

def detect_ocr_anomaly(text, source_used, confidence, inside_bubble, region_type):
    compact = normalize_rule_text(text)
    stripped = strip_text_noise(compact)
    if not stripped:
        return True, "empty_text"
    if is_rescuable_short_dialogue(compact, inside_bubble, region_type):
        return False, ""
    if is_short_dialogue_candidate(compact, inside_bubble):
        return False, ""
    if looks_like_japanese_sentence(compact):
        return False, ""

    jp_chars = count_japanese_chars(stripped)
    ascii_letters = sum(1 for ch in stripped if ch.isascii() and ch.isalpha())
    ascii_digits = sum(1 for ch in stripped if ch.isascii() and ch.isdigit())
    symbol_chars = sum(1 for ch in stripped if not ch.isalnum() and not is_kana_char(ch) and not contains_kanji(ch))
    jp_ratio = jp_chars / max(1, len(stripped))

    if source_used.startswith("easyocr") and confidence < OCR_ANOMALY_MIN_CONFIDENCE:
        if len(stripped) == 1 and not bubble_reaction_hint(stripped):
            return True, "easyocr_single_char_low_conf"
        if len(stripped) <= 2 and contains_kanji(stripped) and not inside_bubble:
            return True, "easyocr_short_kanji_low_conf"
        if len(stripped) <= 3 and jp_ratio < 0.7:
            return True, "easyocr_short_nonjp_low_conf"

    if ascii_letters + ascii_digits >= max(2, len(stripped) - 1) and confidence < 0.75:
        return True, "ascii_dominant_noise"
    if symbol_chars >= max(2, len(stripped) // 2) and confidence < 0.8:
        return True, "symbol_dominant_noise"
    if len(stripped) == 1 and contains_kanji(stripped) and source_used.startswith("easyocr") and confidence < 0.55:
        return True, "isolated_kanji_noise"
    if not inside_bubble and len(stripped) <= 3 and jp_ratio < 0.6 and confidence < 0.6:
        return True, "short_non_dialogue_noise"
    return False, ""

def refine_ocr_candidate_with_secondary_filter(original_text, source_used, rescue_reason,
                                               manga_text, easy_text, easy_conf,
                                               inside_bubble, region_type):
    if not ENABLE_SECONDARY_OCR_FILTER or not original_text:
        return original_text, source_used, rescue_reason, ""

    current_conf = 1.0 if source_used in {"manga_ocr", "bubble_text_rescue"} else float(easy_conf)
    anomalous, anomaly_reason = detect_ocr_anomaly(
        original_text,
        source_used,
        current_conf,
        inside_bubble,
        region_type,
    )
    if not anomalous:
        return original_text, source_used, rescue_reason, ""

    alternatives = []
    manga_compact = normalize_rule_text(manga_text)
    easy_compact = normalize_rule_text(easy_text)
    if manga_compact and manga_compact != normalize_rule_text(original_text):
        alternatives.append((manga_compact, "manga_ocr_secondary", rescue_reason, 1.0))
    if easy_compact and easy_compact != normalize_rule_text(original_text):
        alternatives.append((easy_compact, "easyocr_secondary", rescue_reason, float(easy_conf)))

    for alt_text, alt_source, alt_reason, alt_conf in alternatives:
        alt_bad, _ = detect_ocr_anomaly(alt_text, alt_source, alt_conf, inside_bubble, region_type)
        alt_ok, _ = is_valid_ocr_text(
            alt_text,
            inside_bubble=inside_bubble,
            region_type=region_type,
            source="manga_ocr" if alt_source.startswith("manga") else "easyocr",
            confidence=alt_conf,
        )
        if alt_ok and not alt_bad:
            return alt_text, alt_source, (alt_reason + "|secondary_swap").strip("|"), ""

    return "", source_used, rescue_reason, anomaly_reason

def normalize_cn_dialogue_tone(text):
    if not text:
        return ""
    normalized = text
    normalized = re.sub(r"^(我我|啊啊|嗯嗯)(?=[，。！？])", lambda m: m.group(0)[0], normalized)
    normalized = normalized.replace("……。", "……")
    normalized = normalized.replace("！！", "！")
    normalized = normalized.replace("？？", "？")
    normalized = re.sub(r"([，。！？])\1+", r"\1", normalized)
    normalized = normalized.replace("原，原来", "原来")
    normalized = normalized.replace("喂，喂，", "喂，")
    return normalized.strip()

def clean_translation_output(translated, original_text):
    if not translated:
        return ""

    cleaned = translated.strip()
    cleaned = cleaned.strip("「」『』\"'")
    cleaned = re.sub(r"^(翻译[:：]|译文[:：]|中文[:：])", "", cleaned)
    cleaned = cleaned.replace("\r", "").replace("\n\n", "\n").strip()
    cleaned = re.sub(r"[ \t]+", "", cleaned)
    cleaned = cleaned.replace("...", "……")
    cleaned = re.sub(r"…{3,}", "……", cleaned)
    cleaned = cleaned.translate(JP_PUNCT_TRANSLATION)

    if any(pattern in cleaned for pattern in TRANSLATE_SKIP_PATTERNS):
        return ""
    if cleaned == original_text and looks_like_sfx(original_text):
        return ""
    return normalize_cn_dialogue_tone(cleaned)
# ================== 翻译函数（根据开关决定行为） ==================
def translate_text(text, target_lang="zh", region_type="dialogue_bubble"):
    """
    根据环境变量 USE_DEEPSEEK_API 决定使用真实翻译或模拟翻译。
    若为 'true' 且设置了 DEEPSEEK_API_KEY，则调用 DeepSeek API；
    否则返回  原文。
    """
    use_api = os.environ.get("USE_DEEPSEEK_API", "false").lower() == "true"
    if not use_api:
        return f"{text}"

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("USE_DEEPSEEK_API=true 但未设置 DEEPSEEK_API_KEY，使用模拟翻译")
        return f"{text}"

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": (
                f"将以下日文漫画文本翻译成{target_lang}。"
                "只翻译正常对白或旁白框。"
                "如果是拟声词、动作字、特效字、语义不明碎片，请只输出“跳过”。"
                "保留口语感，不要解释，不要加引号，不要过度意译人名。"
                "省略号和语气词按中文漫画习惯处理。"
                f"当前文本类型：{region_type}。"
            )},
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 200000
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return clean_translation_output(result["choices"][0]["message"]["content"].strip(), text)
    except Exception as e:
        print(f"翻译失败: {e}")
        return text   # 失败时保留原文，避免中断流程


# ================== 嵌字函数（保持不变） ==================
def normalize_cn_dialogue_tone(text):
    if not text:
        return ""
    normalized = text
    normalized = re.sub(r"^(啊啊|嗯嗯|哈啊)(?=[，。！？])", lambda m: m.group(0)[0], normalized)
    normalized = normalized.replace("……。", "……")
    normalized = normalized.replace("！！", "！")
    normalized = normalized.replace("？？", "？")
    normalized = re.sub(r"([，。！？])\1+", r"\1", normalized)
    normalized = normalized.replace("原来，原来", "原来")
    normalized = normalized.replace("啊，啊，", "啊，")
    return normalized.strip()

def clean_translation_output(translated, original_text, region_type="dialogue_bubble"):
    if not translated:
        return ""

    cleaned = translated.strip()
    cleaned = cleaned.strip("「」『』\"'")
    cleaned = re.sub(r"^(翻译[:：]|译文[:：]|中文[:：])", "", cleaned)
    cleaned = cleaned.replace("\r", "").replace("\n\n", "\n").strip()
    cleaned = re.sub(r"[ \t]+", "", cleaned)
    cleaned = cleaned.replace("...", "\u2026\u2026")
    cleaned = re.sub(r"\u2026{3,}", "\u2026\u2026", cleaned)
    cleaned = cleaned.translate(JP_PUNCT_TRANSLATION)

    if any(pattern in cleaned for pattern in TRANSLATE_SKIP_PATTERNS):
        if TRANSLATION_RETRY_FOR_STYLIZED_DIALOGUE and is_stylized_dialogue(
            original_text,
            region_type=region_type,
            inside_bubble=(region_type == "dialogue_bubble"),
        ):
            return normalize_cn_dialogue_tone(build_stylized_dialogue_fallback(original_text))
        return ""
    if cleaned == original_text and looks_like_sfx(original_text, inside_bubble=(region_type == "dialogue_bubble")):
        return ""
    if (
        TRANSLATION_RETRY_FOR_STYLIZED_DIALOGUE
        and not cleaned
        and is_stylized_dialogue(original_text, region_type=region_type, inside_bubble=(region_type == "dialogue_bubble"))
    ):
        return normalize_cn_dialogue_tone(build_stylized_dialogue_fallback(original_text))
    return normalize_cn_dialogue_tone(cleaned)

def translate_text(text, target_lang="zh", region_type="dialogue_bubble"):
    use_api = os.environ.get("USE_DEEPSEEK_API", "false").lower() == "true"
    if not use_api:
        return f"{text}"

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("USE_DEEPSEEK_API=true but DEEPSEEK_API_KEY is not set, using source text.")
        return f"{text}"

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Translate the following Japanese manga text into {target_lang}. "
                    "Treat normal dialogue bubbles, colored emotion bubbles, narration boxes, cute chants, exclamations, and heart-marked lines as normal dialogue. "
                    "Only output SKIP when the text is clearly sound effects, action lettering, or meaningless OCR fragments. "
                    "Keep the result conversational, concise, and natural for Chinese comics. "
                    "Do not explain. Do not add quotation marks. Preserve playful tone, ellipses, exclamation marks, and heart symbols when they are part of the dialogue. "
                    f"Current region type: {region_type}."
                ),
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

def draw_text_vertical(draw, box, text, font_paths,
                       max_font_size=60, min_font_size=8,
                       width_fill_ratio=0.9, margin=5,
                       col_spacing=3, row_spacing=1):
    """
    在四边形框内以竖排方式绘制文本（从上到下，从右向左）。
    - 自动换列：当一列的文字高度超出框高时，自动移到左边新列。
    - 支持 \n 作为强制换列符。
    - 字号自动适应：通过二分查找找到能使所有列总宽度最接近框宽的字号。
    - width_fill_ratio: 总列宽希望占框宽的比例（0~1），默认0.9。
    - col_spacing: 列间距（像素）
    - row_spacing: 字符间垂直额外间距（像素），默认1
    """
    # 计算可用区域
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_width = x_max - x_min - 2 * margin
    box_height = y_max - y_min - 2 * margin
    if box_width <= 0 or box_height <= 0:
        return False

    # 根据 \n 将文本拆成段落（每个段落会强制另起一列）
    paragraphs = text.split('\n')

    # 内部函数：给定字号，计算所有字符排版所需的总列宽
    def get_total_width_for_size(size):
        # 加载字体
        font = None
        for fp in font_paths:
            try:
                font = ImageFont.truetype(fp, size)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()

        # 测量单个全角字符的宽度和高度（以“测”字为参考）
        bbox = draw.textbbox((0, 0), '测', font=font)
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]

        max_chars_per_col = max(1, int((box_height + row_spacing) // (char_h + row_spacing)))

        columns = 0
        for para in paragraphs:
            if not para:
                continue
            # 将当前段落文字按字符拆开
            chars = list(para)
            # 如果段落为空，跳过（columns 不增加额外列，除非有空格需求，这里忽略）
            idx = 0
            while idx < len(chars):
                # 本列还可以容纳的字符数
                remain = len(chars) - idx
                chars_in_col = min(max_chars_per_col, remain)
                idx += chars_in_col
                columns += 1
            # 每个段落结束后相当于一个强制换列（已经在段落之间隐含，因为下一段落从新列开始）
        # 如果没有字符，返回0
        if columns == 0:
            return 0
        total_width = columns * char_w + (columns - 1) * col_spacing
        return total_width

    # ---- 二分搜索最佳字号 ----
    # 先看最大字号能否放入框宽
    if get_total_width_for_size(max_font_size) > box_width:
        # 放不下，缩小字号
        lo, hi = min_font_size, max_font_size
        best_size = min_font_size
        while lo <= hi:
            mid = (lo + hi) // 2
            if get_total_width_for_size(mid) <= box_width:
                best_size = mid
                lo = mid + 1
            else:
                hi = mid - 1
        final_size = best_size
    else:
        # 放得下，尝试放大字号，直到总宽度接近 width_fill_ratio * box_width
        lo, hi = max_font_size, max_font_size * 2
        best_size = max_font_size
        while lo <= hi:
            mid = (lo + hi) // 2
            total_w = get_total_width_for_size(mid)
            if total_w <= box_width * width_fill_ratio:
                best_size = mid
                lo = mid + 1
            else:
                hi = mid - 1
        final_size = best_size

    # ---- 使用最终字号排版并绘制 ----
    # 加载最终字体
    font = None
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, final_size)
            break
        except:
            continue
    if font is None:
        font = ImageFont.load_default()

    # 重新测量最终字号的尺寸
    bbox = draw.textbbox((0, 0), '测', font=font)
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    max_chars_per_col = max(1, int((box_height + row_spacing) // (char_h + row_spacing)))

    # 将文本转化为字符列表，并记录列信息
    columns_chars = []   # 每列是一个字符列表
    for para in paragraphs:
        chars = list(para)
        idx = 0
        while idx < len(chars):
            remain = len(chars) - idx
            chars_in_col = min(max_chars_per_col, remain)
            columns_chars.append(chars[idx:idx + chars_in_col])
            idx += chars_in_col
        # 段落结束不自动增加空列，除非末尾有 \n，但 paragraph 分割已经处理了，
        # 如果两个段落之间需要强制换列，已经通过分段实现，这里不额外加

    if not columns_chars:
        return False

    total_cols = len(columns_chars)
    total_width = total_cols * char_w + (total_cols - 1) * col_spacing

    # 起始 x 坐标：从右向左排，最右列的 x 位置
    start_x_right = x_max - margin - char_w
    # 整个列块的水平偏移，使其在框内水平居中
    x_center_offset = (box_width - total_width) // 2
    start_x = start_x_right - x_center_offset

    # 逐列绘制
    for col_idx, chars in enumerate(columns_chars):
        # 计算本列的 x 坐标（列索引从右向左：col_idx=0 最右）
        col_x = start_x - col_idx * (char_w + col_spacing)
        # 计算本列文字的垂直起始 y 坐标（居中）
        col_text_height = len(chars) * (char_h + row_spacing) - row_spacing
        start_y = y_min + margin + (box_height - col_text_height) // 2

        # 绘制本列每个字符
        for row_idx, ch in enumerate(chars):
            char_y = start_y + row_idx * (char_h + row_spacing)
            # 字符水平居中在其单元格内（char_w宽度）
            # 直接使用 draw.text，传入字符左上角
            draw.text((col_x, char_y), ch, fill=(0, 0, 0), font=font)

    return True

# ================== 框合并函数（保持不变） ==================
def merge_same_line_boxes(boxes, x_gap=20, y_overlap_ratio=0.7):
    """仅合并水平方向紧邻、且垂直高度重叠高的框（同一行内的碎片）"""
    if len(boxes) <= 1:
        return boxes
    rects = []
    for box in boxes:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        rects.append([min(xs), min(ys), max(xs), max(ys)])
    merged, used = [], [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        group, used[i] = [i], True
        while True:
            changed = False
            for j in range(len(boxes)):
                if used[j]: continue
                for idx in group:
                    x1,y1,x2,y2 = rects[idx]
                    x3,y3,x4,y4 = rects[j]
                    overlap_y = max(0, min(y2,y4)-max(y1,y3))
                    h1,h2 = y2-y1, y4-y3
                    min_h = min(h1,h2)
                    if min_h == 0: continue
                    if overlap_y/min_h < y_overlap_ratio: continue
                    gap = max(x1,x3)-min(x2,x4) if (x2<x3 or x4<x1) else 0
                    if gap < x_gap:
                        group.append(j); used[j] = True
                        changed = True; break
                if changed: break
            if not changed: break
        if len(group) == 1:
            merged.append(boxes[group[0]])
        else:
            all_pts = []
            for idx in group: all_pts.extend(boxes[idx])
            xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
            merged.append([[min(xs),min(ys)],[max(xs),min(ys)],
                           [max(xs),max(ys)],[min(xs),max(ys)]])
    return merged

def merge_vertical_boxes(boxes, y_gap=25, x_overlap_ratio=0.6):
    """合并竖排文本中上下相邻的碎片框。"""
    if len(boxes) <= 1:
        return boxes

    rects = [box_to_rect(box) for box in boxes]
    merged, used = [], [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        group = [i]
        used[i] = True

        while True:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                for idx in group:
                    x1, y1, x2, y2 = rects[idx]
                    x3, y3, x4, y4 = rects[j]
                    overlap_x = max(0, min(x2, x4) - max(x1, x3))
                    w1, w2 = x2 - x1, x4 - x3
                    min_w = min(w1, w2)
                    if min_w <= 0:
                        continue
                    if overlap_x / min_w < x_overlap_ratio:
                        continue
                    gap = max(y1, y3) - min(y2, y4) if (y2 < y3 or y4 < y1) else 0
                    if gap < y_gap:
                        group.append(j)
                        used[j] = True
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break

        if len(group) == 1:
            merged.append(boxes[group[0]])
        else:
            all_pts = []
            for idx in group:
                all_pts.extend(boxes[idx])
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            merged.append([[min(xs), min(ys)], [max(xs), min(ys)],
                           [max(xs), max(ys)], [min(xs), max(ys)]])
    return merged

def merge_region_group(group, order_key):
    ordered = sorted(group, key=order_key)
    all_pts = []
    texts = []
    confidences = []
    parts = []
    split_parent_rects = []
    projection_axes = []
    split_confidences = []
    projection_valley_counts = []
    split_sources = []
    blockers = []
    attempted = False
    applied = False
    for region in ordered:
        all_pts.extend(region["box"])
        if region.get("easy_text"):
            texts.append(region["easy_text"])
        confidences.append(float(region.get("easy_conf", 0.0)))
        if region.get("split_parent_rect"):
            split_parent_rects.append(tuple(region["split_parent_rect"]))
        if region.get("projection_axis"):
            projection_axes.append(region["projection_axis"])
        if region.get("split_source"):
            split_sources.append(region["split_source"])
        split_confidences.append(float(region.get("split_confidence", 0.0)))
        projection_valley_counts.append(int(region.get("projection_valley_count", 0)))
        if region.get("single_large_split_blocker"):
            blockers.append(region["single_large_split_blocker"])
        attempted = attempted or bool(region.get("single_large_split_attempted", False))
        applied = applied or bool(region.get("single_large_split_applied", False))
        region_parts = region.get("parts")
        if region_parts:
            parts.extend(region_parts)
        else:
            parts.append({
                "box": [list(pt) for pt in region["box"]],
                "easy_text": region.get("easy_text", ""),
                "easy_conf": float(region.get("easy_conf", 0.0)),
                "bubble_index": region.get("bubble_index", -1),
            })

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    bubble_ids = [region.get("bubble_index", -1) for region in ordered if region.get("bubble_index", -1) >= 0]
    merged = {
        "box": [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]],
        "easy_text": "".join(texts),
        "easy_conf": sum(confidences) / max(1, len(confidences)),
        "bubble_index": bubble_ids[0] if bubble_ids and len(set(bubble_ids)) == 1 else -1,
        "parts": parts,
        "source_count": len(parts),
    }
    unique_parent_rects = list(dict.fromkeys(split_parent_rects))
    unique_axes = list(dict.fromkeys(projection_axes))
    unique_sources = list(dict.fromkeys(split_sources))
    unique_blockers = list(dict.fromkeys(blockers))
    if attempted or applied or unique_parent_rects or unique_axes or unique_sources:
        merged.update({
            "split_from_large_region": applied,
            "split_source": unique_sources[0] if len(unique_sources) == 1 else "",
            "split_parent_rect": list(unique_parent_rects[0]) if len(unique_parent_rects) == 1 else None,
            "projection_axis": unique_axes[0] if len(unique_axes) == 1 else "",
            "split_confidence": max(split_confidences) if split_confidences else 0.0,
            "projection_valley_count": max(projection_valley_counts) if projection_valley_counts else 0,
            "single_large_split_attempted": attempted,
            "single_large_split_applied": applied,
            "single_large_split_blocker": unique_blockers[0] if (attempted and not applied and len(unique_blockers) == 1) else "",
        })
    return merged

def regions_share_projection_parent(region_a, region_b, axis):
    if region_a.get("split_source") != "projection" or region_b.get("split_source") != "projection":
        return False
    if region_a.get("projection_axis") != axis or region_b.get("projection_axis") != axis:
        return False
    parent_a = region_a.get("split_parent_rect")
    parent_b = region_b.get("split_parent_rect")
    return bool(parent_a and parent_b and tuple(parent_a) == tuple(parent_b))

def with_single_large_split_meta(region, attempted=False, applied=False, blocker="", split_source="",
                                 split_parent_rect=None, projection_axis="", split_confidence=0.0,
                                 projection_valley_count=0):
    region["single_large_split_attempted"] = bool(attempted)
    region["single_large_split_applied"] = bool(applied)
    region["single_large_split_blocker"] = blocker or ""
    region["split_from_large_region"] = bool(applied)
    region["split_source"] = split_source or ""
    region["split_parent_rect"] = list(split_parent_rect) if split_parent_rect else None
    region["projection_axis"] = projection_axis or ""
    region["split_confidence"] = round(float(split_confidence), 4)
    region["projection_valley_count"] = int(projection_valley_count)
    return region

def build_projection_ink_mask(crop_rgb):
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_mask = cv2.bitwise_or(adaptive, otsu)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    cleaned = np.zeros_like(ink_mask)
    crop_h, crop_w = crop_rgb.shape[:2]
    crop_area = max(1, crop_h * crop_w)
    min_component_area = max(8, int(crop_rgb.shape[0] * crop_rgb.shape[1] * 0.00035))
    max_component_area = max(1400, int(crop_area * 0.055))
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        top = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        touches_border = (
            left <= 1 or top <= 1 or
            left + width >= crop_w - 1 or
            top + height >= crop_h - 1
        )
        oversized = (
            area > max_component_area or
            (width > int(crop_w * 0.55) and height > int(crop_h * 0.22)) or
            (height > int(crop_h * 0.7) and width > int(crop_w * 0.18))
        )
        if area >= min_component_area and not (oversized and touches_border):
            cleaned[labels == label_idx] = 255
    return cleaned

def choose_projection_axis_for_region(rect):
    width = max(1, rect[2] - rect[0])
    height = max(1, rect[3] - rect[1])
    if width >= height * 1.4:
        return "x", ""
    if height >= width * 1.4:
        return "y", ""
    return "", "projection_valley_not_clear"

def compute_projection_valleys(ink_mask, axis, min_width, max_cuts):
    profile = np.count_nonzero(ink_mask > 0, axis=0 if axis == "x" else 1).astype(np.float32)
    axis_len = int(profile.shape[0])
    if axis_len < max(32, min_width * 3):
        return [], 0.0

    kernel_size = max(5, (axis_len // 32) | 1)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smoothed = np.convolve(profile, kernel, mode="same")
    peak = float(smoothed.max()) if smoothed.size else 0.0
    mean_val = float(smoothed.mean()) if smoothed.size else 0.0
    if peak <= 1.0:
        return [], 0.0

    valley_threshold = min(peak * PROJECTION_VALLEY_RATIO, max(1.0, mean_val * 0.55))
    min_segment_span = max(42, axis_len // 7)
    valleys = []
    start = None
    for idx, value in enumerate(smoothed):
        if value <= valley_threshold:
            if start is None:
                start = idx
        elif start is not None:
            valleys.append((start, idx))
            start = None
    if start is not None:
        valleys.append((start, axis_len))

    candidates = []
    for valley_start, valley_end in valleys:
        width = valley_end - valley_start
        if width < min_width:
            continue
        width_ratio = width / max(1.0, float(axis_len))
        if width_ratio < PROJECTION_VALLEY_MIN_RATIO:
            continue
        center = int(round((valley_start + valley_end) * 0.5))
        if center < min_segment_span or axis_len - center < min_segment_span:
            continue
        valley_slice = smoothed[valley_start:valley_end]
        left_slice = smoothed[max(0, valley_start - min_segment_span):valley_start]
        right_slice = smoothed[valley_end:min(axis_len, valley_end + min_segment_span)]
        if len(left_slice) == 0 or len(right_slice) == 0:
            continue
        left_peak = float(left_slice.max())
        right_peak = float(right_slice.max())
        shoulder = min(left_peak, right_peak)
        valley_mean = float(valley_slice.mean()) if len(valley_slice) else peak
        if shoulder <= 0.0:
            continue
        depth_ratio = 1.0 - (valley_mean / shoulder)
        width_ratio = min(1.0, width / max(float(min_width), 1.0))
        confidence = depth_ratio * 0.8 + width_ratio * 0.2
        if depth_ratio < 0.34 or confidence < 0.42:
            continue
        candidates.append({
            "start": valley_start,
            "end": valley_end,
            "width_ratio": width_ratio,
            "center": center,
            "confidence": confidence,
        })

    candidates.sort(key=lambda item: (-item["confidence"], item["center"]))
    chosen = []
    for candidate in candidates:
        if any(not (candidate["end"] <= other["start"] or candidate["start"] >= other["end"]) for other in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= max(1, max_cuts):
            break

    chosen.sort(key=lambda item: item["center"])
    overall_confidence = min((item["confidence"] for item in chosen), default=0.0)
    return chosen, overall_confidence

def build_projection_child_regions(region, rect, ink_mask, axis, valleys, rgb_image, bubbles):
    cuts = [item["center"] for item in valleys]
    axis_len = ink_mask.shape[1] if axis == "x" else ink_mask.shape[0]
    boundaries = [0] + cuts + [axis_len]
    child_regions = []
    parent_rect = list(rect)
    projection_confidence = min((item["confidence"] for item in valleys), default=0.0)
    valley_count = len(valleys)
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < max(28, PROJECTION_VALLEY_MIN_WIDTH * 2):
            return [], "child_region_too_small"
        if axis == "x":
            segment_mask = ink_mask[:, start:end]
        else:
            segment_mask = ink_mask[start:end, :]
        ys, xs = np.where(segment_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return [], "child_region_too_small"

        if axis == "x":
            child_local_rect = (
                start + int(xs.min()),
                int(ys.min()),
                start + int(xs.max()) + 1,
                int(ys.max()) + 1,
            )
        else:
            child_local_rect = (
                int(xs.min()),
                start + int(ys.min()),
                int(xs.max()) + 1,
                start + int(ys.max()) + 1,
            )

        child_local_rect = (
            max(0, child_local_rect[0] - 2),
            max(0, child_local_rect[1] - 2),
            min(ink_mask.shape[1], child_local_rect[2] + 2),
            min(ink_mask.shape[0], child_local_rect[3] + 2),
        )
        child_rect = clamp_rect(
            (
                rect[0] + child_local_rect[0],
                rect[1] + child_local_rect[1],
                rect[0] + child_local_rect[2],
                rect[1] + child_local_rect[3],
            ),
            rgb_image.shape
        )
        child_w = max(1, child_rect[2] - child_rect[0])
        child_h = max(1, child_rect[3] - child_rect[1])
        child_area = child_w * child_h
        child_ink = int(np.count_nonzero(segment_mask > 0))
        if child_w < 18 or child_h < 18 or child_area < PROJECTION_CHILD_MIN_AREA or child_ink < 24:
            return [], "child_region_too_small"

        child_box = rect_to_box(child_rect)
        bubble_index = find_enclosing_bubble_index(child_box, bubbles)
        child_region = {
            "box": child_box,
            "easy_text": "",
            "easy_conf": 0.0,
            "bubble_index": bubble_index,
            "parts": [{
                "box": [list(pt) for pt in child_box],
                "easy_text": "",
                "easy_conf": 0.0,
                "bubble_index": bubble_index,
            }],
            "source_count": 1,
        }
        with_single_large_split_meta(
            child_region,
            attempted=True,
            applied=True,
            split_source="projection",
            split_parent_rect=parent_rect,
            projection_axis=axis,
            split_confidence=projection_confidence,
            projection_valley_count=valley_count,
        )
        child_regions.append(child_region)

    if len(child_regions) <= 1:
        return [], "would_over_split"
    return child_regions, ""

def split_single_large_region_by_projection(region, rgb_image, bubbles):
    region = with_single_large_split_meta(dict(region), attempted=False, applied=False)
    if not ENABLE_SINGLE_LARGE_SPLIT or int(region.get("source_count", 1)) != 1:
        return [region]

    rect = clamp_rect(box_to_rect(region["box"]), rgb_image.shape)
    width = max(1, rect[2] - rect[0])
    height = max(1, rect[3] - rect[1])
    area = width * height
    long_side = max(width, height)
    if area < SINGLE_LARGE_SPLIT_MIN_AREA or long_side < SINGLE_LARGE_SPLIT_MIN_SIDE:
        return [with_single_large_split_meta(region, attempted=False, applied=False, blocker="area_below_threshold")]

    bubble_index = region.get("bubble_index", -1)
    if bubble_index < 0:
        bubble_index = find_enclosing_bubble_index(region["box"], bubbles)
    bubble = bubbles[bubble_index] if 0 <= bubble_index < len(bubbles) else None
    inside_bubble = bubble is not None

    crop = crop_by_rect(rgb_image, rect)
    if crop is None:
        return [with_single_large_split_meta(region, attempted=True, applied=False, blocker="empty_crop")]

    local_bubble = compute_crop_bubble_features(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
    aspect = width / max(1.0, height)
    outline_contrast = float(local_bubble["outline_contrast"])
    interior_uniformity = float(local_bubble["interior_uniformity"])
    colored_bubble_like = bool(local_bubble["colored_bubble_like"])
    bubble_like_score = float(local_bubble["bubble_like_score"])
    overlap_with_bubble = 0.0
    if bubble is not None:
        overlap_with_bubble = bubble_overlap_ratio(rect, get_bubble_bbox(bubble))
        outline_contrast = max(outline_contrast, float(bubble.get("outline_contrast", 0.0)))
        interior_uniformity = max(interior_uniformity, float(bubble.get("interior_uniformity", 0.0)))
        colored_bubble_like = colored_bubble_like or bool(bubble.get("colored_bubble_like", False))
        bubble_like_score = max(bubble_like_score, float(bubble.get("bubble_like_score", 0.0)))
        bubble_like_score += min(0.75, overlap_with_bubble * 0.9 + float(bubble.get("fill_ratio", 0.0)) * 0.35)

    outlined_bubble_like = (
        outline_contrast >= BUBBLE_OUTLINE_MIN_CONTRAST
        and interior_uniformity >= 0.18
        and edge_density <= 0.42
        and 0.18 <= aspect <= 5.8
    )
    bubble_like_candidate = (
        inside_bubble
        or overlap_with_bubble >= 0.14
        or colored_bubble_like
        or outlined_bubble_like
        or bubble_like_score >= max(1.2, COLORED_BUBBLE_SCORE_THRESHOLD - 0.2)
    )
    if not bubble_like_candidate:
        return [with_single_large_split_meta(region, attempted=True, applied=False, blocker="not_bubble_like")]

    axis, axis_blocker = choose_projection_axis_for_region(rect)
    axis_candidates = [axis] if axis else []
    if not axis_candidates:
        axis_candidates = ["x", "y"] if abs(width - height) <= max(width, height) * 0.22 else ["x" if width >= height else "y"]

    ink_mask = build_projection_ink_mask(crop)
    if not np.any(ink_mask > 0):
        return [with_single_large_split_meta(region, attempted=True, applied=False, blocker="projection_valley_not_clear")]

    best_valleys = []
    best_axis = ""
    best_confidence = 0.0
    for axis_name in axis_candidates:
        axis_length = ink_mask.shape[1] if axis_name == "x" else ink_mask.shape[0]
        min_width = max(PROJECTION_VALLEY_MIN_WIDTH, axis_length // 18)
        valleys, confidence = compute_projection_valleys(
            ink_mask,
            axis_name,
            min_width=min_width,
            max_cuts=SINGLE_LARGE_SPLIT_MAX_CUTS,
        )
        if len(valleys) >= 1 and confidence > best_confidence:
            best_valleys = valleys
            best_axis = axis_name
            best_confidence = confidence

    if not best_valleys:
        blocker = axis_blocker or "projection_valley_not_clear"
        return [with_single_large_split_meta(region, attempted=True, applied=False, blocker=blocker)]

    if not axis and best_confidence < 0.62:
        return [with_single_large_split_meta(region, attempted=True, applied=False, blocker="would_over_split")]

    child_regions, child_blocker = build_projection_child_regions(
        region,
        rect,
        ink_mask,
        best_axis,
        best_valleys,
        rgb_image,
        bubbles,
    )
    if not child_regions:
        blocker = child_blocker or "child_region_too_small"
        return [with_single_large_split_meta(region, attempted=True, applied=False, blocker=blocker,
                                             projection_axis=best_axis,
                                             split_confidence=best_confidence,
                                             projection_valley_count=len(best_valleys))]
    return child_regions

def split_single_large_regions(regions, rgb_image, bubbles):
    refined = []
    for region in regions:
        refined.extend(split_single_large_region_by_projection(region, rgb_image, bubbles))
    return refined

def maybe_split_large_region(region):
    parts = region.get("parts") or []
    if not ENABLE_LARGE_REGION_SPLIT or len(parts) < 2:
        return [region]

    rect = box_to_rect(region["box"])
    merged_w = max(1, rect[2] - rect[0])
    merged_h = max(1, rect[3] - rect[1])
    merged_area = merged_w * merged_h
    part_rects = [box_to_rect(part["box"]) for part in parts]
    part_widths = [max(1, r[2] - r[0]) for r in part_rects]
    part_heights = [max(1, r[3] - r[1]) for r in part_rects]
    median_w = float(np.median(part_widths)) if part_widths else 0.0
    median_h = float(np.median(part_heights)) if part_heights else 0.0

    split_axis = None
    if merged_area >= LARGE_REGION_SPLIT_MIN_AREA:
        if merged_w >= max(220, median_w * 3.0) and merged_h >= max(180, median_h * 1.6):
            split_axis = 0
        elif merged_h >= max(260, median_h * 3.0):
            split_axis = 1
    if split_axis is None:
        return [region]

    ordered = sorted(parts, key=lambda item: box_to_rect(item["box"])[split_axis])
    sizes = []
    gaps = []
    for item in ordered:
        r = box_to_rect(item["box"])
        sizes.append(max(1, (r[2] - r[0]) if split_axis == 0 else (r[3] - r[1])))
    for idx in range(len(ordered) - 1):
        a = box_to_rect(ordered[idx]["box"])
        b = box_to_rect(ordered[idx + 1]["box"])
        a_end = a[2] if split_axis == 0 else a[3]
        b_start = b[0] if split_axis == 0 else b[1]
        gaps.append(max(0, b_start - a_end))

    median_size = float(np.median(sizes)) if sizes else 0.0
    gap_threshold = max(18.0, median_size * LARGE_REGION_SPLIT_GAP_RATIO)
    split_points = [idx + 1 for idx, gap in enumerate(gaps) if gap >= gap_threshold]
    if not split_points:
        return [region]

    groups = []
    start = 0
    for split_idx in split_points + [len(ordered)]:
        subgroup = ordered[start:split_idx]
        if subgroup:
            groups.append(subgroup)
        start = split_idx

    if len(groups) <= 1:
        return [region]

    split_regions = []
    order_key = (lambda item: box_to_rect(item["box"])[0]) if split_axis == 0 else (lambda item: box_to_rect(item["box"])[1])
    for subgroup in groups:
        merged = merge_region_group(subgroup, order_key=order_key)
        merged["split_from_large_region"] = True
        merged["split_axis"] = "x" if split_axis == 0 else "y"
        merged["source_count"] = len(merged.get("parts") or subgroup)
        split_regions.append(merged)
    return split_regions

def split_large_regions(regions):
    refined = []
    for region in regions:
        refined.extend(maybe_split_large_region(region))
    return refined

def merge_same_line_regions(regions, x_gap=20, y_overlap_ratio=0.7):
    if len(regions) <= 1:
        return regions

    rects = [box_to_rect(region["box"]) for region in regions]
    merged = []
    used = [False] * len(regions)

    for i in range(len(regions)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        while True:
            changed = False
            for j in range(len(regions)):
                if used[j]:
                    continue
                for idx in group:
                    if regions[idx].get("split_source") == "projection" or regions[j].get("split_source") == "projection":
                        continue
                    if regions[idx].get("bubble_index", -1) != -1 or regions[j].get("bubble_index", -1) != -1:
                        if regions[idx].get("bubble_index", -1) != regions[j].get("bubble_index", -1):
                            continue
                    x1, y1, x2, y2 = rects[idx]
                    x3, y3, x4, y4 = rects[j]
                    overlap_y = max(0, min(y2, y4) - max(y1, y3))
                    min_h = min(y2 - y1, y4 - y3)
                    if min_h <= 0 or overlap_y / min_h < y_overlap_ratio:
                        continue
                    gap = max(x1, x3) - min(x2, x4) if (x2 < x3 or x4 < x1) else 0
                    if gap < x_gap:
                        group.append(j)
                        used[j] = True
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break

        if len(group) == 1:
            merged.append(regions[group[0]])
        else:
            merged.append(merge_region_group([regions[idx] for idx in group], order_key=lambda item: box_to_rect(item["box"])[0]))
    return merged

def merge_vertical_regions(regions, y_gap=25, x_overlap_ratio=0.6):
    if len(regions) <= 1:
        return regions

    rects = [box_to_rect(region["box"]) for region in regions]
    merged = []
    used = [False] * len(regions)

    for i in range(len(regions)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        while True:
            changed = False
            for j in range(len(regions)):
                if used[j]:
                    continue
                for idx in group:
                    if regions[idx].get("split_source") == "projection" or regions[j].get("split_source") == "projection":
                        continue
                    if regions[idx].get("bubble_index", -1) != -1 or regions[j].get("bubble_index", -1) != -1:
                        if regions[idx].get("bubble_index", -1) != regions[j].get("bubble_index", -1):
                            continue
                    x1, y1, x2, y2 = rects[idx]
                    x3, y3, x4, y4 = rects[j]
                    overlap_x = max(0, min(x2, x4) - max(x1, x3))
                    min_w = min(x2 - x1, x4 - x3)
                    if min_w <= 0 or overlap_x / min_w < x_overlap_ratio:
                        continue
                    gap = max(y1, y3) - min(y2, y4) if (y2 < y3 or y4 < y1) else 0
                    if gap < y_gap:
                        group.append(j)
                        used[j] = True
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break

        if len(group) == 1:
            merged.append(regions[group[0]])
        else:
            merged.append(merge_region_group([regions[idx] for idx in group], order_key=lambda item: box_to_rect(item["box"])[1]))
    return merged

def classify_text_region(region, rgb_full, bubbles):
    rect = box_to_rect(region["box"])
    x_min, y_min, x_max, y_max = rect
    w = x_max - x_min
    h = y_max - y_min
    area = w * h
    crop = crop_by_rect(rgb_full, rect)
    easy_text = normalize_rule_text(region.get("easy_text", ""))
    easy_conf = float(region.get("easy_conf", 0.0))
    inside_bubble = region.get("bubble_index", -1) >= 0

    info = {
        "rect": rect,
        "easy_text": easy_text,
        "easy_conf": easy_conf,
        "inside_bubble": inside_bubble,
        "area": area,
        "aspect": (w / h) if h else 999,
        "polygon_fill_ratio": 0.0,
    }

    if crop is None or area <= 0:
        return "unknown_skip", "empty_crop", info

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    white_bg = has_white_background(crop)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / max(1, crop.shape[0] * crop.shape[1])
    long_or_sentence = looks_like_japanese_sentence(easy_text)
    sfx_traits = detect_sfx_traits(easy_text)
    sfx_like = looks_like_sfx(easy_text, inside_bubble=inside_bubble)
    poly_area = polygon_area(region["box"])
    polygon_fill_ratio = poly_area / max(1.0, area)
    dialogue_score = 0.0
    narration_score = 0.0
    sfx_score = 0.0
    bubble_text_score = 0.0
    classification_rescue = False

    info.update({
        "white_bg": white_bg,
        "edge_density": edge_density,
        "long_or_sentence": long_or_sentence,
        "sfx_like": sfx_like,
        "sfx_traits": sfx_traits,
        "polygon_fill_ratio": polygon_fill_ratio,
        "bubble_text_score": 0.0,
        "classification_rescue": False,
    })

    if inside_bubble:
        dialogue_score += 2.6
    else:
        narration_score += 0.3
        sfx_score += 0.2

    if white_bg:
        dialogue_score += 1.4 if inside_bubble else 0.2
        narration_score += 1.8 if not inside_bubble else 0.2
    else:
        sfx_score += 1.8

    if edge_density <= 0.26:
        dialogue_score += 0.9
        narration_score += 0.7
    elif edge_density >= 0.4:
        sfx_score += 1.5

    if 450 <= area <= 50000:
        dialogue_score += 0.8 if inside_bubble else 0.0
        narration_score += 0.7 if not inside_bubble else 0.0
    elif area < 350:
        sfx_score += 1.2

    if 0.22 <= info["aspect"] <= 4.8:
        narration_score += 0.7
        dialogue_score += 0.5
    else:
        sfx_score += 1.2

    if polygon_fill_ratio >= 0.72:
        narration_score += 0.7
        dialogue_score += 0.4
    elif polygon_fill_ratio <= 0.45:
        sfx_score += 0.8

    if long_or_sentence:
        dialogue_score += 0.7 if inside_bubble else 0.2
        narration_score += 0.7 if not inside_bubble else 0.1
    if contains_kanji(easy_text):
        dialogue_score += 0.4
        narration_score += 0.4
    if is_short_dialogue_candidate(easy_text, inside_bubble):
        dialogue_score += 1.2
    if inside_bubble:
        bubble_text_score, bubble_reasons = score_bubble_text_candidate(
            easy_text,
            confidence=easy_conf,
            source="easyocr"
        )
        info["bubble_text_reasons"] = bubble_reasons
        if bubble_text_score >= 1.3:
            dialogue_score += min(1.2, 0.5 + bubble_text_score * 0.25)
        if easy_conf >= BUBBLE_RESCUE_MIN_CONFIDENCE and len(easy_text) <= BUBBLE_SHORT_TEXT_MAX_LEN and not looks_like_sfx(easy_text, inside_bubble=True):
            dialogue_score += 0.45

    if sfx_like:
        sfx_score += 1.4 + 0.3 * len(sfx_traits)
    if len(easy_text) <= SFX_SKIP_MAX_LEN and easy_conf < MIN_OCR_CONFIDENCE:
        sfx_score += 0.8
    if not inside_bubble and len(easy_text) <= SFX_SKIP_MAX_LEN and is_kana_only(easy_text):
        sfx_score += 1.2
    if not inside_bubble and not white_bg:
        sfx_score += 1.0
    if inside_bubble and sfx_like and not is_short_dialogue_candidate(easy_text, True):
        dialogue_score -= 0.8

    scores = {
        "dialogue_bubble": dialogue_score,
        "narration_box": narration_score,
        "sfx_or_noise": sfx_score,
    }
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1]

    info.update({
        "dialogue_score": dialogue_score,
        "narration_score": narration_score,
        "sfx_score": sfx_score,
        "best_score": best_score,
        "second_score": second_score,
        "bubble_text_score": bubble_text_score,
        "classification_rescue": classification_rescue,
    })

    if best_type == "dialogue_bubble" and best_score < DIALOGUE_SCORE_THRESHOLD:
        return "unknown_skip", "dialogue_score_too_low", info
    if best_type == "narration_box" and best_score < NARRATION_SCORE_THRESHOLD:
        return "unknown_skip", "narration_score_too_low", info
    margin_threshold = BUBBLE_CLASSIFICATION_MARGIN if inside_bubble else CLASSIFICATION_MARGIN
    if best_score - second_score < margin_threshold:
        if inside_bubble and bubble_text_score >= 1.75 and dialogue_score + 0.1 >= sfx_score:
            info["classification_rescue"] = True
            return "dialogue_bubble", "dialogue_bubble_rescue", info
        return "unknown_skip", "classification_margin_too_small", info
    return best_type, f"score:{best_type}", info

def save_debug_preview(image_rgb, debug_items, debug_output_path):
    if not debug_output_path:
        return

    preview = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(preview)
    colors = {
        "dialogue_bubble": (0, 180, 0),
        "narration_box": (0, 120, 255),
        "sfx_or_noise": (255, 120, 0),
        "unknown_skip": (180, 0, 0),
    }

    for item in debug_items:
        rect = item["rect"]
        color = colors.get(item["region_type"], (255, 255, 0))
        label = f'{item["region_type"]}:{item["reason"]}'
        if item.get("render_mode"):
            label += f'/{item["render_mode"]}'
        if item.get("split_from_large_region") and item.get("projection_axis"):
            label += f'/split:{item["projection_axis"]}'
        if item.get("spiky_bubble_like"):
            label += '/spiky'
        draw.rectangle(rect, outline=color, width=3)
        if item.get("split_parent_rect"):
            draw.rectangle(item["split_parent_rect"], outline=(255, 80, 80), width=2)
        if item.get("search_rect"):
            draw.rectangle(item["search_rect"], outline=(255, 220, 0), width=2)
        if item.get("bubble_bbox"):
            outline_color = (255, 0, 255) if item.get("fallback_mode") == "narration_small_rect" else (255, 255, 255)
            draw.rectangle(item["bubble_bbox"], outline=outline_color, width=2)
        if item.get("content_rect"):
            draw.rectangle(item["content_rect"], outline=(120, 255, 255), width=2)
        if item.get("core_content_rect"):
            draw.rectangle(item["core_content_rect"], outline=(255, 64, 180), width=2)
        if item.get("classification_rescue") or item.get("crop_retry_applied"):
            marker_x = rect[0] + 6
            marker_y = rect[1] + 6
            draw.ellipse((marker_x, marker_y, marker_x + 10, marker_y + 10), fill=(255, 0, 0))
        if item.get("structured_dialogue_candidate"):
            draw.rectangle((rect[0] + 12, rect[1] + 6, rect[0] + 20, rect[1] + 14), fill=(255, 220, 0))
        draw.text((rect[0] + 2, max(0, rect[1] - 16)), label, fill=color)

    debug_output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(debug_output_path, quality=95)

def save_debug_summary(debug_items, debug_output_path):
    if not debug_output_path:
        return
    summary_path = debug_output_path.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(debug_items, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

# ================== 单张图片处理流水线 ==================
def load_font_from_paths(font_paths, size):
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()

def measure_vertical_layout(draw, paragraphs, font_paths, size, box_height, col_spacing, row_spacing):
    font = load_font_from_paths(font_paths, size)
    bbox = draw.textbbox((0, 0), "测", font=font)
    char_w = max(1, bbox[2] - bbox[0])
    char_h = max(1, bbox[3] - bbox[1])
    max_chars_per_col = max(1, int((box_height + row_spacing) // max(1, char_h + row_spacing)))

    columns_chars = []
    for para in paragraphs:
        chars = list(para)
        if not chars:
            continue
        idx = 0
        while idx < len(chars):
            remain = len(chars) - idx
            chars_in_col = min(max_chars_per_col, remain)
            columns_chars.append(chars[idx:idx + chars_in_col])
            idx += chars_in_col

    total_cols = len(columns_chars)
    total_width = total_cols * char_w + max(0, total_cols - 1) * col_spacing
    return {
        "font": font,
        "font_size": size,
        "char_w": char_w,
        "char_h": char_h,
        "columns_chars": columns_chars,
        "total_width": total_width,
        "col_spacing": col_spacing,
        "row_spacing": row_spacing,
    }

def draw_text_vertical(draw, box, text, font_paths,
                       max_font_size=60, min_font_size=8,
                       width_fill_ratio=0.9, margin=5,
                       col_spacing=3, row_spacing=1,
                       return_meta=False):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_width = x_max - x_min - 2 * margin
    box_height = y_max - y_min - 2 * margin
    if box_width <= 0 or box_height <= 0:
        meta = {"overflow_strategy": "invalid_box", "fit_scale": 0.0}
        return (False, meta) if return_meta else False

    paragraphs = text.split("\n")
    variants = [
        ("compact", min(width_fill_ratio, RENDER_MAX_FILL_RATIO), max(0, col_spacing), max(0, row_spacing)),
        ("tight_columns", min(width_fill_ratio, max(0.62, RENDER_MAX_FILL_RATIO - 0.04)), max(0, col_spacing - 1), max(0, row_spacing)),
        ("tight_all", min(width_fill_ratio, max(0.58, RENDER_MAX_FILL_RATIO - 0.08)), max(0, col_spacing - 1), max(0, row_spacing - 1)),
    ]

    def find_layout(target_width, variant_col_spacing, variant_row_spacing):
        lo, hi = min_font_size, max_font_size
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = measure_vertical_layout(
                draw,
                paragraphs,
                font_paths,
                mid,
                box_height,
                variant_col_spacing,
                variant_row_spacing,
            )
            if candidate["columns_chars"] and candidate["total_width"] <= target_width:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    best_layout = None
    chosen_variant = "overflow"
    for variant_name, variant_fill_ratio, variant_col_spacing, variant_row_spacing in variants:
        target_width = max(1, int(box_width * variant_fill_ratio))
        candidate = find_layout(target_width, variant_col_spacing, variant_row_spacing)
        if candidate is None:
            candidate = find_layout(box_width, variant_col_spacing, variant_row_spacing)
        if candidate is None or not candidate["columns_chars"]:
            continue
        candidate["fit_scale"] = candidate["font_size"] / max(1.0, float(max_font_size))
        candidate["variant_name"] = variant_name
        if candidate["total_width"] > box_width * TEXT_OVERFLOW_SKIP_THRESHOLD:
            continue
        if best_layout is None or candidate["font_size"] > best_layout["font_size"] or (
            candidate["font_size"] == best_layout["font_size"] and candidate["total_width"] < best_layout["total_width"]
        ):
            best_layout = candidate
            chosen_variant = variant_name

    if best_layout is None:
        meta = {"overflow_strategy": "layout_not_fit", "fit_scale": 0.0}
        return (False, meta) if return_meta else False

    if (
        best_layout["fit_scale"] < TEXT_FIT_MIN_SCALE
        and len(normalize_rule_text(text)) > 6
        and (box_width < 62 or box_height < 112)
    ):
        meta = {
            "overflow_strategy": "font_scale_below_threshold",
            "fit_scale": round(float(best_layout["fit_scale"]), 4),
            "font_size": best_layout["font_size"],
        }
        return (False, meta) if return_meta else False

    font = best_layout["font"]
    char_w = best_layout["char_w"]
    char_h = best_layout["char_h"]
    columns_chars = best_layout["columns_chars"]
    total_width = best_layout["total_width"]
    variant_col_spacing = best_layout["col_spacing"]
    variant_row_spacing = best_layout["row_spacing"]

    start_x_right = x_max - margin - char_w
    x_center_offset = max(0, (box_width - total_width) // 2)
    start_x = start_x_right - x_center_offset

    for col_idx, chars in enumerate(columns_chars):
        col_x = start_x - col_idx * (char_w + variant_col_spacing)
        col_text_height = len(chars) * (char_h + variant_row_spacing) - variant_row_spacing
        start_y = y_min + margin + max(0, (box_height - col_text_height) // 2)
        for row_idx, ch in enumerate(chars):
            char_y = start_y + row_idx * (char_h + variant_row_spacing)
            draw.text((col_x, char_y), ch, fill=(0, 0, 0), font=font)

    meta = {
        "overflow_strategy": chosen_variant,
        "fit_scale": round(float(best_layout["fit_scale"]), 4),
        "font_size": best_layout["font_size"],
        "total_width": int(total_width),
    }
    return (True, meta) if return_meta else True

def draw_text_in_box(draw, box, text, font_paths,
                     max_font_size=30, min_font_size=8, margin=3,
                     return_meta=False):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_width = x_max - x_min - 2 * margin
    box_height = y_max - y_min - 2 * margin
    if box_width <= 0 or box_height <= 0:
        meta = {"overflow_strategy": "invalid_box", "fit_scale": 0.0}
        return (False, meta) if return_meta else False

    compact = text.replace("\n", "")
    best = None
    for size in range(max_font_size, min_font_size - 1, -1):
        font = load_font_from_paths(font_paths, size)
        probe_bbox = draw.textbbox((0, 0), "测", font=font)
        char_w = max(1, probe_bbox[2] - probe_bbox[0])
        line_height = max(1, probe_bbox[3] - probe_bbox[1])
        chars_per_line = max(1, box_width // char_w)
        lines = [compact[i:i + chars_per_line] for i in range(0, len(compact), chars_per_line)] or [compact]
        total_height = len(lines) * line_height
        max_width = 0
        for line in lines:
            line_bbox = draw.textbbox((0, 0), line, font=font)
            max_width = max(max_width, line_bbox[2] - line_bbox[0])
        if total_height <= box_height and max_width <= box_width:
            best = (font, size, lines, line_height, max_width)
            break

    if best is None:
        meta = {"overflow_strategy": "layout_not_fit", "fit_scale": 0.0}
        return (False, meta) if return_meta else False

    font, size, lines, line_height, max_width = best
    fit_scale = size / max(1.0, float(max_font_size))
    if fit_scale < TEXT_FIT_MIN_SCALE and len(compact) > 8 and (box_width < 84 or box_height < 84):
        meta = {"overflow_strategy": "font_scale_below_threshold", "fit_scale": round(float(fit_scale), 4)}
        return (False, meta) if return_meta else False

    y = y_min + margin + max(0, (box_height - len(lines) * line_height) // 2)
    for line in lines:
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_width = line_bbox[2] - line_bbox[0]
        x = x_min + margin + max(0, (box_width - line_width) // 2)
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += line_height

    meta = {
        "overflow_strategy": "horizontal_fit",
        "fit_scale": round(float(fit_scale), 4),
        "font_size": size,
        "total_width": int(max_width),
    }
    return (True, meta) if return_meta else True

def bubble_overlap_ratio(rect, bubble_bbox):
    return rect_intersection_area(rect, bubble_bbox) / max(1.0, rect_area(rect))

def classify_text_region(region, rgb_full, bubbles):
    raw_rect = box_to_rect(region["box"])
    rect = clamp_rect(raw_rect, rgb_full.shape)
    crop_retry_applied = rect != raw_rect
    x_min, y_min, x_max, y_max = rect
    w = x_max - x_min
    h = y_max - y_min
    area = w * h
    crop = crop_by_rect(rgb_full, rect)
    easy_text = normalize_rule_text(region.get("easy_text", ""))
    easy_conf = float(region.get("easy_conf", 0.0))
    bubble_index = region.get("bubble_index", -1)
    if bubble_index < 0 or crop_retry_applied:
        bubble_index = find_enclosing_bubble_index(rect_to_box(rect), bubbles)
    inside_bubble = bubble_index >= 0
    bubble = bubbles[bubble_index] if inside_bubble and bubble_index < len(bubbles) else None

    info = {
        "rect": rect,
        "easy_text": easy_text,
        "easy_conf": easy_conf,
        "inside_bubble": inside_bubble,
        "bubble_index": bubble_index,
        "area": area,
        "aspect": (w / h) if h else 999,
        "polygon_fill_ratio": 0.0,
        "classification_blocker": "",
        "crop_retry_applied": crop_retry_applied,
    }
    if crop is None or area <= 0:
        info["classification_blocker"] = "empty_crop"
        return "unknown_skip", "empty_crop", info

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    white_bg = has_white_background(crop)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / max(1, crop.shape[0] * crop.shape[1])
    local_bubble = compute_crop_bubble_features(crop)
    outline_contrast = float(local_bubble["outline_contrast"])
    interior_uniformity = float(local_bubble["interior_uniformity"])
    mean_saturation = float(local_bubble["mean_saturation"])
    colored_bubble_like = bool(local_bubble["colored_bubble_like"])
    spiky_bubble_like = bool(local_bubble.get("spiky_bubble_like", False))
    bubble_like_score = float(local_bubble["bubble_like_score"])
    overlap_with_bubble = 0.0
    bubble_mask_hit = False
    bubble_match_mode = ""
    if bubble is not None:
        bubble_match = compute_bubble_match_details(rect, bubble)
        bubble_bbox = get_bubble_bbox(bubble)
        overlap_with_bubble = bubble_match["overlap"]
        bubble_mask_hit = bubble_match["bubble_mask_hit"]
        bubble_match_mode = bubble_match["bubble_match_mode"]
        outline_contrast = max(outline_contrast, float(bubble.get("outline_contrast", 0.0)))
        interior_uniformity = max(interior_uniformity, float(bubble.get("interior_uniformity", 0.0)))
        mean_saturation = max(mean_saturation, float(bubble.get("mean_saturation", 0.0)))
        colored_bubble_like = colored_bubble_like or bool(bubble.get("colored_bubble_like", False))
        spiky_bubble_like = spiky_bubble_like or bool(bubble.get("spiky_bubble_like", False))
        bubble_like_score = max(bubble_like_score, float(bubble.get("bubble_like_score", 0.0)))
        bubble_like_score += min(0.75, overlap_with_bubble * 0.9 + float(bubble.get("fill_ratio", 0.0)) * 0.35)

    long_or_sentence = looks_like_japanese_sentence(easy_text)
    sfx_traits = detect_sfx_traits(easy_text)
    sfx_like = looks_like_sfx(easy_text, inside_bubble=inside_bubble)
    poly_area = polygon_area(region["box"])
    polygon_fill_ratio = poly_area / max(1.0, area)
    outlined_bubble_like = (
        outline_contrast >= BUBBLE_OUTLINE_MIN_CONTRAST
        and interior_uniformity >= 0.18
        and edge_density <= 0.42
        and 0.18 <= info["aspect"] <= 5.6
        and area >= 420
    )
    if ENABLE_SPIKY_BUBBLE_RESCUE and spiky_bubble_like:
        bubble_like_score += 0.30

    dialogue_score = 0.0
    narration_score = 0.0
    sfx_score = 0.0
    bubble_text_score = 0.0
    classification_rescue = False
    info.update({
        "white_bg": white_bg,
        "edge_density": edge_density,
        "long_or_sentence": long_or_sentence,
        "sfx_like": sfx_like,
        "sfx_traits": sfx_traits,
        "polygon_fill_ratio": polygon_fill_ratio,
        "bubble_text_score": 0.0,
        "classification_rescue": False,
        "bubble_like_score": bubble_like_score,
        "colored_bubble_like": colored_bubble_like,
        "spiky_bubble_like": spiky_bubble_like,
        "outlined_bubble_like": outlined_bubble_like,
        "outline_contrast": outline_contrast,
        "interior_uniformity": interior_uniformity,
        "overlap_with_bubble": overlap_with_bubble,
        "bubble_mask_hit": bubble_mask_hit,
        "bubble_match_mode": bubble_match_mode,
        "classification_blocker": "",
    })

    if inside_bubble:
        dialogue_score += 2.3
        narration_score += 0.1
    else:
        narration_score += 0.25
        sfx_score += 0.15

    if overlap_with_bubble >= 0.42:
        dialogue_score += 0.9
    elif overlap_with_bubble >= 0.20:
        dialogue_score += 0.45
    if bubble_mask_hit:
        dialogue_score += 0.65
    if spiky_bubble_like:
        dialogue_score += 0.55 if inside_bubble else 0.35
        sfx_score -= 0.15

    if white_bg:
        dialogue_score += 1.0 if inside_bubble else 0.25
        narration_score += 1.5 if not inside_bubble else 0.2
    elif colored_bubble_like or outlined_bubble_like or spiky_bubble_like:
        dialogue_score += 1.3 if inside_bubble else 0.8
        narration_score += 0.45 if not inside_bubble else 0.15
        sfx_score += 0.15
    else:
        sfx_score += 1.0

    if edge_density <= 0.30:
        dialogue_score += 0.8
        narration_score += 0.6
    elif edge_density >= 0.44 and not (colored_bubble_like or outlined_bubble_like or spiky_bubble_like):
        sfx_score += 1.4

    if 450 <= area <= 65000:
        dialogue_score += 0.75 if (inside_bubble or colored_bubble_like or outlined_bubble_like or spiky_bubble_like) else 0.1
        narration_score += 0.65 if not inside_bubble else 0.0
    elif area < 350 and not (inside_bubble or colored_bubble_like or outlined_bubble_like or spiky_bubble_like):
        sfx_score += 1.1

    if 0.22 <= info["aspect"] <= 5.2:
        narration_score += 0.6
        dialogue_score += 0.55
    else:
        sfx_score += 0.9

    if polygon_fill_ratio >= 0.72:
        narration_score += 0.55
        dialogue_score += 0.35
    elif polygon_fill_ratio <= 0.42 and not (colored_bubble_like or outlined_bubble_like or spiky_bubble_like):
        sfx_score += 0.8

    if bubble_like_score >= COLORED_BUBBLE_SCORE_THRESHOLD:
        dialogue_score += min(1.7, 0.55 + bubble_like_score * 0.45)
        if not inside_bubble and long_or_sentence:
            narration_score += 0.35

    if long_or_sentence:
        dialogue_score += 0.55 if (inside_bubble or bubble_like_score >= COLORED_BUBBLE_SCORE_THRESHOLD) else 0.25
        narration_score += 0.65 if not inside_bubble else 0.15
    if contains_kanji(easy_text):
        dialogue_score += 0.35
        narration_score += 0.45
    if has_dialogue_marks(easy_text):
        dialogue_score += 0.45

    short_dialogue_hint = is_short_dialogue_candidate(
        easy_text,
        inside_bubble or bubble_like_score >= COLORED_BUBBLE_SCORE_THRESHOLD or colored_bubble_like or spiky_bubble_like
    )
    if short_dialogue_hint:
        dialogue_score += 1.1

    if inside_bubble or colored_bubble_like or spiky_bubble_like or bubble_like_score >= COLORED_BUBBLE_SCORE_THRESHOLD:
        bubble_text_score, bubble_reasons = score_bubble_text_candidate(
            easy_text,
            confidence=easy_conf,
            source="easyocr"
        )
        info["bubble_text_reasons"] = bubble_reasons
        if bubble_text_score >= 1.3:
            dialogue_score += min(1.2, 0.45 + bubble_text_score * 0.28)
        if easy_conf >= BUBBLE_RESCUE_MIN_CONFIDENCE and len(easy_text) <= BUBBLE_SHORT_TEXT_MAX_LEN and not looks_like_sfx(easy_text, inside_bubble=True):
            dialogue_score += 0.35

    if sfx_like:
        sfx_bonus = 1.0 + 0.25 * len(sfx_traits)
        if colored_bubble_like or bubble_like_score >= COLORED_BUBBLE_SCORE_THRESHOLD or long_or_sentence:
            sfx_bonus -= 0.8
        sfx_score += max(0.2, sfx_bonus)
    if len(easy_text) <= SFX_SKIP_MAX_LEN and easy_conf < MIN_OCR_CONFIDENCE and not (colored_bubble_like or spiky_bubble_like or bubble_like_score >= COLORED_BUBBLE_SCORE_THRESHOLD):
        sfx_score += 0.8
    if not inside_bubble and len(easy_text) <= SFX_SKIP_MAX_LEN and is_kana_only(easy_text) and not (colored_bubble_like or outlined_bubble_like or spiky_bubble_like):
        sfx_score += 1.0
    if not inside_bubble and not white_bg and not (colored_bubble_like or outlined_bubble_like or spiky_bubble_like):
        sfx_score += 0.8
    if (colored_bubble_like or outlined_bubble_like or spiky_bubble_like) and long_or_sentence:
        dialogue_score += 0.75

    visual_colored_candidate = (
        colored_bubble_like
        or spiky_bubble_like
        or outlined_bubble_like
        or bubble_like_score >= COLORED_BUBBLE_SCORE_THRESHOLD
        or (
            mean_saturation >= 18.0
            and edge_density <= 0.34
            and 0.22 <= info["aspect"] <= 5.4
            and area >= 900
        )
    )
    if visual_colored_candidate and not inside_bubble:
        dialogue_score += 0.45

    clean_box_like = (
        outline_contrast >= BUBBLE_OUTLINE_MIN_CONTRAST * 0.95
        and edge_density <= 0.11
        and polygon_fill_ratio >= 0.92
        and 0.22 <= info["aspect"] <= 1.25
        and area >= 12000
    )
    if clean_box_like:
        narration_score += 0.85
        dialogue_score += 0.18

    structured_dialogue_candidate = (
        ENABLE_STRUCTURED_DIALOGUE_RESCUE
        and 900 <= area <= 90000
        and 0.18 <= info["aspect"] <= 5.2
        and edge_density <= 0.40
        and (
            white_bg
            or inside_bubble
            or bubble_mask_hit
            or overlap_with_bubble >= 0.14
            or outlined_bubble_like
            or spiky_bubble_like
            or visual_colored_candidate
            or clean_box_like
        )
        and not (sfx_like and len(easy_text) <= SFX_SKIP_MAX_LEN and not long_or_sentence and not has_dialogue_marks(easy_text))
        and (
            inside_bubble
            or white_bg
            or bubble_mask_hit
            or overlap_with_bubble >= 0.20
            or long_or_sentence
            or contains_kanji(easy_text)
            or len(easy_text) >= 4
            or area >= 14000
            or clean_box_like
        )
    )
    if structured_dialogue_candidate:
        if inside_bubble or spiky_bubble_like or bubble_mask_hit:
            dialogue_score += 0.38
        if (white_bg or clean_box_like) and not inside_bubble:
            narration_score += 0.55
            dialogue_score += 0.10

    small_dialogue_candidate = (
        ENABLE_SMALL_DIALOGUE_RESCUE
        and 120 <= area <= 3600
        and 0.24 <= info["aspect"] <= 5.2
        and edge_density <= 0.36
        and (inside_bubble or white_bg or visual_colored_candidate or overlap_with_bubble >= 0.14)
        and not (sfx_like and not short_dialogue_hint and not long_or_sentence and not has_dialogue_marks(easy_text))
        and (
            short_dialogue_hint
            or long_or_sentence
            or has_dialogue_marks(easy_text)
            or contains_kanji(easy_text)
            or bubble_text_score >= 1.0
            or easy_conf >= max(MIN_OCR_CONFIDENCE, BUBBLE_RESCUE_MIN_CONFIDENCE)
            or overlap_with_bubble >= 0.22
        )
    )
    if small_dialogue_candidate:
        dialogue_score += 0.28 if inside_bubble else 0.18
        if white_bg and not inside_bubble:
            narration_score += 0.15

    scores = {
        "dialogue_bubble": dialogue_score,
        "narration_box": narration_score,
        "sfx_or_noise": sfx_score,
    }
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1]

    info.update({
        "dialogue_score": dialogue_score,
        "narration_score": narration_score,
        "sfx_score": sfx_score,
        "best_score": best_score,
        "second_score": second_score,
        "bubble_text_score": bubble_text_score,
        "classification_rescue": classification_rescue,
        "small_dialogue_candidate": small_dialogue_candidate,
        "structured_dialogue_candidate": structured_dialogue_candidate,
    })

    color_rescue = (
        ENABLE_COLORED_BUBBLE_RESCUE
        and visual_colored_candidate
        and dialogue_score + (COLORED_BUBBLE_MARGIN + 0.10) >= sfx_score
        and (
            long_or_sentence
            or has_dialogue_marks(easy_text)
            or contains_kanji(easy_text)
            or len(easy_text) >= 5
            or is_stylized_dialogue(easy_text, region_type="dialogue_bubble", inside_bubble=inside_bubble)
            or (visual_colored_candidate and area >= 1200)
        )
    )

    effective_dialogue_threshold = DIALOGUE_SCORE_THRESHOLD - (0.35 if visual_colored_candidate else 0.0) - (SPIKY_BUBBLE_MARGIN_BONUS if spiky_bubble_like else 0.0)
    effective_narration_threshold = NARRATION_SCORE_THRESHOLD - (0.25 if visual_colored_candidate else 0.0)

    if best_type == "dialogue_bubble" and best_score < effective_dialogue_threshold:
        if color_rescue:
            info["classification_rescue"] = True
            info["classification_blocker"] = "dialogue_threshold_colored_rescue"
            return "dialogue_bubble", "colored_bubble_rescue", info
        if structured_dialogue_candidate and max(dialogue_score, narration_score) >= STRUCTURED_DIALOGUE_MIN_SCORE and dialogue_score + 0.20 >= sfx_score:
            info["classification_rescue"] = True
            info["classification_blocker"] = "dialogue_threshold_structured_rescue"
            return "dialogue_bubble", "structured_dialogue_rescue", info
        if small_dialogue_candidate and dialogue_score >= SMALL_DIALOGUE_MIN_SCORE and dialogue_score + 0.18 >= sfx_score:
            info["classification_rescue"] = True
            info["classification_blocker"] = "dialogue_threshold_small_dialogue_rescue"
            return "dialogue_bubble", "small_dialogue_rescue", info
        info["classification_blocker"] = "dialogue_score_too_low"
        return "unknown_skip", "dialogue_score_too_low", info
    if best_type == "narration_box" and best_score < effective_narration_threshold:
        if color_rescue:
            info["classification_rescue"] = True
            info["classification_blocker"] = "narration_threshold_colored_rescue"
            return "narration_box", "colored_bubble_rescue", info
        if structured_dialogue_candidate and max(dialogue_score, narration_score) >= STRUCTURED_DIALOGUE_MIN_SCORE and max(dialogue_score, narration_score) + 0.10 >= sfx_score:
            info["classification_rescue"] = True
            info["classification_blocker"] = "narration_threshold_structured_rescue"
            target_type = "dialogue_bubble" if (inside_bubble or spiky_bubble_like or bubble_mask_hit) and dialogue_score >= narration_score - 0.18 else "narration_box"
            return target_type, "structured_dialogue_rescue", info
        if small_dialogue_candidate and max(dialogue_score, narration_score) >= SMALL_DIALOGUE_MIN_SCORE and dialogue_score + 0.12 >= sfx_score:
            info["classification_rescue"] = True
            info["classification_blocker"] = "narration_threshold_small_dialogue_rescue"
            target_type = "dialogue_bubble" if dialogue_score >= narration_score - 0.18 else "narration_box"
            return target_type, "small_dialogue_rescue", info
        info["classification_blocker"] = "narration_score_too_low"
        return "unknown_skip", "narration_score_too_low", info

    margin_threshold = BUBBLE_CLASSIFICATION_MARGIN if inside_bubble else CLASSIFICATION_MARGIN
    if spiky_bubble_like:
        margin_threshold = max(0.18, margin_threshold - SPIKY_BUBBLE_MARGIN_BONUS)
    if best_score - second_score < margin_threshold:
        if inside_bubble and bubble_text_score >= 1.75 and dialogue_score + 0.15 >= sfx_score:
            info["classification_rescue"] = True
            info["classification_blocker"] = "dialogue_bubble_rescue"
            return "dialogue_bubble", "dialogue_bubble_rescue", info
        if structured_dialogue_candidate and max(dialogue_score, narration_score) >= STRUCTURED_DIALOGUE_MIN_SCORE and max(dialogue_score, narration_score) + 0.12 >= sfx_score:
            info["classification_rescue"] = True
            info["classification_blocker"] = "structured_dialogue_margin_rescue"
            if inside_bubble or spiky_bubble_like or bubble_mask_hit:
                target_type = "dialogue_bubble"
            elif white_bg and narration_score >= dialogue_score - 0.18:
                target_type = "narration_box"
            else:
                target_type = "dialogue_bubble" if dialogue_score >= narration_score - 0.18 else "narration_box"
            return target_type, "structured_dialogue_rescue", info
        if small_dialogue_candidate and max(dialogue_score, narration_score) >= SMALL_DIALOGUE_MIN_SCORE and dialogue_score + 0.12 >= sfx_score:
            info["classification_rescue"] = True
            info["classification_blocker"] = "small_dialogue_margin_rescue"
            target_type = "dialogue_bubble" if dialogue_score >= narration_score - 0.18 else "narration_box"
            return target_type, "small_dialogue_rescue", info
        if color_rescue:
            info["classification_rescue"] = True
            info["classification_blocker"] = "margin_colored_bubble_rescue"
            target_type = "dialogue_bubble" if dialogue_score >= narration_score - 0.2 else "narration_box"
            return target_type, "colored_bubble_rescue", info
        info["classification_blocker"] = "classification_margin_too_small"
        return "unknown_skip", "classification_margin_too_small", info

    if best_type == "sfx_or_noise" and color_rescue:
        info["classification_rescue"] = True
        info["classification_blocker"] = "sfx_overridden_by_colored_bubble_rescue"
        target_type = "dialogue_bubble" if dialogue_score >= narration_score - 0.2 else "narration_box"
        return target_type, "colored_bubble_rescue", info
    if best_type == "sfx_or_noise" and structured_dialogue_candidate and max(dialogue_score, narration_score) >= STRUCTURED_DIALOGUE_MIN_SCORE and max(dialogue_score, narration_score) + 0.10 >= sfx_score:
        info["classification_rescue"] = True
        info["classification_blocker"] = "sfx_overridden_by_structured_rescue"
        if inside_bubble or spiky_bubble_like or bubble_mask_hit:
            target_type = "dialogue_bubble"
        elif white_bg and narration_score >= dialogue_score - 0.18:
            target_type = "narration_box"
        else:
            target_type = "dialogue_bubble" if dialogue_score >= narration_score - 0.18 else "narration_box"
        return target_type, "structured_dialogue_rescue", info
    return best_type, f"score:{best_type}", info

def process_single_image(image_path, output_path, manga_ocr, reader, font_paths,
                         use_filter=True, use_vertical=True, debug_output_path=None):
    """
    处理单张漫画图片。

    use_filter: 是否启用区域分类过滤
    use_vertical: True=竖排嵌字，False=横排
    """
    print(f"处理 {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告：无法读取 {image_path}，跳过")
        return False

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bubbles = detect_bubbles_by_edges(image)

    print("  检测文字区域...")
    ocr_result = reader.readtext(
        image_path,
        text_threshold=0.5,
        low_text=0.2,
        paragraph=False,
        canvas_size=1280
    )

    regions = []
    for item in ocr_result:
        if not item:
            continue
        box = item[0]
        easy_text = normalize_rule_text(item[1]) if len(item) > 1 else ""
        easy_conf = float(item[2]) if len(item) > 2 else 0.0
        regions.append({
            "box": box,
            "easy_text": easy_text,
            "easy_conf": easy_conf,
            "bubble_index": find_enclosing_bubble_index(box, bubbles),
            "parts": [{
                "box": [list(pt) for pt in box],
                "easy_text": easy_text,
                "easy_conf": easy_conf,
                "bubble_index": find_enclosing_bubble_index(box, bubbles),
            }],
            "source_count": 1,
        })

    bubbles = augment_bubbles_from_regions(image, regions, bubbles)
    for region in regions:
        region["bubble_index"] = find_enclosing_bubble_index(region["box"], bubbles)

    regions = split_single_large_regions(regions, rgb_image, bubbles)
    bubbles = augment_bubbles_from_regions(image, regions, bubbles)
    for region in regions:
        region["bubble_index"] = find_enclosing_bubble_index(region["box"], bubbles)
    regions = merge_same_line_regions(regions, x_gap=20, y_overlap_ratio=0.7)
    for region in regions:
        region["bubble_index"] = find_enclosing_bubble_index(region["box"], bubbles)
    regions = merge_vertical_regions(regions, y_gap=25, x_overlap_ratio=0.6)
    for region in regions:
        region["bubble_index"] = find_enclosing_bubble_index(region["box"], bubbles)
    print(f"  合并后文本框数量: {len(regions)}")

    regions = split_large_regions(regions)
    for region in regions:
        region["bubble_index"] = find_enclosing_bubble_index(region["box"], bubbles)
    print(f"  澶ф鎷嗗垎鍚庢枃鏈鏁伴噺: {len(regions)}")

    pil_img = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    debug_items = []

    for i, region in enumerate(regions):
        region_type, reason, info = classify_text_region(region, rgb_image, bubbles)
        info_bubble_index = int(info.get("bubble_index", region.get("bubble_index", -1)))
        debug_entry = {
            "index": i,
            "rect": list(info["rect"]),
            "region_type": region_type,
            "reason": reason,
            "easy_text": region.get("easy_text", ""),
            "easy_conf": round(float(region.get("easy_conf", 0.0)), 4),
            "manga_ocr_text": "",
            "final_text": "",
            "translated_text": "",
            "skip_reason": "",
            "render_mode": "",
            "mask_confidence": 0.0,
            "content_rect": None,
            "core_content_rect": None,
            "bubble_bbox": list(get_bubble_bbox(bubbles[info_bubble_index])) if info_bubble_index >= 0 and info_bubble_index < len(bubbles) else None,
            "fill_color": None,
            "fallback_used": False,
            "search_rect": None,
            "candidate_count": 0,
            "candidate_area_ratio": 0.0,
            "shape_score": 0.0,
            "rejection_reason": "",
            "fallback_mode": "",
            "ocr_choice": "",
            "rescue_applied": False,
            "rescue_reason": "",
            "classification_rescue": bool(info.get("classification_rescue", False)),
            "bubble_text_score": round(float(info.get("bubble_text_score", 0.0)), 4),
            "bubble_like_score": round(float(info.get("bubble_like_score", 0.0)), 4),
            "colored_bubble_like": bool(info.get("colored_bubble_like", False)),
            "spiky_bubble_like": bool(info.get("spiky_bubble_like", False)),
            "outline_contrast": round(float(info.get("outline_contrast", 0.0)), 4),
            "interior_uniformity": round(float(info.get("interior_uniformity", 0.0)), 4),
            "overlap_with_bubble": round(float(info.get("overlap_with_bubble", 0.0)), 4),
            "bubble_mask_hit": bool(info.get("bubble_mask_hit", False)),
            "bubble_match_mode": info.get("bubble_match_mode", ""),
            "classification_blocker": info.get("classification_blocker", ""),
            "translation_drop_reason": "",
            "core_mask_area_ratio": 0.0,
            "render_shrink_applied": False,
            "overflow_strategy": "",
            "crop_retry_applied": bool(info.get("crop_retry_applied", False)),
            "structured_dialogue_candidate": bool(info.get("structured_dialogue_candidate", False)),
            "split_from_large_region": bool(region.get("split_from_large_region", False)),
            "split_source": region.get("split_source", ""),
            "split_parent_rect": list(region["split_parent_rect"]) if region.get("split_parent_rect") else None,
            "projection_axis": region.get("projection_axis", ""),
            "split_confidence": round(float(region.get("split_confidence", 0.0)), 4),
            "projection_valley_count": int(region.get("projection_valley_count", 0)),
            "single_large_split_applied": bool(region.get("single_large_split_applied", False)),
            "single_large_split_attempted": bool(region.get("single_large_split_attempted", False)),
            "single_large_split_blocker": region.get("single_large_split_blocker", ""),
            "source_count": int(region.get("source_count", len(region.get("parts") or []))),
            "ocr_filter_reason": "",
            "ocr_crop_mode": "original",
            "ocr_crop_rect": list(info["rect"]),
            "ocr_crop_from_bubble_mask": False,
            "bubble_shape_rescue": False,
            "ocr_retry_reason": "",
        }
        if ENABLE_DEBUG_CLASSIFY or ENABLE_DEBUG_JSON or ENABLE_DEBUG_RENDER:
            debug_items.append(debug_entry)

        if ENABLE_DEBUG_CLASSIFY:
            print(
                f'  区域{i} 分类: {region_type} ({reason}) '
                f'easy="{region.get("easy_text", "")}" conf={region.get("easy_conf", 0.0):.2f}'
            )

        if use_filter and region_type not in {"dialogue_bubble", "narration_box"}:
            continue

        bbox_rect = info["rect"]
        crop = crop_by_rect(rgb_image, bbox_rect)
        if crop is None:
            continue

        crop_pil = prepare_ocr_crop_pil(crop)
        manga_text = normalize_rule_text(manga_ocr(crop_pil).strip())
        inside_bubble = info["inside_bubble"]
        bubble_shape_rescue = bool(
            region_type == "dialogue_bubble" and (
                inside_bubble
                or info.get("spiky_bubble_like", False)
                or info.get("structured_dialogue_candidate", False)
                or info.get("white_bg", False)
                or info.get("bubble_mask_hit", False)
                or (
                    float(info.get("outline_contrast", 0.0)) >= BUBBLE_OUTLINE_MIN_CONTRAST * 0.95
                    and float(info.get("edge_density", 1.0)) <= 0.11
                    and float(info.get("polygon_fill_ratio", 0.0)) >= 0.92
                )
                or float(info.get("overlap_with_bubble", 0.0)) >= 0.22
            )
        )
        constrained_manga_text = ""
        if bubble_shape_rescue:
            bubble_idx = int(info.get("bubble_index", -1))
            bubble = bubbles[bubble_idx] if 0 <= bubble_idx < len(bubbles) else None
            constrained_crop, constrained_meta = choose_bubble_crop_for_ocr(rgb_image, bbox_rect, bubble)
            debug_entry["ocr_crop_mode"] = constrained_meta.get("ocr_crop_mode", "original")
            debug_entry["ocr_crop_rect"] = constrained_meta.get("ocr_crop_rect", list(bbox_rect))
            debug_entry["ocr_crop_from_bubble_mask"] = bool(constrained_meta.get("ocr_crop_from_bubble_mask", False))
            if constrained_crop is not None and constrained_meta.get("ocr_crop_mode") == "bubble_constrained":
                constrained_pil = prepare_ocr_crop_pil(constrained_crop)
                constrained_manga_text = normalize_rule_text(manga_ocr(constrained_pil).strip())
                constrained_score, _ = score_bubble_text_candidate(constrained_manga_text, 1.0, source="manga_ocr")
                original_score, _ = score_bubble_text_candidate(manga_text, 1.0, source="manga_ocr")
                if constrained_manga_text and (not manga_text or constrained_score >= original_score - 0.15):
                    if manga_text and manga_text != constrained_manga_text and original_score > constrained_score:
                        debug_entry["ocr_retry_reason"] = "kept_original_crop"
                    else:
                        debug_entry["ocr_retry_reason"] = "preferred_constrained_crop"
                    manga_text = constrained_manga_text
                elif not constrained_manga_text:
                    debug_entry["ocr_retry_reason"] = "constrained_empty_fallback_original"
                else:
                    debug_entry["ocr_retry_reason"] = "constrained_weaker_fallback_original"

        debug_entry["manga_ocr_text"] = manga_text
        original_text = ""
        source_used = ""
        rescue_reason = ""
        debug_entry["classification_rescue"] = bool(info.get("classification_rescue", False))
        debug_entry["bubble_text_score"] = round(float(info.get("bubble_text_score", 0.0)), 4)
        debug_entry["bubble_like_score"] = round(float(info.get("bubble_like_score", 0.0)), 4)
        debug_entry["colored_bubble_like"] = bool(info.get("colored_bubble_like", False))
        debug_entry["spiky_bubble_like"] = bool(info.get("spiky_bubble_like", False))
        debug_entry["outline_contrast"] = round(float(info.get("outline_contrast", 0.0)), 4)
        debug_entry["interior_uniformity"] = round(float(info.get("interior_uniformity", 0.0)), 4)
        debug_entry["overlap_with_bubble"] = round(float(info.get("overlap_with_bubble", 0.0)), 4)
        debug_entry["bubble_mask_hit"] = bool(info.get("bubble_mask_hit", False))
        debug_entry["bubble_match_mode"] = info.get("bubble_match_mode", "")
        debug_entry["classification_blocker"] = info.get("classification_blocker", "")
        debug_entry["structured_dialogue_candidate"] = bool(info.get("structured_dialogue_candidate", False))
        debug_entry["bubble_shape_rescue"] = bubble_shape_rescue

        if region_type == "dialogue_bubble" and (inside_bubble or bubble_shape_rescue):
            easy_candidate = region.get("easy_text", "")
            original_text, source_used, rescue_reason, bubble_choice_score = choose_bubble_ocr_candidate(
                manga_text,
                easy_candidate,
                region.get("easy_conf", 0.0),
                region_type=region_type,
                bubble_shape_rescue=bubble_shape_rescue,
            )
            if not original_text and bubble_shape_rescue and constrained_manga_text and constrained_manga_text != manga_text:
                original_text, source_used, rescue_reason, bubble_choice_score = choose_bubble_ocr_candidate(
                    constrained_manga_text,
                    easy_candidate,
                    region.get("easy_conf", 0.0),
                    region_type=region_type,
                    bubble_shape_rescue=True,
                )
                if original_text:
                    debug_entry["ocr_retry_reason"] = (debug_entry["ocr_retry_reason"] + "|retry_constrained_candidate").strip("|")
            if original_text:
                debug_entry["ocr_choice"] = source_used
                debug_entry["rescue_reason"] = rescue_reason
                debug_entry["rescue_applied"] = source_used in {"bubble_text_rescue", "easyocr_bubble_override", "bubble_shape_rescue"} or "bubble_" in rescue_reason
                debug_entry["bubble_text_score"] = round(max(debug_entry["bubble_text_score"], float(bubble_choice_score)), 4)
            else:
                debug_entry["skip_reason"] = f"ocr_invalid:{rescue_reason}"
                continue
        else:
            manga_ok, manga_reason = is_valid_ocr_text(
                manga_text,
                inside_bubble=inside_bubble,
                region_type=region_type,
                source="manga_ocr",
                confidence=1.0
            )
            if manga_ok:
                original_text = manga_text
                source_used = "manga_ocr"
            else:
                easy_candidate = region.get("easy_text", "")
                easy_ok, easy_reason = is_valid_ocr_text(
                    easy_candidate,
                    inside_bubble=inside_bubble,
                    region_type=region_type,
                    source="easyocr",
                    confidence=region.get("easy_conf", 0.0)
                )
                if easy_ok and len(normalize_rule_text(easy_candidate)) >= MANGA_OCR_FALLBACK_MIN_LEN:
                    original_text = normalize_rule_text(easy_candidate)
                    source_used = "easyocr_fallback"
                elif is_rescuable_short_dialogue(manga_text or easy_candidate, inside_bubble, region_type):
                    original_text = normalize_rule_text(manga_text or easy_candidate)
                    source_used = "short_dialogue_rescue"
                    rescue_reason = "legacy_short_dialogue_rescue"
                else:
                    debug_entry["skip_reason"] = f"ocr_invalid:{manga_reason}/{easy_reason}"
                    continue
            debug_entry["ocr_choice"] = source_used
            if rescue_reason:
                debug_entry["rescue_reason"] = rescue_reason
                debug_entry["rescue_applied"] = True

        original_text, source_used, rescue_reason, ocr_filter_reason = refine_ocr_candidate_with_secondary_filter(
            original_text,
            source_used,
            rescue_reason,
            manga_text,
            region.get("easy_text", ""),
            region.get("easy_conf", 0.0),
            inside_bubble,
            region_type,
        )
        if not original_text:
            debug_entry["skip_reason"] = f"ocr_anomaly:{ocr_filter_reason}"
            debug_entry["ocr_filter_reason"] = ocr_filter_reason
            continue
        debug_entry["ocr_choice"] = source_used
        debug_entry["rescue_reason"] = rescue_reason
        if "secondary_swap" in rescue_reason:
            debug_entry["rescue_applied"] = True
        if ocr_filter_reason:
            debug_entry["ocr_filter_reason"] = ocr_filter_reason
            debug_entry["rescue_applied"] = True

        debug_entry["final_text"] = original_text

        if should_skip_short_text(original_text, inside_bubble, region_type, region.get("easy_conf", 0.0)):
            debug_entry["skip_reason"] = "short_text_rule_skip"
            if ENABLE_DEBUG_CLASSIFY:
                print(f"  区域{i} 跳过 OCR 文本: {original_text}")
            continue

        print(f"  区域{i} [{region_type}/{source_used}]: {original_text}")
        translated = translate_text(original_text, target_lang="zh", region_type=region_type)
        translated = clean_translation_output(translated, original_text, region_type=region_type)
        if not translated and original_text in SHORT_TRANSLATION_MAP and not looks_like_sfx(original_text, inside_bubble=inside_bubble):
            translated = SHORT_TRANSLATION_MAP[original_text]
        if not translated and is_rescuable_short_dialogue(original_text, inside_bubble, region_type):
            translated = BUBBLE_SHORT_TRANSLATION_MAP.get(normalize_rule_text(original_text), "")
            if not translated:
                translated = BUBBLE_SHORT_TRANSLATION_MAP.get(strip_text_noise(original_text), "")
            if translated:
                debug_entry["rescue_applied"] = True
                debug_entry["rescue_reason"] = (debug_entry["rescue_reason"] + "|bubble_short_translation").strip("|")
        if not translated and TRANSLATION_RETRY_FOR_STYLIZED_DIALOGUE and is_stylized_dialogue(original_text, region_type=region_type, inside_bubble=inside_bubble):
            translated = build_stylized_dialogue_fallback(original_text)
            if translated:
                debug_entry["rescue_applied"] = True
                debug_entry["rescue_reason"] = (debug_entry["rescue_reason"] + "|stylized_dialogue_fallback").strip("|")
        if not translated:
            debug_entry["skip_reason"] = "translation_empty_or_skip"
            debug_entry["translation_drop_reason"] = "translation_empty_or_skip"
            if ENABLE_DEBUG_CLASSIFY:
                print(f"         => 跳过")
            continue
        debug_entry["translated_text"] = translated
        print(f"         => {translated}")

        render_target = build_render_target(region, region_type, info, rgb_image, bubbles)
        render_rect = render_target["core_content_rect"]
        render_box = rect_to_box(render_rect)
        debug_entry["render_mode"] = render_target["render_mode"]
        debug_entry["mask_confidence"] = render_target["mask_confidence"]
        debug_entry["content_rect"] = list(render_target["content_rect"])
        debug_entry["core_content_rect"] = list(render_target["core_content_rect"])
        debug_entry["bubble_bbox"] = render_target["bubble_bbox"]
        debug_entry["fill_color"] = render_target["fill_color"]
        debug_entry["fallback_used"] = render_target["fallback_used"]
        debug_entry["search_rect"] = render_target["search_rect"]
        debug_entry["candidate_count"] = render_target["candidate_count"]
        debug_entry["candidate_area_ratio"] = render_target["candidate_area_ratio"]
        debug_entry["shape_score"] = render_target["shape_score"]
        debug_entry["rejection_reason"] = render_target["rejection_reason"]
        debug_entry["fallback_mode"] = render_target["fallback_mode"]
        debug_entry["core_mask_area_ratio"] = render_target["core_mask_area_ratio"]
        debug_entry["render_shrink_applied"] = bool(render_target["render_shrink_applied"])

        probe_img = pil_img.copy()
        probe_draw = ImageDraw.Draw(probe_img)

        if use_vertical:
            success, layout_meta = draw_text_vertical(
                probe_draw, render_box, translated, font_paths,
                max_font_size=60, min_font_size=8,
                width_fill_ratio=RENDER_MAX_FILL_RATIO, margin=4,
                col_spacing=2, row_spacing=1,
                return_meta=True
            )
        else:
            success, layout_meta = draw_text_in_box(
                probe_draw, render_box, translated, font_paths,
                max_font_size=30, min_font_size=8, margin=4,
                return_meta=True
            )
        debug_entry["overflow_strategy"] = layout_meta.get("overflow_strategy", "")
        if not success:
            debug_entry["skip_reason"] = "text_overflow_skip"
            print("  嵌字失败，框太小")
            continue

        fill_mask = refine_fill_mask_to_layout(
            render_target["core_mask"],
            render_target["core_content_rect"],
            layout_meta,
            rgb_image.shape,
            vertical=use_vertical,
        )
        trial_img = apply_mask_fill(
            pil_img,
            fill_mask,
            tuple(render_target["fill_color"]),
            fill_alpha=render_target["fill_alpha"],
        )
        trial_draw = ImageDraw.Draw(trial_img)
        if use_vertical:
            draw_text_vertical(
                trial_draw, render_box, translated, font_paths,
                max_font_size=60, min_font_size=8,
                width_fill_ratio=RENDER_MAX_FILL_RATIO, margin=4,
                col_spacing=2, row_spacing=1
            )
        else:
            draw_text_in_box(
                trial_draw, render_box, translated, font_paths,
                max_font_size=30, min_font_size=8, margin=4
            )
        debug_entry["core_mask_area_ratio"] = round(
            rect_area(mask_to_rect(fill_mask) or render_target["core_content_rect"]) /
            max(1.0, rect_area(mask_to_rect(render_target["render_mask"]) or render_target["core_content_rect"])),
            4,
        )
        pil_img = trial_img
        draw = ImageDraw.Draw(pil_img)

    pil_img.save(output_path, quality=95)
    if (ENABLE_DEBUG_CLASSIFY or ENABLE_DEBUG_RENDER) and debug_output_path is not None:
        save_debug_preview(rgb_image, debug_items, debug_output_path)
    if ENABLE_DEBUG_JSON and debug_output_path is not None:
        save_debug_summary(debug_items, debug_output_path)
    print(f"  保存到 {output_path}")
    return True


# ================== 批量处理函数 ==================
def batch_process(input_dir, output_dir):
    """
    批量处理 input_dir 下所有支持格式的图片（包含子文件夹），结果保存到 output_dir，保持目录结构。
    """
    extensions = {".png", ".jpg", ".jpeg"}
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_path = Path(output_dir)
    debug_output_root = Path(DEBUG_OUTPUT_DIR) if (ENABLE_DEBUG_CLASSIFY or ENABLE_DEBUG_JSON or ENABLE_DEBUG_RENDER) else None

    # 递归收集所有图片文件
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f"*{ext}"))
        image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    # 去重并排序
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print("未找到任何图片文件")
        return

    print(f"共找到 {len(image_files)} 张图片")

    # 加载模型（只加载一次）
    print("加载 Manga OCR...")
    manga_ocr = MangaOcr()
    print("加载 EasyOCR 检测器 (日文)...")
    reader = easyocr.Reader(['ja'], gpu=torch.cuda.is_available())

    # 字体路径
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]

    success_count = 0
    for img_file in image_files:
        # 计算相对路径，并在输出目录中创建对应的子文件夹
        relative_path = img_file.relative_to(input_path)
        out_file = output_path / relative_path
        debug_file = debug_output_root / relative_path if debug_output_root is not None else None
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            if process_single_image(str(img_file), str(out_file), manga_ocr, reader, font_paths,
                                    debug_output_path=debug_file):
                success_count += 1
        except Exception as e:
            print(f"处理 {relative_path} 时出错: {e}")

    print(f"批量处理完成：{success_count}/{len(image_files)} 张图片处理成功")


# ================== 主函数 ==================
if __name__ == "__main__":
    # 配置输入输出文件夹（可直接修改，或改为命令行参数）
    INPUT_DIR = "input_images"   # 放原图的文件夹
    OUTPUT_DIR = "output_images" # 结果输出文件夹

    batch_process(INPUT_DIR, OUTPUT_DIR)
