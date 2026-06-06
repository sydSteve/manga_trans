from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import AppConfig
from .models import TextRegion
from .utils import clamp_box, contains_kanji, expand_box, is_kana_char, kana_ratio, normalize_text


ROOT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = ROOT_DIR / "app"
TEST_MANGA_OCR_DIR = ROOT_DIR / "test" / "manga-ocr"
MANGA_OCR_IMPORT_ROOT = APP_DIR if (APP_DIR / "manga_ocr").exists() else TEST_MANGA_OCR_DIR
if str(MANGA_OCR_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(MANGA_OCR_IMPORT_ROOT))

from manga_ocr import MangaOcr


PUNCT_ONLY_CHARS = set(".!?.,()[]-~") | {"…", "。", "、", "，", "．", "？", "！", "・", "ー", "「", "」"}
ACCEPTABLE_VALIDITY = {"valid", "short_japanese", "punctuation_only"}


def _crop_for_region(
    image_rgb,
    region: TextRegion,
    padding: int,
) -> tuple[object, str, tuple[int, int, int, int]]:
    source_box = expand_box(region.bubble_bbox if region.bubble_bbox else region.box, image_rgb.shape, padding)
    x1, y1, x2, y2 = source_box
    crop_mode = "bubble_box" if region.bubble_bbox else "region_box"
    return image_rgb[y1:y2, x1:x2], crop_mode, source_box


def _contains_japanese(text: str) -> bool:
    return any(is_kana_char(ch) or ("\u4e00" <= ch <= "\u9fff") for ch in text)


def _script_ratios(text: str) -> dict[str, float]:
    compact = normalize_text(text)
    if not compact:
        return {"latin_digit": 0.0, "digit": 0.0, "japanese": 0.0, "kana": 0.0, "kanji": 0.0}
    latin_digit = sum(1 for ch in compact if ch.isascii() and (ch.isalpha() or ch.isdigit())) / len(compact)
    digit = sum(1 for ch in compact if ch.isdigit()) / len(compact)
    japanese = sum(1 for ch in compact if is_kana_char(ch) or ("\u4e00" <= ch <= "\u9fff")) / len(compact)
    kana = kana_ratio(compact)
    kanji = sum(1 for ch in compact if "\u4e00" <= ch <= "\u9fff") / len(compact)
    return {"latin_digit": latin_digit, "digit": digit, "japanese": japanese, "kana": kana, "kanji": kanji}


def _max_pattern_repeat_ratio(text: str, max_unit: int = 4) -> float:
    compact = normalize_text(text)
    if len(compact) < 8:
        return 0.0
    best = 0.0
    for unit in range(1, min(max_unit, len(compact) // 2) + 1):
        chunks = [compact[index : index + unit] for index in range(0, len(compact) - unit + 1, unit)]
        if len(chunks) < 4:
            continue
        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk] = counts.get(chunk, 0) + 1
        best = max(best, max(counts.values()) / len(chunks))
    return best


def _text_validity(
    text: str,
    min_length: int,
    short_text_min_length: int,
    crop_box: tuple[int, int, int, int] | None = None,
) -> tuple[str, str]:
    compact = normalize_text(text)
    if not compact:
        return "empty", "no_text"
    if "\ufffd" in compact:
        return "garbled", "replacement_character"
    if all(ch in PUNCT_ONLY_CHARS for ch in compact):
        return "punctuation_only", "punctuation_only"

    ratios = _script_ratios(compact)
    weird_punct_ratio = sum(1 for ch in compact if ch in PUNCT_ONLY_CHARS) / len(compact)
    mixed_scripts = ratios["japanese"] >= 0.2 and ratios["digit"] >= 0.2 and ratios["latin_digit"] >= 0.35
    if mixed_scripts and len(compact) >= 6:
        return "garbled", "mixed_scripts"
    punctuated_short_japanese = (
        len(compact) <= 24
        and ratios["japanese"] >= 0.35
        and ratios["latin_digit"] < 0.15
    )
    if weird_punct_ratio >= 0.5 and len(compact) >= 8 and _contains_japanese(compact) and not punctuated_short_japanese:
        return "garbled", "excessive_punctuation_noise"
    if ratios["latin_digit"] >= 0.7 and ratios["japanese"] < 0.2:
        return "background_microtext", "latin_digit_dominant"
    if len(compact) >= 80:
        return "garbled", "runaway_length"
    if len(compact) >= 24 and ratios["japanese"] >= 0.65 and _max_pattern_repeat_ratio(compact) >= 0.58:
        return "garbled", "repetitive_generation"

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        if height <= 24 and width >= height * 4.5 and ratios["latin_digit"] >= 0.35 and ratios["japanese"] < 0.35:
            return "background_microtext", "thin_banner"
        area = width * height
        if area > 0 and len(compact) >= max(64, int(area / 220)):
            return "garbled", "density_overflow"

    if len(compact) < min_length:
        if _contains_japanese(compact) and len(compact) >= short_text_min_length:
            return "short_japanese", "short_japanese"
        return "too_short", "too_short"

    return "valid", "valid"


def _clean_manga_ocr_output(text: str) -> str:
    compact = normalize_text(text)
    if not compact:
        return ""
    compact = compact.replace("．．．", "…").replace("...", "…")
    compact = re.sub(r"(…){2,}", "…", compact)
    compact = re.sub(r"([!?！？]){3,}", r"\1\1", compact)
    compact = re.sub(r"([。．、，]){2,}", r"\1", compact)
    return normalize_text(compact)


class OCRService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model_path = Path(str(config.ocr.model_path or "")).resolve() if config.ocr.model_path else Path()
        self.model_loaded = False
        self.model_path_hit = self.model_path.exists()
        self.load_error_code = ""
        self.load_error_message = ""
        self.device = "cpu"
        self.engine: MangaOcr | None = None
        self._init_manga_ocr()

    def _init_manga_ocr(self) -> None:
        if not self.model_path_hit:
            self.load_error_code = "missing_model_path"
            self.load_error_message = f"Model path not found: {self.model_path}"
            return
        try:
            force_cpu = self.config.device == "cpu"
            self.engine = MangaOcr(
                pretrained_model_name_or_path=str(self.model_path),
                force_cpu=force_cpu,
            )
            self.device = "cpu" if force_cpu else str(self.engine.model.device)
            self.model_loaded = True
            self.load_error_code = ""
            self.load_error_message = ""
        except Exception as exc:  # noqa: BLE001
            self.engine = None
            self.model_loaded = False
            self.load_error_code = "manga_ocr_base_unavailable"
            self.load_error_message = str(exc)

    def prefill_text_hints(self, image_rgb, regions: list[TextRegion]) -> None:
        if not self.config.ocr.prefill_enabled:
            return
        for region in regions:
            hint_text = normalize_text(region.detector_text_hint)
            if len(hint_text) >= self.config.ocr.prefill_min_length or hint_text in PUNCT_ONLY_CHARS:
                region.prefill_text = hint_text
                region.prefill_source = "detector_hint"
            else:
                region.prefill_text = ""
                region.prefill_source = ""

    def recognize(self, image_rgb, region: TextRegion) -> str:
        if region.ocr_text:
            return region.ocr_text

        crop, crop_mode, crop_box = _crop_for_region(image_rgb, region, self.config.ocr.crop_padding)
        region.debug["ocr_crop_mode"] = crop_mode
        region.debug["ocr_model_path_hit"] = self.model_path_hit
        region.debug["ocr_offline_mode"] = True
        region.debug["ocr_choice"] = "manga_ocr_base"
        region.debug["ocr_variant"] = crop_mode
        region.debug["bubble_polygon"] = list(region.bubble_polygon)

        if crop.size == 0:
            region.skip_reason = "empty_ocr_crop"
            region.debug["ocr_text_validity"] = "empty"
            region.debug["ocr_reason"] = "empty_crop"
            region.ocr_text = ""
            return ""

        if not self.model_loaded or self.engine is None:
            region.skip_reason = "ocr_backend_unavailable"
            region.debug["ocr_text_validity"] = "empty"
            region.debug["ocr_reason"] = self.load_error_code or "backend_unavailable"
            region.ocr_text = ""
            return ""

        try:
            crop_image = Image.fromarray(crop).convert("RGB")
            text = normalize_text(self.engine(crop_image))
        except Exception as exc:  # noqa: BLE001
            region.skip_reason = "ocr_runtime_error"
            region.debug["ocr_text_validity"] = "empty"
            region.debug["ocr_reason"] = f"runtime_error:{exc}"
            region.ocr_text = ""
            return ""

        validity, reason = _text_validity(
            text,
            min_length=self.config.ocr.min_text_length,
            short_text_min_length=self.config.ocr.short_text_min_length,
            crop_box=crop_box,
        )
        region.ocr_primary_text = text
        region.ocr_fallback_text = ""
        region.ocr_text = text
        region.debug["ocr_text_validity"] = validity
        region.debug["ocr_reason"] = reason
        if not text:
            region.skip_reason = region.skip_reason or "ocr_empty"
        return region.ocr_text

    def runtime_summary(self) -> dict[str, object]:
        return {
            "backend_name": "manga_ocr_base",
            "model_path_hit": self.model_path_hit,
            "model_loaded": self.model_loaded,
            "resolved_source": str(self.model_path),
            "engine_source": str(MANGA_OCR_IMPORT_ROOT / "manga_ocr"),
            "device": self.device,
            "load_error_code": self.load_error_code,
            "load_error_message": self.load_error_message,
        }
