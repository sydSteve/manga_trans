from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    model_path: str = "../third_party/comic-text-and-bubble-detector"
    repo_path: str = ""
    min_confidence: float = 0.18
    enable_easyocr_fusion: bool = True
    fusion_iou_threshold: float = 0.45
    text_label_keywords: list[str] = field(
        default_factory=lambda: ["text", "dialogue", "caption", "narration", "speech"]
    )
    bubble_label_keywords: list[str] = field(default_factory=lambda: ["bubble", "balloon", "speech"])


@dataclass
class BubbleConfig:
    enabled: bool = True
    backend: str = "speech_bubble_segmentation"
    model_path: str = "../third_party/speech-bubble-segmentation/model.pt"
    min_confidence: float = 0.2
    fallback: str = "edges"


@dataclass
class OCRConfig:
    model_path: str = "../third_party/manga-ocr-base"
    offline_only: bool = True
    retry_with_orientation: bool = True
    language: list[str] = field(default_factory=lambda: ["ja"])
    min_text_length: int = 2
    crop_padding: int = 6
    prefill_enabled: bool = True
    prefill_min_length: int = 2
    short_text_min_length: int = 2
    prompt: str = "OCR:"
    max_new_tokens: int = 256


@dataclass
class TranslatorConfig:
    base_url: str = "https://api.deepseek.com/v1/chat/completions"
    model: str = "deepseek-chat"
    target_lang: str = "zh"
    batch_size: int = 8
    timeout_seconds: int = 45
    max_retries: int = 3
    temperature: float = 0.2
    enabled: bool = True


@dataclass
class InpaintConfig:
    bubble_inner_shrink: int = 4
    failure_fill_mode: str = "bubble_white_overlay"
    backend: str = "lama_cleaner"
    model_dir: str = "../third_party/torch"
    bubble_model: str = "lama"
    outside_text_model: str = "ldm"
    device: str = "cuda"
    fallback_backend: str = "translucent_white_overlay"
    hd_strategy: str = "Crop"
    hd_strategy_crop_margin: int = 32
    hd_strategy_crop_trigger_size: int = 512
    hd_strategy_resize_limit: int = 512
    ldm_steps: int = 20
    ldm_sampler: str = "plms"


@dataclass
class StyleConfig:
    model_path: str = "../third_party/yuzumarker-font-detection/yuzumarker-font-detection.safetensors"
    labels_path: str = "../third_party/yuzumarker-font-detection/font-labels.json"
    labels_ex_path: str = "../third_party/yuzumarker-font-detection/font-labels-ex.json"
    top_k: int = 5
    min_confidence: float = 0.2
    font_family_map: dict[str, list[str]] = field(
        default_factory=lambda: {
            "sans/gothic": [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/msgothic.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ],
            "serif/mincho": [
                "C:/Windows/Fonts/simsun.ttc",
                "C:/Windows/Fonts/msmincho.ttc",
                "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc",
            ],
            "handwritten_or_display": [
                "C:/Windows/Fonts/simkai.ttf",
                "C:/Windows/Fonts/msyh.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ],
        }
    )
    default_text_color: list[int] = field(default_factory=lambda: [0, 0, 0])
    default_stroke_color: list[int] = field(default_factory=lambda: [255, 255, 255])
    default_stroke_width: int = 2


@dataclass
class RenderConfig:
    vertical_priority: bool = True
    max_fill_ratio: float = 0.84
    dialogue_max_fill_ratio: float = 0.82
    narration_max_fill_ratio: float = 0.88
    min_font_size: int = 10
    max_font_size: int = 56
    small_box_stroke_scale: float = 0.75
    font_paths: list[str] = field(
        default_factory=lambda: [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msgothic.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
    )
    text_color: list[int] = field(default_factory=lambda: [0, 0, 0])
    stroke_color: list[int] = field(default_factory=lambda: [255, 255, 255])
    stroke_width: int = 2
    font_size_offset: int = 0
    line_spacing: int = 3
    column_spacing: int = 5
    margin: int = 4


@dataclass
class DebugConfig:
    enabled: bool = True
    output_dirname: str = "_debug"


@dataclass
class AppConfig:
    device: str = "cuda"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    bubble: BubbleConfig = field(default_factory=BubbleConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    inpaint: InpaintConfig = field(default_factory=InpaintConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_relative_path(base_dir: Path, value: str) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_config_paths(config_dir: Path, data: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(data)
    if "detector" in resolved:
        detector = dict(resolved["detector"])
        detector["model_path"] = _resolve_relative_path(config_dir, str(detector.get("model_path", "")))
        detector["repo_path"] = _resolve_relative_path(config_dir, str(detector.get("repo_path", "")))
        resolved["detector"] = detector
    if "bubble" in resolved:
        bubble = dict(resolved["bubble"])
        bubble["model_path"] = _resolve_relative_path(config_dir, str(bubble.get("model_path", "")))
        resolved["bubble"] = bubble
    if "ocr" in resolved:
        ocr = dict(resolved["ocr"])
        ocr["model_path"] = _resolve_relative_path(config_dir, str(ocr.get("model_path", "")))
        resolved["ocr"] = ocr
    if "inpaint" in resolved:
        inpaint = dict(resolved["inpaint"])
        inpaint["model_dir"] = _resolve_relative_path(config_dir, str(inpaint.get("model_dir", "")))
        resolved["inpaint"] = inpaint
    if "style" in resolved:
        style = dict(resolved["style"])
        for key in ("model_path", "labels_path", "labels_ex_path"):
            style[key] = _resolve_relative_path(config_dir, str(style.get(key, "")))
        resolved["style"] = style
    return resolved


def _to_dataclass(data: dict[str, Any]) -> AppConfig:
    return AppConfig(
        device=str(data.get("device", "cuda")),
        detector=DetectorConfig(**data.get("detector", {})),
        bubble=BubbleConfig(**data.get("bubble", {})),
        ocr=OCRConfig(**data.get("ocr", {})),
        translator=TranslatorConfig(**data.get("translator", {})),
        inpaint=InpaintConfig(**data.get("inpaint", {})),
        style=StyleConfig(**data.get("style", {})),
        render=RenderConfig(**data.get("render", {})),
        debug=DebugConfig(**data.get("debug", {})),
    )


def load_config(config_path: str | Path | None) -> AppConfig:
    default_data = asdict(AppConfig())
    loaded: dict[str, Any] = {}
    config_dir = Path.cwd()
    if config_path:
        path = Path(config_path)
        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            config_dir = path.resolve().parent

    merged = _deep_merge(default_data, loaded)
    merged = _resolve_config_paths(config_dir, merged)
    env_device = os.environ.get("MANGA_TRANSLATOR_DEVICE")
    if env_device:
        merged["device"] = env_device
    return _to_dataclass(merged)
