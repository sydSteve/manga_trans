from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from .config import load_config
from .pipeline import MangaTranslationPipeline

APP_DIR = Path(__file__).resolve().parents[1]


def _safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        fixed = [str(arg).encode("gbk", errors="replace").decode("gbk") for arg in args]
        print(*fixed, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Japanese manga auto-translation pipeline")
    parser.add_argument("--input", default="../data/input_images", help="Input image directory")
    parser.add_argument("--output", default="../data/output_images", help="Output image directory")
    parser.add_argument("--config", default="config.yaml", help="YAML config file")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Override runtime device")
    parser.add_argument("--debug", action="store_true", help="Write per-image debug JSON")
    return parser


def configure_runtime(device: str) -> str:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    torch.set_num_threads(4)
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (APP_DIR / path).resolve()


def main() -> None:
    load_dotenv(APP_DIR / ".env")
    args = build_parser().parse_args()
    config = load_config(_resolve_path(args.config))
    if args.device:
        config.device = args.device
    config.device = configure_runtime(config.device)

    pipeline = MangaTranslationPipeline(config)
    input_dir = _resolve_path(args.input)
    output_dir = _resolve_path(args.output)
    success = pipeline.process_directory(input_dir, output_dir, debug=args.debug)
    _safe_print(f"Processed {success} image(s). Device={config.device}. Input={input_dir} Output={output_dir}")
    summary = pipeline.runtime_summary()
    _safe_print(
        "Active backends:",
        f"detector={summary.get('detector_backend', '')}",
        f"bubble={summary.get('bubble_backend', '')}",
        f"ocr={summary.get('ocr_backend', '')}",
        f"inpaint={summary.get('inpaint_backend', '')}",
        f"style={summary.get('style_backend', '')}",
        f"translator={'deepseek' if summary.get('translator_enabled', False) else 'local'}",
    )
    _safe_print("Model hits:", summary.get("model_hits", {}))
