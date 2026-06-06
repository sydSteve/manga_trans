from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from importlib.util import find_spec


@dataclass
class ComicTextAndBubbleDetectorStatus:
    configured_source: str
    transformers_available: bool
    torch_available: bool
    detector_loaded: bool = False
    load_error_code: str = ""
    load_error_message: str = ""
    resolved_source: str = ""
    attempted_sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "configured_source": self.configured_source,
            "transformers_available": self.transformers_available,
            "torch_available": self.torch_available,
            "detector_loaded": self.detector_loaded,
            "load_error_code": self.load_error_code,
            "load_error_message": self.load_error_message,
            "resolved_source": self.resolved_source,
            "attempted_sources": list(self.attempted_sources),
        }


class ComicTextAndBubbleDetectorBackend:
    def __init__(self, config):
        self.config = config
        detector_cfg = config.detector
        self.status = ComicTextAndBubbleDetectorStatus(
            configured_source=detector_cfg.model_path or detector_cfg.repo_path,
            transformers_available=find_spec("transformers") is not None,
            torch_available=find_spec("torch") is not None,
        )
        self.processor = None
        self.model = None
        self.torch = None
        if self.status.transformers_available and self.status.torch_available:
            self._load()
        else:
            missing = []
            if not self.status.transformers_available:
                missing.append("transformers")
            if not self.status.torch_available:
                missing.append("torch")
            self.status.load_error_code = "missing_dependencies"
            self.status.load_error_message = ",".join(missing)

    def _candidate_sources(self) -> list[str]:
        candidates: list[str] = []
        configured = (self.config.detector.model_path or "").strip()
        repo_path = (self.config.detector.repo_path or "").strip()
        for value in [configured, repo_path]:
            if value and value not in candidates:
                candidates.append(value)
        return candidates

    def _load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.torch = torch
        last_error_code = "missing_model"
        last_error_message = "No local comic-text-and-bubble-detector source configured"

        for source in self._candidate_sources():
            self.status.attempted_sources.append(source)
            try:
                source_path = Path(source)
                local_only = source_path.exists()
                processor = AutoImageProcessor.from_pretrained(
                    source,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                model = AutoModelForObjectDetection.from_pretrained(
                    source,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                if self.config.device == "cuda" and torch.cuda.is_available():
                    model = model.to("cuda")
                model.eval()
                self.processor = processor
                self.model = model
                self.status.detector_loaded = True
                self.status.load_error_code = ""
                self.status.load_error_message = ""
                self.status.resolved_source = str(source_path if source_path.exists() else source)
                return
            except Exception as exc:  # noqa: BLE001
                last_error_code, last_error_message = self._map_error(exc)

        self.status.load_error_code = last_error_code
        self.status.load_error_message = last_error_message

    @staticmethod
    def _map_error(exc: Exception) -> tuple[str, str]:
        message = str(exc)
        if "does not appear to have a file named config.json" in message:
            return "missing_model_files", message
        if "Connection error" in message or "LocalEntryNotFoundError" in message:
            return "model_not_cached", message
        return "load_failed", message

    def detect(self, image_rgb) -> list[dict[str, Any]]:
        if self.model is None or self.processor is None or self.torch is None:
            return []

        from PIL import Image

        torch = self.torch
        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        if self.config.device == "cuda" and torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image_rgb.shape[:2]], device=outputs.logits.device)
        if hasattr(self.processor, "post_process_object_detection"):
            processed = self.processor.post_process_object_detection(
                outputs,
                threshold=self.config.detector.min_confidence,
                target_sizes=target_sizes,
            )[0]
        else:
            return []

        id2label = getattr(self.model.config, "id2label", {})
        detections: list[dict[str, Any]] = []
        for score, label, box in zip(processed["scores"], processed["labels"], processed["boxes"]):
            label_id = int(label.item()) if hasattr(label, "item") else int(label)
            label_name = str(id2label.get(label_id, label_id))
            xyxy = [int(round(float(num))) for num in box.tolist()]
            detections.append(
                {
                    "label": label_name,
                    "score": float(score.item() if hasattr(score, "item") else score),
                    "box": xyxy,
                }
            )
        return detections

    def runtime_summary(self) -> dict[str, object]:
        summary = self.status.as_dict()
        summary["backend_name"] = (
            "comic_text_and_bubble_detector" if self.status.detector_loaded else "easyocr_fallback"
        )
        summary["model_path_hit"] = bool(self.status.resolved_source)
        return summary
