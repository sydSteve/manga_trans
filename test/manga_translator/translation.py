from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict

import requests

from .config import AppConfig
from .models import TextRegion
from .utils import chunked, normalize_text


JP_PUNCT_TRANSLATION = str.maketrans(
    {
        "…": "……",
        "‥": "……",
        "。": "。",
        "、": "，",
        "！": "！",
        "？": "？",
    }
)


def clean_translation_output(translated: str, original_text: str) -> str:
    cleaned = normalize_text(translated)
    cleaned = cleaned.strip("「」『』【】\"'")
    cleaned = re.sub(r"^(翻译[:：]|译文[:：]|中文[:：])", "", cleaned)
    cleaned = cleaned.replace("...", "……").translate(JP_PUNCT_TRANSLATION)
    cleaned = re.sub(r"(……){2,}", "……", cleaned)
    if cleaned.upper() == "SKIP":
        return ""
    if cleaned == normalize_text(original_text) and len(cleaned) <= 3:
        return cleaned
    return cleaned


def _direct_short_text_fallback(region: TextRegion) -> str:
    validity = str(region.debug.get("ocr_text_validity", ""))
    original = normalize_text(region.ocr_text)
    if validity == "punctuation_only":
        return clean_translation_output(original, original)
    if validity == "short_japanese" and len(original) <= 3:
        return clean_translation_output(original, original)
    return ""


class DeepSeekBatchTranslator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session = requests.Session()
        self.last_api_requested = False
        self.last_api_failure_count = 0
        self.last_fallback_to_ocr_count = 0
        self.last_failure_reasons: dict[str, int] = {}

    def _reset_stats(self) -> None:
        self.last_api_requested = False
        self.last_api_failure_count = 0
        self.last_fallback_to_ocr_count = 0
        self.last_failure_reasons = {}

    def _record_failure_reason(self, reason: str) -> None:
        if not reason:
            return
        self.last_failure_reasons[reason] = self.last_failure_reasons.get(reason, 0) + 1

    @staticmethod
    def _classify_translation_error(message: str) -> str:
        lowered = str(message or "").lower()
        if "winerror 10013" in lowered or "failed to establish a new connection" in lowered:
            return "network_permission_denied"
        if "timeout" in lowered:
            return "api_timeout"
        if "json" in lowered or "expecting value" in lowered:
            return "response_parse_failed"
        if "401" in lowered or "403" in lowered:
            return "api_auth_failed"
        if "http" in lowered or "httpsconnectionpool" in lowered:
            return "api_request_failed"
        return "api_runtime_error"

    def _set_direct_output(
        self,
        region: TextRegion,
        *,
        backend: str,
        fallback_to_ocr: bool,
        failure_reason: str = "",
        translation_error: str = "",
    ) -> None:
        region.translation = region.ocr_text
        region.debug["translation_backend"] = backend
        region.debug["translation_fallback_to_ocr"] = fallback_to_ocr
        region.debug["translation_failure_reason"] = failure_reason
        if translation_error:
            region.debug["translation_error"] = translation_error

    def translate_regions(self, regions: list[TextRegion]) -> None:
        self._reset_stats()
        translatable = [
            region
            for region in regions
            if (
                region.region_type in {"dialogue_bubble", "narration_box"}
                and region.ocr_text
                and not bool(region.debug.get("short_text_skip", False))
            )
        ]
        if not translatable:
            return

        if not self.config.translator.enabled or os.environ.get("USE_DEEPSEEK_API", "false").lower() != "true":
            for region in translatable:
                self._set_direct_output(
                    region,
                    backend="disabled",
                    fallback_to_ocr=False,
                    failure_reason="translator_disabled",
                )
            return

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            for region in translatable:
                self._set_direct_output(
                    region,
                    backend="missing_api_key",
                    fallback_to_ocr=True,
                    failure_reason="missing_api_key",
                )
                self.last_fallback_to_ocr_count += 1
            return

        grouped: dict[str, list[TextRegion]] = defaultdict(list)
        for region in translatable:
            grouped[region.region_type].append(region)

        for region_type, group in grouped.items():
            for batch in chunked(group, max(1, self.config.translator.batch_size)):
                mapping, failure_reason, last_error = self._translate_batch(batch, region_type, api_key)
                for region in batch:
                    if failure_reason:
                        self._set_direct_output(
                            region,
                            backend="deepseek_failed_fallback_to_ocr",
                            fallback_to_ocr=True,
                            failure_reason=failure_reason,
                            translation_error=last_error,
                        )
                        self.last_fallback_to_ocr_count += 1
                        continue
                    translated = clean_translation_output(mapping.get(str(region.index), ""), region.ocr_text)
                    if not translated:
                        translated = _direct_short_text_fallback(region)
                    if not translated:
                        region.skip_reason = region.skip_reason or "translation_skip"
                        region.debug["final_skip_stage"] = "translation"
                    region.translation = translated
                    region.debug["translation_backend"] = "deepseek" if translated else "deepseek_skip"
                    region.debug["translation_fallback_to_ocr"] = False
                    region.debug["translation_failure_reason"] = ""

    def _translate_batch(
        self,
        batch: list[TextRegion],
        region_type: str,
        api_key: str,
    ) -> tuple[dict[str, str], str, str]:
        payload_items = [
            {
                "id": str(region.index),
                "text": region.ocr_text,
                "validity": region.debug.get("ocr_text_validity", ""),
                "short_text": len(normalize_text(region.ocr_text)) <= 6,
            }
            for region in batch
        ]
        payload = {
            "model": self.config.translator.model,
            "temperature": self.config.translator.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You translate Japanese manga text into concise natural Simplified Chinese. "
                        "Translate dialogue and narration only. "
                        "Never return SKIP for short dialogue-like utterances, short narration, ellipsis, "
                        "or very short emotional words inside dialogue or narration regions. "
                        "If text is punctuation only, preserve it naturally in Chinese punctuation. "
                        "Only return SKIP when an item is clearly background UI text, meaningless OCR noise, "
                        "or a non-story sound effect. "
                        "Return a strict JSON object mapping item id to translation, with no markdown. "
                        f"Current region type: {region_type}."
                    ),
                },
                {"role": "user", "content": json.dumps(payload_items, ensure_ascii=False)},
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        self.last_api_requested = True
        last_error = ""
        for attempt in range(self.config.translator.max_retries):
            try:
                response = self.session.post(
                    self.config.translator.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.translator.timeout_seconds,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                parsed = json.loads(content)
                return {str(key): str(value) for key, value in parsed.items()}, "", ""
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                time.sleep(min(4, 2**attempt))

        failure_reason = self._classify_translation_error(last_error)
        self.last_api_failure_count += 1
        self._record_failure_reason(failure_reason)
        for region in batch:
            region.debug["translation_error"] = last_error
        return {}, failure_reason, last_error

    def runtime_summary(self) -> dict[str, object]:
        api_enabled = self.config.translator.enabled and os.environ.get("USE_DEEPSEEK_API", "false").lower() == "true"
        return {
            "configured_backend": "deepseek",
            "enabled": api_enabled,
            "model": self.config.translator.model,
            "api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
            "api_requested": self.last_api_requested,
            "api_failure_count": self.last_api_failure_count,
            "fallback_to_ocr_count": self.last_fallback_to_ocr_count,
            "failure_reasons": dict(self.last_failure_reasons),
        }
