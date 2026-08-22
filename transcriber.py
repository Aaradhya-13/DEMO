"""
edge_nlp/transcriber.py
-------------------------
Offline, on-device speech recognition using `faster-whisper` (CTranslate2
backend), with dynamic device fallback: CUDA -> CPU/INT8. Handles automatic
language detection across Indic dialects (or any language faster-whisper
supports) without hardcoding a language list — whatever the model detects
or the caller supplies is used as-is.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration_seconds: float


class TranscriberInitError(RuntimeError):
    """Raised when no compute backend (CUDA or CPU) can load the model."""


class WhisperTranscriber:
    """Lazy-loaded, device-adaptive faster-whisper wrapper."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        self.model_size = model_size or settings.whisper_model_size
        self.requested_device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self._model = None
        self._active_device: Optional[str] = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        candidate_devices: list[tuple[str, str]]
        if self.requested_device == "auto":
            candidate_devices = [
                ("cuda", "float16"),
                ("cpu", self.compute_type),
            ]
        else:
            candidate_devices = [(self.requested_device, self.compute_type)]

        last_error: Optional[Exception] = None
        for device, compute_type in candidate_devices:
            try:
                logger.info(
                    "Loading faster-whisper model",
                    extra={
                        "context": {
                            "model_size": self.model_size,
                            "device": device,
                            "compute_type": compute_type,
                        }
                    },
                )
                self._model = WhisperModel(
                    self.model_size, device=device, compute_type=compute_type
                )
                self._active_device = device
                return self._model
            except Exception as exc:  # noqa: BLE001 - intentional fallback chain
                logger.warning(
                    "Failed to load whisper model on device",
                    extra={"context": {"device": device, "error": str(exc)}},
                )
                last_error = exc

        raise TranscriberInitError(
            f"Unable to load faster-whisper model '{self.model_size}' on any candidate "
            f"device {candidate_devices}: {last_error}"
        )

    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe raw WAV bytes. If `language` is None, faster-whisper's
        automatic language identification is used (covers Indic dialects
        such as Assamese, Bengali, Hindi, etc. among the languages the
        underlying Whisper model supports).
        """
        model = self._load_model()
        lang_hint = language or settings.whisper_default_language

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            segments, info = model.transcribe(
                tmp.name,
                language=lang_hint,
                vad_filter=True,
                beam_size=5,
            )
            text_parts = [seg.text.strip() for seg in segments]

        full_text = " ".join(part for part in text_parts if part).strip()
        result = TranscriptionResult(
            text=full_text,
            language=info.language,
            language_probability=float(info.language_probability),
            duration_seconds=float(info.duration),
        )
        logger.info(
            "Transcription complete",
            extra={
                "context": {
                    "device": self._active_device,
                    "language": result.language,
                    "language_probability": result.language_probability,
                    "chars": len(result.text),
                }
            },
        )
        return result

    def transcribe_file(self, path: str | Path, language: Optional[str] = None) -> TranscriptionResult:
        with open(path, "rb") as f:
            return self.transcribe(f.read(), language=language)
