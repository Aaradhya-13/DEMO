"""
edge_nlp/audio_recorder.py
----------------------------
Dynamic microphone capture via `sounddevice`, with a `PyAudio` fallback if
`sounddevice`/PortAudio isn't available on the host. Nothing about the
recording duration, sample rate, channel count, or device index is
hardcoded — all pulled from `core.config.settings` or passed by the caller.
"""

from __future__ import annotations

import io
import wave
from typing import Optional

import numpy as np

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class AudioRecorderError(RuntimeError):
    """Raised when no working audio backend/device can be found."""


class AudioRecorder:
    """Records microphone audio to an in-memory 16-bit PCM WAV buffer."""

    def __init__(
        self,
        sample_rate_hz: Optional[int] = None,
        channels: Optional[int] = None,
        device_index: Optional[int] = None,
    ):
        self.sample_rate_hz = sample_rate_hz or settings.audio_sample_rate_hz
        self.channels = channels or settings.audio_channels
        self.device_index = device_index
        self._backend = self._detect_backend()
        logger.info(
            "AudioRecorder initialized",
            extra={"context": {"backend": self._backend, "sample_rate_hz": self.sample_rate_hz}},
        )

    @staticmethod
    def _detect_backend() -> str:
        try:
            import sounddevice  # noqa: F401
            return "sounddevice"
        except Exception:
            pass
        try:
            import pyaudio  # noqa: F401
            return "pyaudio"
        except Exception:
            pass
        raise AudioRecorderError(
            "Neither 'sounddevice' nor 'pyaudio' is available/functional on this host. "
            "Install one of them, or supply pre-recorded WAV bytes directly to the transcriber."
        )

    def list_devices(self) -> list[dict]:
        """Enumerate available input devices for the active backend."""
        if self._backend == "sounddevice":
            import sounddevice as sd
            devices = sd.query_devices()
            return [
                {"index": i, "name": d["name"], "max_input_channels": d["max_input_channels"]}
                for i, d in enumerate(devices)
                if d["max_input_channels"] > 0
            ]
        else:
            import pyaudio
            pa = pyaudio.PyAudio()
            try:
                result = []
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    if info.get("maxInputChannels", 0) > 0:
                        result.append(
                            {
                                "index": i,
                                "name": info.get("name"),
                                "max_input_channels": info.get("maxInputChannels"),
                            }
                        )
                return result
            finally:
                pa.terminate()

    def record(self, duration_seconds: Optional[float] = None) -> bytes:
        """
        Block and record `duration_seconds` (default: settings.audio_max_record_seconds)
        of audio from the microphone. Returns a complete WAV file as bytes.
        """
        duration = duration_seconds or settings.audio_max_record_seconds
        if duration <= 0:
            raise ValueError("duration_seconds must be positive")

        if self._backend == "sounddevice":
            pcm = self._record_sounddevice(duration)
        else:
            pcm = self._record_pyaudio(duration)

        return self._pcm_to_wav_bytes(pcm)

    def _record_sounddevice(self, duration: float) -> np.ndarray:
        import sounddevice as sd

        num_frames = int(duration * self.sample_rate_hz)
        logger.info(
            "Recording via sounddevice",
            extra={"context": {"duration_seconds": duration, "device_index": self.device_index}},
        )
        recording = sd.rec(
            num_frames,
            samplerate=self.sample_rate_hz,
            channels=self.channels,
            dtype="int16",
            device=self.device_index,
        )
        sd.wait()
        return recording

    def _record_pyaudio(self, duration: float) -> np.ndarray:
        import pyaudio

        pa = pyaudio.PyAudio()
        chunk = 1024
        frames = []
        stream = None
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate_hz,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=chunk,
            )
            logger.info(
                "Recording via pyaudio",
                extra={"context": {"duration_seconds": duration, "device_index": self.device_index}},
            )
            total_chunks = int(self.sample_rate_hz / chunk * duration)
            for _ in range(total_chunks):
                frames.append(stream.read(chunk, exception_on_overflow=False))
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            pa.terminate()

        raw = b"".join(frames)
        pcm = np.frombuffer(raw, dtype=np.int16)
        if self.channels > 1:
            pcm = pcm.reshape(-1, self.channels)
        return pcm

    def _pcm_to_wav_bytes(self, pcm: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.sample_rate_hz)
            wf.writeframes(np.ascontiguousarray(pcm).tobytes())
        return buffer.getvalue()
