"""
backend/Voice/asr.py

Automatic Speech Recognition using faster-whisper.
Loads the model once at module level (lazy, thread-safe).
Exposes a single blocking function: transcribe(audio_path) -> str
"""

import os
import threading
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE     = os.environ.get("WHISPER_DEVICE",     "cpu")
WHISPER_COMPUTE    = os.environ.get("WHISPER_COMPUTE",    "int8")

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_model      = None
_model_lock = threading.Lock()


def _get_model():
    """Return the WhisperModel, loading it on first call."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel
                logger.info(
                    "Loading Whisper model '%s' on %s (%s)…",
                    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE,
                )
                _model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE,
                )
                logger.info("Whisper model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preload() -> None:
    """Call at startup to avoid cold-start latency on the first request."""
    _get_model()


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file and return the recognised text.

    Parameters
    ----------
    audio_path : str
        Path to any audio format supported by ffmpeg (webm, mp3, wav, ogg…).

    Returns
    -------
    str
        Concatenated transcript, stripped of leading/trailing whitespace.
        Returns an empty string if nothing was detected.
    """
    model = _get_model()
    segments, _info = model.transcribe(
        audio_path,
        beam_size=5,
        language="en",      # force English; remove for auto-detect
        vad_filter=True,    # skip silent segments
    )
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()
