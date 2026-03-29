"""
backend/Voice/tts.py

Text-to-Speech using piper-tts.
Loads the voice model once (lazy, thread-safe).
Exposes a single blocking function: synthesize(text) -> bytes  (WAV)

Model setup
-----------
Download the model files before starting the server:

    MODEL_DIR=backend/Voice/models
    mkdir -p $MODEL_DIR
    BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium
    curl -L "$BASE/en_US-lessac-medium.onnx"      -o "$MODEL_DIR/en_US-lessac-medium.onnx"
    curl -L "$BASE/en_US-lessac-medium.onnx.json" -o "$MODEL_DIR/en_US-lessac-medium.onnx.json"

Override model path via the PIPER_MODEL_PATH environment variable.
"""

import io
import os
import threading
import wave
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "models", "en_US-lessac-medium.onnx"
)
PIPER_MODEL_PATH = os.environ.get("PIPER_MODEL_PATH", _DEFAULT_MODEL)

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_voice      = None
_voice_lock = threading.Lock()
_available  = None   # cached availability flag


def is_available() -> bool:
    """Return True if piper and its model file are both present."""
    global _available
    if _available is None:
        try:
            import piper  # noqa: F401
            _available = os.path.isfile(PIPER_MODEL_PATH)
            if not _available:
                logger.warning(
                    "Piper model not found at '%s'. TTS disabled. "
                    "See backend/Voice/tts.py docstring for download instructions.",
                    PIPER_MODEL_PATH,
                )
        except ImportError:
            _available = False
            logger.warning("piper-tts not installed. TTS disabled.")
    return _available


def _get_voice():
    """Return the PiperVoice instance, loading it on first call."""
    global _voice
    if _voice is None:
        with _voice_lock:
            if _voice is None:
                from piper.voice import PiperVoice
                logger.info("Loading Piper voice from '%s'…", PIPER_MODEL_PATH)
                _voice = PiperVoice.load(PIPER_MODEL_PATH, use_cuda=False)
                logger.info("Piper voice loaded.")
    return _voice


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preload() -> None:
    """Call at startup to warm up the model (no-op if unavailable)."""
    if is_available():
        _get_voice()


def synthesize(text: str) -> bytes:
    """Convert text to WAV audio bytes.

    Parameters
    ----------
    text : str
        The sentence or phrase to speak.

    Returns
    -------
    bytes
        A valid WAV file in memory.

    Raises
    ------
    RuntimeError
        If piper is not installed or the model file is missing.
    """
    if not is_available():
        raise RuntimeError(
            "Piper TTS is not available. "
            "Install piper-tts and download the model file."
        )

    voice   = _get_voice()
    buf     = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize(text, wav_file)
    return buf.getvalue()
