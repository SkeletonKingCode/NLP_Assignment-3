"""
tests/conftest.py

Shared fixtures for the Ali Real Estate Chatbot test suite.
Patches heavy dependencies (Ollama, ASR, TTS) so tests run
without any external services.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure backend modules are importable
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ---------------------------------------------------------------------------
# Stub out ALL heavy external dependencies BEFORE anything imports them.
# Order matters: ollama must be in sys.modules before conversation.py loads.
# ---------------------------------------------------------------------------

# 1) ollama — used by Conversation.conversation
_ollama_stub = MagicMock()
_ollama_stub.AsyncClient = MagicMock
_ollama_stub.ResponseError = type("ResponseError", (Exception,), {"error": ""})
sys.modules.setdefault("ollama", _ollama_stub)

# 2) faster_whisper — used by Voice.asr
sys.modules.setdefault("faster_whisper", MagicMock())

# 3) piper / piper.voice — used by Voice.tts
sys.modules.setdefault("piper", MagicMock())
sys.modules.setdefault("piper.voice", MagicMock())

# 4) Voice module stubs — so api.main can do `from Voice import asr, tts`
_asr_stub = MagicMock()
_asr_stub.preload = MagicMock()
_asr_stub.transcribe = MagicMock(return_value="hello world")

_tts_stub = MagicMock()
_tts_stub.preload = MagicMock()
_tts_stub.is_available = MagicMock(return_value=False)
_tts_stub.synthesize = MagicMock(return_value=b"RIFF----WAVEfmt ")

sys.modules["Voice"] = MagicMock(asr=_asr_stub, tts=_tts_stub)
sys.modules["Voice.asr"] = _asr_stub
sys.modules["Voice.tts"] = _tts_stub


@pytest.fixture()
def clear_sessions():
    """Wipe the in-memory session store before each test."""
    from Conversation.conversation import _sessions
    _sessions.clear()
    yield
    _sessions.clear()
