"""
backend/api/main.py

FastAPI server for the Ali Real Estate chatbot — with voice I/O.

Endpoints
---------
POST   /session              — create a new session
GET    /session/{id}         — get session state summary
DELETE /session/{id}         — delete a session
POST   /transcribe           — transcribe uploaded audio → {"text": "..."}
WS     /ws/chat              — streaming chat over WebSocket

WebSocket protocol
------------------
Client  → server:
    {"session_id": "<uuid>", "message": "<text>", "voice": true|false}

Server  → client:
    {"type": "token",       "data": "<token>"}       streamed LLM output
    {"type": "audio_chunk", "data": "<base64_wav>"}  one sentence of TTS audio
    {"type": "done",        "data": ""}              end of turn
    {"type": "state",       "data": {session info}}  after done
    {"type": "error",       "data": "<message>"}     on failure
    {"type": "session_created", "data": "<id>"}      if session was auto-created
"""

import asyncio
import base64
import json
import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _find_backend_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for _ in range(4):
        if (candidate / "Conversation" / "conversation.py").exists():
            return candidate
        candidate = candidate.parent
    return Path(__file__).resolve().parent


_BACKEND_ROOT = _find_backend_root()
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_PROJECT_ROOT = _BACKEND_ROOT.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"

from Conversation.conversation import (   # noqa: E402
    create_session,
    get_session,
    delete_session,
    get_session_info,
    stream_response,
)
from Voice import asr, tts                # noqa: E402

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Ali Real Estate API starting up — preloading voice models…")
    # Preload both models in a thread so the event loop stays unblocked.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, asr.preload)
    await loop.run_in_executor(None, tts.preload)
    logger.info("Voice models ready. Server is up.")
    yield
    logger.info("Ali Real Estate API shutting down.")


app = FastAPI(
    title="Ali Real Estate Chatbot API",
    description="Local conversational AI for Pakistani property sales — with voice.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _FRONTEND_DIR.is_dir():
    app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")


@app.get("/", include_in_schema=False)
async def root():
    index = _FRONTEND_DIR / "index.html"
    if index.is_file():
        return FileResponse(str(index))
    return {"message": "Ali Real Estate API", "docs": "/docs", "health": "/health"}


# ---------------------------------------------------------------------------
# Active WebSocket tracker
# ---------------------------------------------------------------------------

_active_connections: set[WebSocket] = set()

# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

class SessionResponse(BaseModel):
    session_id: str
    message: str


@app.post("/session", response_model=SessionResponse, status_code=201)
async def create_new_session():
    sid = create_session()
    return SessionResponse(session_id=sid, message="Session created successfully.")


@app.get("/session/{session_id}")
async def get_session_state(session_id: str):
    info = get_session_info(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return info


@app.delete("/session/{session_id}", status_code=200)
async def end_session(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    delete_session(session_id)
    return {"message": "Session deleted.", "session_id": session_id}


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "active_connections": len(_active_connections),
        "tts_available": tts.is_available(),
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Transcription endpoint  (POST /transcribe)
# ---------------------------------------------------------------------------

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Accept an audio file upload and return the transcribed text.

    The browser sends a WebM/Opus blob from MediaRecorder.
    faster-whisper handles the format via ffmpeg automatically.

    Returns
    -------
    {"text": "<transcribed string>"}
    """
    # Write the upload to a temporary file so faster-whisper can read it
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await audio.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, asr.transcribe, tmp_path)
    finally:
        os.unlink(tmp_path)

    return {"text": text}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Real-time streaming chat with optional sentence-level TTS audio.

    Client payload
    --------------
    {
        "session_id": "<uuid>",
        "message":    "<user text>",
        "voice":      true          ← optional, default false
    }

    When voice=true the server streams audio_chunk frames (base64 WAV)
    sentence-by-sentence interleaved with token frames so the browser can
    start playing before the LLM has finished generating.
    """
    await websocket.accept()
    _active_connections.add(websocket)
    loop = asyncio.get_event_loop()

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, "error", "Invalid JSON payload.")
                continue

            session_id:   str  = payload.get("session_id", "")
            user_message: str  = payload.get("message", "").strip()
            voice_enabled: bool = bool(payload.get("voice", False))

            if not session_id or get_session(session_id) is None:
                session_id = create_session()
                await _send(websocket, "session_created", session_id)

            if not user_message:
                await _send(websocket, "error", "Empty message received.")
                continue

            # ── Stream LLM tokens + optional TTS ──────────────────────────
            sentence_buf = ""      # accumulates tokens until a sentence ends

            try:
                async for token in stream_response(session_id, user_message):
                    if token.startswith("[ERROR]"):
                        await _send(websocket, "error", token)
                        break

                    # Always send the text token first so UI updates instantly
                    await _send(websocket, "token", token)

                    if voice_enabled and tts.is_available():
                        sentence_buf += token
                        # Flush when we hit a sentence-ending punctuation mark.
                        # We wait for trailing whitespace so "Dr." doesn't trigger.
                        stripped = sentence_buf.rstrip()
                        if stripped and stripped[-1] in ".!?" and (
                            sentence_buf != stripped         # trailing space exists
                            or sentence_buf.endswith("\n")
                        ):
                            to_speak = stripped
                            sentence_buf = ""
                            if to_speak:
                                audio_bytes = await loop.run_in_executor(
                                    None, tts.synthesize, to_speak
                                )
                                b64 = base64.b64encode(audio_bytes).decode()
                                await _send(websocket, "audio_chunk", b64)

            except Exception as exc:
                await _send(websocket, "error", f"Streaming error: {exc}")
                continue

            # Flush any remaining text that didn't end with punctuation
            if voice_enabled and tts.is_available() and sentence_buf.strip():
                try:
                    audio_bytes = await loop.run_in_executor(
                        None, tts.synthesize, sentence_buf.strip()
                    )
                    b64 = base64.b64encode(audio_bytes).decode()
                    await _send(websocket, "audio_chunk", b64)
                except Exception as exc:
                    logger.warning("TTS flush error: %s", exc)

            await _send(websocket, "done", "")
            state = get_session_info(session_id)
            if state:
                await websocket.send_text(json.dumps({"type": "state", "data": state}))

    finally:
        _active_connections.discard(websocket)


async def _send(ws: WebSocket, msg_type: str, data: str) -> None:
    try:
        await ws.send_text(json.dumps({"type": msg_type, "data": data}))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
