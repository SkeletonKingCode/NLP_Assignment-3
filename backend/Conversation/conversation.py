"""
backend/Conversation/conversation.py

Conversation manager for Ali — a Pakistani real estate assistant chatbot.
Handles session management, context window trimming, stage tracking,
off-topic policy enforcement, and streaming Ollama integration.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import ollama

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "ali-realestate"
SESSION_TTL_SECONDS = 30 * 60          # 30-minute inactivity timeout
MAX_HISTORY_TURNS = 10                  # sliding window (user+assistant pairs)

CORE_IDENTITY = (
    "You are Ali, a friendly and professional real estate assistant for a "
    "property agency based in Pakistan. You only discuss real estate topics: "
    "properties, prices, visits, and agent bookings. "
    "Never discuss anything outside real estate."
)

# Stage hint templates injected into the dynamic system prompt
STAGE_HINTS: dict[str, str] = {
    "greeting": (
        "You are at the GREETING stage. Warmly welcome the customer and ask "
        "which type of property they are interested in: Shops, Houses/Villas, "
        "or Apartments."
    ),
    "category_selection": (
        "The customer is choosing a property CATEGORY. "
        "Present the three categories (Shops, Houses/Villas, Apartments) "
        "and invite them to pick one."
    ),
    "subtype_selection": (
        "The customer has selected a category. "
        "Present the relevant subtypes with their PKR prices clearly, "
        "then ask if they would like to schedule a visit or speak to an agent."
    ),
    "closing": (
        "The customer is ready to close. "
        "Offer to schedule a property visit or connect them with a human agent. "
        "Be warm and encouraging."
    ),
}

OFF_TOPIC_REMINDER = (
    "[POLICY] The user's last message appears unrelated to real estate. "
    "Politely acknowledge it, then redirect the conversation back to "
    "property categories, prices, visits, or agent bookings."
)

# Keywords that indicate real-estate relevance (broad set)
REALESTATE_KEYWORDS: list[str] = [
    "shop", "house", "villa", "apartment", "flat", "property", "properties",
    "marla", "kanal", "bedroom", "price", "pkr", "crore", "lac", "lakh",
    "buy", "purchase", "rent", "visit", "agent", "booking", "schedule",
    "real estate", "floor", "plot", "area", "location", "size", "category",
    "hello", "hi", "hey", "thanks", "thank", "bye", "goodbye", "yes", "no",
    "okay", "ok", "sure", "please", "show", "tell", "more", "info",
]

# Keywords that advance the stage machine
CATEGORY_KEYWORDS: list[str] = [
    "shop", "house", "villa", "apartment", "flat",
]
SUBTYPE_KEYWORDS: list[str] = [
    "marla", "kanal", "bedroom", "1 bed", "2 bed", "3 bed",
    "5 marla", "7 marla", "8 marla", "10 marla", "1 kanal",
]
CLOSING_KEYWORDS: list[str] = [
    "visit", "schedule", "book", "agent", "speak", "contact",
    "thanks", "thank", "bye", "goodbye", "done", "ok", "sure",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Holds all state for a single user conversation."""

    session_id: str
    history: list[dict] = field(default_factory=list)
    stage: str = "greeting"
    last_active: float = field(default_factory=time.time)
    # Stores the very first assistant message for permanent context anchoring
    first_assistant_turn: Optional[dict] = None


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_sessions: dict[str, Session] = {}


def create_session() -> str:
    """Create a new session and return its unique session_id."""
    sid = str(uuid.uuid4())
    _sessions[sid] = Session(session_id=sid)
    return sid


def get_session(session_id: str) -> Optional[Session]:
    """Return the Session for session_id, or None if expired / not found."""
    _purge_expired_sessions()
    return _sessions.get(session_id)


def delete_session(session_id: str) -> None:
    """Remove a session from the store."""
    _sessions.pop(session_id, None)


def _purge_expired_sessions() -> None:
    """Remove all sessions that have been inactive beyond SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid for sid, s in _sessions.items()
        if now - s.last_active > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]


# ---------------------------------------------------------------------------
# Off-topic detection
# ---------------------------------------------------------------------------

def _is_off_topic(message: str) -> bool:
    """
    Return True when the user message contains no real-estate-related keywords.

    Uses a simple keyword scan; no external NLP libraries required.
    Very short acknowledgements (≤ 3 words) are considered on-topic so the
    model can handle them naturally.
    """
    words = message.strip().split()
    if len(words) <= 3:
        return False
    lower = message.lower()
    return not any(kw in lower for kw in REALESTATE_KEYWORDS)


# ---------------------------------------------------------------------------
# Stage tracking
# ---------------------------------------------------------------------------

def _advance_stage(session: Session, text: str) -> None:
    """
    Inspect text (user or assistant) and advance session.stage if appropriate.

    Stages flow in one direction only:
        greeting → category_selection → subtype_selection → closing
    """
    lower = text.lower()

    if session.stage == "greeting":
        if any(kw in lower for kw in CATEGORY_KEYWORDS):
            session.stage = "category_selection"

    elif session.stage == "category_selection":
        if any(kw in lower for kw in SUBTYPE_KEYWORDS + CATEGORY_KEYWORDS):
            session.stage = "subtype_selection"

    elif session.stage == "subtype_selection":
        if any(kw in lower for kw in CLOSING_KEYWORDS):
            session.stage = "closing"

    # "closing" is terminal — no further advancement


# ---------------------------------------------------------------------------
# Context window management
# ---------------------------------------------------------------------------

def _trimmed_history(session: Session) -> list[dict]:
    """
    Return a sliding window of at most MAX_HISTORY_TURNS turns from history.

    The very first assistant turn (the greeting) is always prepended so the
    model retains its opening context even after trimming.
    """
    history = session.history

    # Each "turn" is one dict entry; pairs = user + assistant
    max_entries = MAX_HISTORY_TURNS * 2
    if len(history) > max_entries:
        trimmed = history[-max_entries:]
    else:
        trimmed = list(history)

    # Prepend the preserved greeting if it exists and isn't already first
    if session.first_assistant_turn and (
        not trimmed or trimmed[0] != session.first_assistant_turn
    ):
        trimmed = [session.first_assistant_turn] + trimmed

    return trimmed


# ---------------------------------------------------------------------------
# Prompt orchestration
# ---------------------------------------------------------------------------

def _build_system_prompt(session: Session, off_topic: bool) -> dict:
    """
    Construct the dynamic system prompt dict sent as the first message to Ollama.

    Includes:
    - Core identity
    - Stage-aware hint
    - Optional off-topic policy reminder
    """
    stage_hint = STAGE_HINTS.get(session.stage, "")
    parts = [CORE_IDENTITY, stage_hint]

    if off_topic:
        parts.append(OFF_TOPIC_REMINDER)

    return {"role": "system", "content": "\n\n".join(parts)}


# ---------------------------------------------------------------------------
# Ollama streaming integration
# ---------------------------------------------------------------------------

async def stream_response(
    session_id: str, user_message: str
) -> AsyncGenerator[str, None]:
    """
    Core async generator that drives a single conversational turn.

    Steps:
    1.  Retrieve (or create) the session.
    2.  Detect off-topic content.
    3.  Advance the conversation stage based on the user message.
    4.  Append the user turn to history.
    5.  Build the full message list: [system] + [trimmed_history].
    6.  Stream the Ollama response, yielding tokens as they arrive.
    7.  Accumulate the full response text, append it to history,
        and cache the first assistant turn if not yet saved.

    Yields:
        Individual string tokens from the model, or an error message string.
    """
    session = get_session(session_id)
    if session is None:
        yield "[ERROR] Session not found or expired. Please start a new session."
        return

    session.last_active = time.time()

    # --- Policy check ---
    off_topic = _is_off_topic(user_message)

    # --- Stage advancement on user input ---
    _advance_stage(session, user_message)

    # --- Append user turn to history ---
    session.history.append({"role": "user", "content": user_message})

    # --- Build message list ---
    system_prompt = _build_system_prompt(session, off_topic)
    messages = [system_prompt] + _trimmed_history(session)

    # --- Stream from Ollama ---
    client = ollama.AsyncClient()
    full_response: list[str] = []

    try:
        async for chunk in await client.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            think=False, 
        ):
            token: str = chunk.message.content or ""
            if token:
                full_response.append(token)
                yield token

    except ollama.ResponseError as exc:
        error_msg = f"\n[ERROR] Ollama ResponseError: {exc.error}"
        yield error_msg
        # Remove the user turn we already appended so history stays clean
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        return

    except Exception as exc:  # noqa: BLE001 — connection / IO errors
        error_msg = f"\n[ERROR] Could not reach Ollama: {exc}"
        yield error_msg
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        return

    # --- Persist assistant turn ---
    assistant_text = "".join(full_response)
    assistant_turn = {"role": "assistant", "content": assistant_text}
    session.history.append(assistant_turn)

    # Cache the very first assistant response (greeting) for context anchoring
    if session.first_assistant_turn is None:
        session.first_assistant_turn = assistant_turn

    # Advance stage on assistant reply too (catches cases where the model
    # proactively moves the conversation forward)
    _advance_stage(session, assistant_text)
    session.last_active = time.time()


# ---------------------------------------------------------------------------
# Multi-turn smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    TEST_TURNS: list[tuple[str, str]] = [
        ("Turn 1", "Hi"),
        ("Turn 2", "I want to buy a house"),
        ("Turn 3", "Show me 10 marla options"),
        ("Turn 4", "What's the weather today?"),   # off-topic
        ("Turn 5", "I'd like to schedule a visit"),
        ("Turn 6", "Thanks, goodbye"),
    ]

    async def run_test() -> None:
        """Simulate a 6-turn dialogue on a single session and print results."""
        sid = create_session()
        print(f"\n=== Ali Real Estate Chatbot — Smoke Test ===")
        print(f"Session ID: {sid}\n")

        for label, user_msg in TEST_TURNS:
            session = get_session(sid)
            stage_before = session.stage if session else "unknown"

            print(f"[{label}] User  : {user_msg}")
            print(f"          Stage : {stage_before}")
            print(f"          Ali   : ", end="", flush=True)

            async for token in stream_response(sid, user_msg):
                print(token, end="", flush=True)

            session = get_session(sid)
            stage_after = session.stage if session else "unknown"
            print(f"\n          Stage → {stage_after}\n")

        print("=== Test Complete ===\n")

    asyncio.run(run_test())