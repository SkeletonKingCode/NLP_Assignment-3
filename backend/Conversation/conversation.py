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
SESSION_TTL_SECONDS = 30 * 60   # 30-minute inactivity timeout
MAX_HISTORY_TURNS = 10           # sliding window (user+assistant pairs)

CORE_IDENTITY = (
    "You are Ali, a friendly and professional real estate assistant for a "
    "property agency based in Pakistan.\n"
    "You ONLY discuss real estate: properties, prices, visits, and agent bookings.\n"
    "STRICT RULE: ONLY offer properties from the AUTHORISED INVENTORY below. "
    "NEVER invent locations, addresses, square footage, or prices. "
    "If a subtype is not in the inventory, say it is not available.\n\n"
    "AUTHORISED INVENTORY\n"
    "====================\n"
    "SHOPS\n"
    "  - 5 Marla Shop     : PKR 1.2 Crore\n"
    "  - 8 Marla Shop     : PKR 2.1 Crore\n"
    "  - 1 Kanal Shop     : PKR 3.8 Crore\n\n"
    "HOUSES / VILLAS\n"
    "  - 5 Marla House    : PKR 1.8 Crore\n"
    "  - 7 Marla House    : PKR 2.6 Crore\n"
    "  - 10 Marla House   : PKR 4.2 Crore\n"
    "  - 1 Kanal Villa    : PKR 8.5 Crore\n\n"
    "APARTMENTS\n"
    "  - 1 Bedroom Apt    : PKR 55 Lac\n"
    "  - 2 Bedroom Apt    : PKR 95 Lac\n"
    "  - 3 Bedroom Apt    : PKR 1.5 Crore\n\n"
    "DO NOT add, modify, estimate, or paraphrase any price or property "
    "outside this list. These are the ONLY properties that exist."
)

# Stage hint injected per turn so the model always knows its current goal
STAGE_HINTS: dict[str, str] = {
    "greeting": (
        "CURRENT GOAL: Greet the customer warmly and ask which category they "
        "want — Shops, Houses/Villas, or Apartments. Do NOT list prices yet."
    ),
    "category_selection": (
        "CURRENT GOAL: The customer has shown interest in a category. "
        "List ONLY the exact subtypes and PKR prices for that category "
        "from the AUTHORISED INVENTORY above. "
        "Do NOT present subtypes from other categories. "
        "Do NOT invent new properties."
    ),
    "subtype_selection": (
        "CURRENT GOAL: The customer has selected a specific subtype. "
        "State its exact price from the AUTHORISED INVENTORY. "
        "Describe it briefly (good for family/business, size), "
        "then ask: would they like to schedule a visit or speak to an agent?"
    ),
    "closing": (
        "CURRENT GOAL: Help the customer book a property visit or connect "
        "with an agent. Be warm, confirm the chosen property, and offer "
        "clear next steps."
    ),
}

OFF_TOPIC_REMINDER = (
    "[POLICY REMINDER] The customer's last message is not about real estate. "
    "Acknowledge it briefly and warmly, then redirect to property categories, "
    "prices, visits, or agent bookings. Do NOT answer the off-topic question."
)

# ---------------------------------------------------------------------------
# Stage-transition keyword tables
# IMPORTANT: Only checked against USER messages.
#            Assistant text NEVER drives stage advancement.
# ---------------------------------------------------------------------------

# User mentions a property category → greeting → category_selection
_CATEGORY_KW: list[str] = [
    "shop", "house", "villa", "apartment", "flat",
]

# User mentions a specific size/subtype → category_selection → subtype_selection
# Must be explicit size phrases, NOT bare category words like "house"
_SUBTYPE_KW: list[str] = [
    "5 marla", "7 marla", "8 marla", "10 marla", "1 kanal",
    "1 bedroom", "2 bedroom", "3 bedroom",
    "1bed", "2bed", "3bed",
]

# User explicitly requests next action → subtype_selection → closing
_CLOSING_KW: list[str] = [
    "schedule", "book a visit", "visit", "agent", "speak to",
    "contact agent", "arrange", "i'd like to visit", "i want to visit",
]

# Broad set used by off-topic detector
REALESTATE_KEYWORDS: list[str] = [
    "shop", "house", "villa", "apartment", "flat", "property", "properties",
    "marla", "kanal", "bedroom", "price", "pkr", "crore", "lac", "lakh",
    "buy", "purchase", "rent", "visit", "agent", "booking", "schedule",
    "real estate", "plot", "area", "size", "category",
    "hello", "hi", "hey", "thanks", "thank", "bye", "goodbye",
    "yes", "no", "okay", "ok", "sure", "please", "show", "tell",
    "more", "info", "interested", "looking",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Holds all mutable state for one user conversation."""

    session_id: str
    history: list[dict] = field(default_factory=list)
    stage: str = "greeting"
    last_active: float = field(default_factory=time.time)
    # The very first assistant reply is pinned for context-window anchoring
    first_assistant_turn: Optional[dict] = None


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_sessions: dict[str, Session] = {}


def create_session() -> str:
    """Create a new session, store it, and return its UUID string."""
    sid = str(uuid.uuid4())
    _sessions[sid] = Session(session_id=sid)
    return sid


def get_session(session_id: str) -> Optional[Session]:
    """Return the Session for session_id, or None if expired / not found.

    Also purges all stale sessions as a side-effect.
    """
    _purge_expired_sessions()
    return _sessions.get(session_id)


def delete_session(session_id: str) -> None:
    """Remove a session from the store immediately."""
    _sessions.pop(session_id, None)


def _purge_expired_sessions() -> None:
    """Delete all sessions inactive for longer than SESSION_TTL_SECONDS."""
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
    """Return True when the user message has no real-estate-related keywords.

    Short messages of three words or fewer are always treated as on-topic
    so simple acknowledgements like 'yes', 'ok', 'sure' are never flagged.
    """
    words = message.strip().split()
    if len(words) <= 3:
        return False
    lower = message.lower()
    return not any(kw in lower for kw in REALESTATE_KEYWORDS)


# ---------------------------------------------------------------------------
# Stage tracking — USER-INTENT ONLY
# ---------------------------------------------------------------------------

def _advance_stage_on_user(session: Session, user_message: str) -> None:
    """Advance session.stage based solely on the USER's message keywords.

    Stages are strictly one-directional:
        greeting → category_selection → subtype_selection → closing

    Key design decision: the assistant's own reply text NEVER drives stage
    advancement. Ali routinely says words like 'visit' and 'agent' as part
    of every response — if those triggered transitions the stage would jump
    to 'closing' after the very first reply, which is exactly the bug this
    function is designed to prevent.
    """
    lower = user_message.lower()

    if session.stage == "greeting":
        if any(kw in lower for kw in _CATEGORY_KW):
            session.stage = "category_selection"

    elif session.stage == "category_selection":
        # Only an explicit size/subtype keyword (e.g. "10 marla") advances
        # the stage. Repeating a bare category word ("house") keeps the model
        # presenting the subtype menu as intended.
        if any(kw in lower for kw in _SUBTYPE_KW):
            session.stage = "subtype_selection"

    elif session.stage == "subtype_selection":
        if any(kw in lower for kw in _CLOSING_KW):
            session.stage = "closing"

    # "closing" is terminal — no further advancement


# ---------------------------------------------------------------------------
# Context window management
# ---------------------------------------------------------------------------

def _trimmed_history(session: Session) -> list[dict]:
    """Return a sliding window of at most MAX_HISTORY_TURNS turns.

    If older entries were dropped, the first assistant turn (the greeting)
    is re-inserted at the front so the model never loses its opening context.
    """
    max_entries = MAX_HISTORY_TURNS * 2   # each turn = one dict entry

    if len(session.history) <= max_entries:
        return list(session.history)

    trimmed = session.history[-max_entries:]

    if (
        session.first_assistant_turn is not None
        and trimmed[0] != session.first_assistant_turn
    ):
        trimmed = [session.first_assistant_turn] + trimmed

    return trimmed


# ---------------------------------------------------------------------------
# Prompt orchestration
# ---------------------------------------------------------------------------

def _build_system_prompt(session: Session, off_topic: bool) -> dict:
    """Build the dynamic system prompt dict sent as message[0] to Ollama.

    Combines:
    - CORE_IDENTITY  — inventory + hard rules (always present)
    - Stage hint     — one-sentence goal for the current stage
    - Policy reminder — only appended when off_topic is True
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
    """Async generator driving one complete conversational turn.

    Pipeline
    --------
    1. Validate session.
    2. Detect off-topic content.
    3. Advance stage based on USER intent ONLY (never on assistant text).
    4. Append user turn to history.
    5. Build: [dynamic_system_prompt] + [trimmed_history].
    6. Stream Ollama response; yield each content token.
    7. Append complete assistant turn; pin greeting anchor if needed.

    Yields
    ------
    str
        Individual content tokens from the model, or a single [ERROR] string.
    """
    session = get_session(session_id)
    if session is None:
        yield "[ERROR] Session not found or expired. Please start a new session."
        return

    session.last_active = time.time()

    # ── 1. Policy / off-topic check ──────────────────────────────────────────
    off_topic = _is_off_topic(user_message)

    # ── 2. Stage advancement on USER message ─────────────────────────────────
    _advance_stage_on_user(session, user_message)

    # ── 3. Append user turn BEFORE slicing history ───────────────────────────
    session.history.append({"role": "user", "content": user_message})

    # ── 4. Build final Ollama message list ───────────────────────────────────
    system_msg = _build_system_prompt(session, off_topic)
    messages = [system_msg] + _trimmed_history(session)

    # ── 5. Stream ─────────────────────────────────────────────────────────────
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
        yield f"\n[ERROR] Ollama ResponseError: {exc.error}"
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        return

    except Exception as exc:  # noqa: BLE001
        yield f"\n[ERROR] Could not reach Ollama: {exc}"
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        return

    # ── 6. Persist assistant turn ─────────────────────────────────────────────
    assistant_turn = {"role": "assistant", "content": "".join(full_response)}
    session.history.append(assistant_turn)

    # Pin the first assistant reply for context-window anchor re-insertion
    if session.first_assistant_turn is None:
        session.first_assistant_turn = assistant_turn

    session.last_active = time.time()


# ---------------------------------------------------------------------------
# Multi-turn smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    TEST_TURNS: list[tuple[str, str]] = [
        ("Turn 1", "Hi"),
        ("Turn 2", "I want to buy a house"),
        ("Turn 3", "Show me 10 marla options"),
        ("Turn 4", "What's the weather today?"),    # ← off-topic test
        ("Turn 5", "I'd like to schedule a visit"),
        ("Turn 6", "Thanks, goodbye"),
    ]

    async def run_test() -> None:
        """Simulate the 6-turn test dialogue on a single session."""
        sid = create_session()
        print("\n=== Ali Real Estate Chatbot — Smoke Test ===")
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