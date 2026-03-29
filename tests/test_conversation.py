"""
tests/test_conversation.py

Unit tests for the conversation manager — session CRUD, stage advancement,
off-topic detection, context window trimming, and prompt building.
All tests are pure-Python and do NOT require Ollama.
"""

import sys
import time
from pathlib import Path

import pytest

# Ensure the backend package is on the path
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from Conversation.conversation import (
    Session,
    create_session,
    get_session,
    delete_session,
    get_session_info,
    _advance_stage_on_user,
    _is_off_topic,
    _trimmed_history,
    _build_system_prompt,
    _build_conversation_state,
    _sessions,
    SESSION_TTL_SECONDS,
    MAX_HISTORY_TURNS,
    INVENTORY,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fresh():
    """Create a fresh session and return (session_id, session)."""
    _sessions.clear()
    sid = create_session()
    return sid, _sessions[sid]


# ── Session CRUD ─────────────────────────────────────────────────────────────

class TestSessionCRUD:
    """Tests for create / get / delete / info operations."""

    def test_create_session(self):
        sid, session = _fresh()
        assert isinstance(sid, str) and len(sid) == 36  # UUID format
        assert session.stage == "greeting"
        assert session.selected_category is None
        assert session.history == []

    def test_get_session_valid(self):
        sid, _ = _fresh()
        assert get_session(sid) is not None

    def test_get_session_not_found(self):
        _sessions.clear()
        assert get_session("nonexistent-id") is None

    def test_get_session_expired(self):
        sid, session = _fresh()
        session.last_active = time.time() - SESSION_TTL_SECONDS - 1
        assert get_session(sid) is None

    def test_delete_session(self):
        sid, _ = _fresh()
        delete_session(sid)
        assert sid not in _sessions

    def test_get_session_info_structure(self):
        sid, _ = _fresh()
        info = get_session_info(sid)
        assert info is not None
        assert set(info.keys()) == {
            "session_id", "stage", "selected_category",
            "selected_subtype", "selected_price", "turn_count",
        }
        assert info["stage"] == "greeting"
        assert info["turn_count"] == 0


# ── Stage Advancement ────────────────────────────────────────────────────────

class TestStageAdvancement:
    """Tests the deterministic stage machine driven by user messages."""

    def test_greeting_to_category_house(self):
        _, s = _fresh()
        _advance_stage_on_user(s, "I want to buy a house")
        assert s.stage == "category_selection"
        assert s.selected_category == "Houses/Villas"

    def test_greeting_to_category_shop(self):
        _, s = _fresh()
        _advance_stage_on_user(s, "I need a shop")
        assert s.stage == "category_selection"
        assert s.selected_category == "Shops"

    def test_greeting_to_category_apartment(self):
        _, s = _fresh()
        _advance_stage_on_user(s, "Show me apartments")
        assert s.stage == "category_selection"
        assert s.selected_category == "Apartments"

    def test_greeting_no_keyword_stays(self):
        _, s = _fresh()
        _advance_stage_on_user(s, "Hello, how are you?")
        assert s.stage == "greeting"

    def test_category_to_subtype(self):
        _, s = _fresh()
        s.stage = "category_selection"
        s.selected_category = "Houses/Villas"
        _advance_stage_on_user(s, "I want the 10 marla option")
        assert s.stage == "subtype_selection"
        assert s.selected_subtype == "10 Marla House"
        assert s.selected_price == "PKR 4.2 Crore"

    def test_category_to_subtype_apartment(self):
        _, s = _fresh()
        s.stage = "category_selection"
        s.selected_category = "Apartments"
        _advance_stage_on_user(s, "I want a 2 bedroom")
        assert s.stage == "subtype_selection"
        assert s.selected_subtype == "2 Bedroom Apt"
        assert s.selected_price == "PKR 95 Lac"

    def test_subtype_to_closing(self):
        _, s = _fresh()
        s.stage = "subtype_selection"
        _advance_stage_on_user(s, "I'd like to schedule a visit")
        assert s.stage == "closing"

    def test_closing_is_terminal(self):
        _, s = _fresh()
        s.stage = "closing"
        _advance_stage_on_user(s, "I want a house")
        assert s.stage == "closing"  # should not change


# ── Off-Topic Detection ──────────────────────────────────────────────────────

class TestOffTopicDetection:
    """Tests the keyword-based off-topic detector."""

    def test_real_estate_is_on_topic(self):
        assert _is_off_topic("I want to buy a house in Lahore") is False

    def test_weather_is_off_topic(self):
        assert _is_off_topic("What is the weather today in Islamabad?") is True

    def test_short_messages_always_on_topic(self):
        assert _is_off_topic("yes") is False
        assert _is_off_topic("ok sure") is False
        assert _is_off_topic("no thanks") is False

    def test_greeting_is_on_topic(self):
        assert _is_off_topic("hi there") is False


# ── Context Window Trimming ──────────────────────────────────────────────────

class TestContextWindow:
    """Tests the sliding window that keeps history bounded."""

    def test_under_limit_unchanged(self):
        _, s = _fresh()
        for i in range(4):
            s.history.append({"role": "user", "content": f"msg {i}"})
            s.history.append({"role": "assistant", "content": f"reply {i}"})
        trimmed = _trimmed_history(s)
        assert len(trimmed) == 8

    def test_over_limit_trimmed(self):
        _, s = _fresh()
        for i in range(20):
            s.history.append({"role": "user", "content": f"msg {i}"})
            s.history.append({"role": "assistant", "content": f"reply {i}"})
        trimmed = _trimmed_history(s)
        assert len(trimmed) == MAX_HISTORY_TURNS * 2
        # Most recent turn should be the last one added
        assert trimmed[-1]["content"] == "reply 19"

    def test_original_history_unmodified(self):
        _, s = _fresh()
        for i in range(20):
            s.history.append({"role": "user", "content": f"msg {i}"})
            s.history.append({"role": "assistant", "content": f"reply {i}"})
        original_len = len(s.history)
        _trimmed_history(s)
        assert len(s.history) == original_len  # no mutation


# ── System Prompt Building ───────────────────────────────────────────────────

class TestSystemPrompt:
    """Tests that the dynamically assembled system prompt is structured correctly."""

    def test_prompt_contains_state_block(self):
        _, s = _fresh()
        s.selected_category = "Houses/Villas"
        state_str = _build_conversation_state(s)
        assert "Stage" in state_str
        assert "Houses/Villas" in state_str

    def test_prompt_role_is_system(self):
        _, s = _fresh()
        prompt = _build_system_prompt(s, off_topic=False)
        assert prompt["role"] == "system"

    def test_prompt_includes_identity(self):
        _, s = _fresh()
        prompt = _build_system_prompt(s, off_topic=False)
        assert "Ali" in prompt["content"]

    def test_off_topic_appended(self):
        _, s = _fresh()
        prompt = _build_system_prompt(s, off_topic=True)
        assert "off-topic" in prompt["content"].lower()

    def test_no_off_topic_when_false(self):
        _, s = _fresh()
        prompt = _build_system_prompt(s, off_topic=False)
        assert "[POLICY]" not in prompt["content"]


# ── Inventory ────────────────────────────────────────────────────────────────

class TestInventory:
    """Smoke tests to ensure the inventory constant is valid."""

    def test_has_three_categories(self):
        assert set(INVENTORY.keys()) == {"Shops", "Houses/Villas", "Apartments"}

    def test_each_category_has_entries(self):
        for category, items in INVENTORY.items():
            assert len(items) > 0, f"{category} is empty"
            for name, price in items:
                assert "PKR" in price
