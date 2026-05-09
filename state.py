from typing import Literal, NotRequired, TypedDict

import streamlit as st

_CHAT_STATE_PREFIX = "chat_state::"
PendingChatAction = Literal["new_user_prompt", "regenerate_after_edit"]


class ChatMessage(TypedDict):
    role: str
    content: str
    _contrast: NotRequired[object]
    _needs_contrast: NotRequired[bool]


class ChatState(TypedDict):
    messages: list[ChatMessage]
    persona_id: str | None
    prompt_mode: str


def chat_session_key(model_name: str, dataset_source: str) -> str:
    """Build the session-state key for a chat context."""

    return f"{_CHAT_STATE_PREFIX}{model_name}::{dataset_source}"


def default_chat_state() -> ChatState:
    return {
        "messages": [],
        "persona_id": None,
        "prompt_mode": "templated",
    }


def reset_chat_context_state(
    state: ChatState,
    persona_id: str,
    prompt_mode: str,
    *ui_keys: str,
) -> None:
    """Reset one chat context and clear any related widget state."""

    state["messages"] = []
    state["persona_id"] = persona_id
    state["prompt_mode"] = prompt_mode
    for key in ui_keys:
        st.session_state.pop(key, None)


def get_chat_state(model_name: str, _remote: bool, dataset_source: str) -> ChatState:
    """Return the mutable chat state for the active context."""

    key = chat_session_key(model_name, dataset_source)
    state = st.session_state.setdefault(key, default_chat_state())
    return state
