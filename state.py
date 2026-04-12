import streamlit as st

_CHAT_STATE_PREFIX = "chat_state::"


def chat_session_key(model_name: str, dataset_source: str) -> str:
    """Build the session-state key for a chat context."""

    return f"{_CHAT_STATE_PREFIX}{model_name}::{dataset_source}"


def default_chat_state() -> dict[str, object]:
    return {
        "messages": [],
        "persona_id": None,
        "prompt_mode": "templated",
        "past_key_values": None,
    }


def reset_chat_context_state(
    state: dict[str, object],
    persona_id: str,
    prompt_mode: str,
    *ui_keys: str,
) -> None:
    """Reset one chat context and clear any related widget state."""

    state["messages"] = []
    state["past_key_values"] = None
    state["persona_id"] = persona_id
    state["prompt_mode"] = prompt_mode
    for key in ui_keys:
        st.session_state.pop(key, None)


def _evict_inactive_kv_caches(active_key: str) -> None:
    """Drop past_key_values from every chat context except the active one."""

    for key in st.session_state:
        if (
            isinstance(key, str)
            and key.startswith(_CHAT_STATE_PREFIX)
            and key != active_key
        ):
            state = st.session_state[key]
            if isinstance(state, dict) and state.get("past_key_values") is not None:
                state["past_key_values"] = None


def get_chat_state(
    model_name: str, remote: bool, dataset_source: str
) -> dict[str, object]:
    """Return the mutable chat state for the active context."""

    key = chat_session_key(model_name, dataset_source)
    state = st.session_state.get(key)
    if state is None:
        state = default_chat_state()
        st.session_state[key] = state
    else:
        for default_key, default_value in default_chat_state().items():
            state.setdefault(default_key, default_value)
    _evict_inactive_kv_caches(key)
    if remote and state.get("past_key_values") is not None:
        state["past_key_values"] = None
    return state
