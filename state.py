import streamlit as st

_CHAT_STATE_PREFIX = "chat_state::"
_CHAT_KEYS_REGISTRY = "chat_state::_registered_keys"


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

    for key in st.session_state.get(_CHAT_KEYS_REGISTRY, ()):
        if key != active_key:
            state = st.session_state.get(key)
            if isinstance(state, dict) and state.get("past_key_values") is not None:
                state["past_key_values"] = None


def get_chat_state(
    model_name: str, remote: bool, dataset_source: str
) -> dict[str, object]:
    """Return the mutable chat state for the active context."""

    key = chat_session_key(model_name, dataset_source)
    registry = st.session_state.get(_CHAT_KEYS_REGISTRY)
    if registry is None:
        registry = set()
        st.session_state[_CHAT_KEYS_REGISTRY] = registry
    registry.add(key)

    state = st.session_state.get(key)
    if state is None:
        state = default_chat_state()
        st.session_state[key] = state
    else:
        state.setdefault("messages", [])
        state.setdefault("persona_id", None)
        state.setdefault("prompt_mode", "templated")
        state.setdefault("past_key_values", None)
    _evict_inactive_kv_caches(key)
    if remote and state.get("past_key_values") is not None:
        state["past_key_values"] = None
    return state
