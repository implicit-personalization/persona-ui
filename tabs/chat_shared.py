from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

from state import ChatState
from tabs.chat_ui import GenerationConfig, render_persona_prompt_controls
from utils.chat import ChatReply, generate_chat_reply
from utils.datasets import load_persona_list
from utils.helpers import session_key

if TYPE_CHECKING:
    from persona_data.synth_persona import PersonaData


@dataclass(frozen=True)
class ChatSelection:
    persona: PersonaData
    prompt_mode: str
    changed: bool


def load_chat_personas(dataset_source: str) -> list[PersonaData] | None:
    personas_file_key = session_key("extract", "personas_file")
    qa_file_key = session_key("extract", "qa_file")
    try:
        personas, dataset_status = load_persona_list(
            dataset_source,
            personas_file=st.session_state.get(personas_file_key),
            qa_file=st.session_state.get(qa_file_key),
        )
        st.caption(dataset_status)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.info("Check the selected dataset source or upload both JSONL files.")
        return None

    if not personas:
        st.warning("No personas found in the selected dataset.")
        st.info("Try a different dataset source or upload a non-empty personas file.")
        return None
    return personas


def hydrate_chat_state(
    state: ChatState,
    *,
    persisted_persona_key: str,
    persisted_prompt_key: str,
    default_prompt_mode: str = "templated",
) -> None:
    if state["persona_id"] is None:
        state["persona_id"] = st.session_state.get(persisted_persona_key)
        state["prompt_mode"] = st.session_state.get(
            persisted_prompt_key,
            default_prompt_mode,
        )


def render_chat_selection(
    personas: list[PersonaData],
    current_persona_id: str | None,
    current_prompt_mode: str,
    persona_key: str,
    prompt_key: str,
    *,
    persisted_persona_key: str,
    persisted_prompt_key: str,
    column_widths: tuple[int, int] = (3, 2),
) -> ChatSelection:
    selected_persona, prompt_mode, changed = render_persona_prompt_controls(
        personas,
        current_persona_id,
        current_prompt_mode,
        persona_key,
        prompt_key,
        column_widths=column_widths,
    )
    st.session_state[persisted_persona_key] = selected_persona.id
    st.session_state[persisted_prompt_key] = prompt_mode
    return ChatSelection(selected_persona, prompt_mode, changed)


def generate_chat_reply_result(
    *,
    model: object,
    messages: list[dict[str, str]],
    remote: bool,
    generation: GenerationConfig,
    on_error: Callable[[Exception], None] | None = None,
) -> tuple[ChatReply | None, Exception | None]:
    try:
        return (
            generate_chat_reply(
                model=model,
                messages=messages,
                remote=remote,
                **generation.to_generate_kwargs(),
            ),
            None,
        )
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        return None, exc
