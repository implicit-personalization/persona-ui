from __future__ import annotations

from typing import TYPE_CHECKING, cast

import streamlit as st

from state import (
    ChatState,
    PendingChatAction,
    chat_session_key,
    get_chat_state,
    reset_chat_context_state,
)
from tabs.chat_shared import (
    generate_chat_reply_result,
    hydrate_chat_state,
    load_chat_personas,
    mark_model_loaded,
    model_load_status,
    render_chat_selection,
)
from tabs.chat_ui import (
    GenerationConfig,
    render_advanced_settings,
    render_chat_window,
    render_system_prompt,
)
from utils.chat import build_chat_messages, resolve_system_prompt
from utils.chat_export import save_chat_export
from utils.helpers import format_ndif_status, session_key, widget_key
from utils.runtime import cached_model, session_ndif_api_key

if TYPE_CHECKING:
    from persona_data.synth_persona import PersonaData

_LAST_PERSONA_ID_KEY = session_key("chat", "last_persona_id")
_LAST_PROMPT_MODE_KEY = session_key("chat", "last_prompt_mode")
_LAST_COMPARE_MODE_KEY = session_key("chat", "last_compare_mode")
_LAST_PROBE_ENABLED_KEY = session_key("chat", "last_probe_enabled")
_LAST_TOKEN_CONTRAST_KEY = session_key("chat", "last_token_contrast")


def _render_single_chat_footer(
    *,
    model_name: str,
    dataset_source: str,
    persona: PersonaData,
    prompt_mode: str,
    system_prompt: str | None,
    chat_state: ChatState,
    generation: GenerationConfig,
    export_key: str,
    reset_key: str,
    on_reset,
) -> None:
    footer = st.container()
    with footer:
        exp_col, rst_col, _spacer = st.columns([0.5, 0.5, 10], gap="xsmall")
        with exp_col:
            if st.button(
                "",
                icon=":material/download:",
                key=export_key,
                help="Export chat",
            ):
                save_chat_export(
                    model_name=model_name,
                    dataset_source=dataset_source,
                    persona_id=persona.id,
                    persona_name=getattr(persona, "name", None),
                    prompt_mode=prompt_mode,
                    system_prompt=system_prompt,
                    messages=chat_state["messages"],
                    generation=generation.to_export_dict(),
                )
                st.toast("Exported", icon=":material/check:")
        with rst_col:
            if st.button(
                "",
                icon=":material/delete_sweep:",
                key=reset_key,
                help="Reset chat",
            ):
                on_reset()
                st.rerun()


def _handle_single_chat_generation(
    *,
    remote: bool,
    model_name: str,
    chat_state: ChatState,
    active_system_prompt: str | None,
    generation: GenerationConfig,
    pending_action: PendingChatAction,
    chat_log,
) -> None:
    messages = build_chat_messages(active_system_prompt, chat_state["messages"])
    status_box = st.empty()

    def _show_phase(text: str) -> None:
        status_box.caption(text)

    def _show_ndif_status(job_id: str, status_name: str, description: str) -> None:
        status_box.caption(
            format_ndif_status(
                job_id,
                status_name,
                description,
                completed_detail="Downloading result...",
            )
        )

    with st.spinner("Generating reply..."):
        _show_phase(model_load_status(model_name))
        model = cached_model(model_name=model_name)
        mark_model_loaded(model_name)
        _show_phase("Submitting to NDIF..." if remote else "Generating locally...")

        def _show_error(exc: Exception) -> None:
            with chat_log:
                st.error(f"Could not generate a reply: {exc}")
                st.info("Try a shorter prompt, reset the chat, or switch personas.")

        reply, error = generate_chat_reply_result(
            model=model,
            messages=messages,
            remote=remote,
            generation=generation,
            on_status=_show_ndif_status if remote else None,
            on_error=_show_error,
            ndif_api_key=session_ndif_api_key(),
        )
        if error is not None:
            status_box.empty()
            if pending_action == "new_user_prompt" and chat_state["messages"]:
                chat_state["messages"].pop()
            return
        if reply is None:
            status_box.empty()
            return

    status_box.empty()
    chat_state["messages"].append({"role": "assistant", "content": reply.text})
    st.rerun()


def render_chat_tab(remote: bool, model_name: str, dataset_source: str) -> None:
    """Render the chat tab."""

    st.title("Chat")
    st.caption("Chat with a persona, optionally side-by-side or with token contrast.")

    context_key = chat_session_key(model_name, dataset_source)
    chat_state = get_chat_state(model_name, dataset_source)
    hydrate_chat_state(
        chat_state,
        persisted_persona_key=_LAST_PERSONA_ID_KEY,
        persisted_prompt_key=_LAST_PROMPT_MODE_KEY,
    )

    personas = load_chat_personas(dataset_source)
    if personas is None:
        return

    generation, tools = render_advanced_settings(
        context_key,
        remote,
        last_compare_mode_key=_LAST_COMPARE_MODE_KEY,
        last_probe_enabled_key=_LAST_PROBE_ENABLED_KEY,
        last_token_contrast_key=_LAST_TOKEN_CONTRAST_KEY,
    )
    if tools.compare_mode:
        from tabs.compare_chat import render_compare_mode

        render_compare_mode(
            remote,
            model_name,
            context_key,
            dataset_source,
            personas,
            generation,
            contrast_enabled=tools.token_contrast,
        )
        return

    probe_container = st.container()

    persona_select_key = widget_key(context_key, "persona_select")
    prompt_mode_select_key = widget_key(context_key, "system_prompt_select")
    prompt_key = widget_key(context_key, "custom_system_prompt")
    chat_input_key = widget_key(context_key, "chat_input")
    pending_key = widget_key(context_key, "pending_prompt")
    export_key = widget_key(context_key, "export_chat")
    reset_key = widget_key(context_key, "reset")
    edit_key = widget_key(context_key, "edit_idx")

    selection = render_chat_selection(
        personas,
        chat_state["persona_id"],
        chat_state["prompt_mode"],
        persona_select_key,
        prompt_mode_select_key,
        persisted_persona_key=_LAST_PERSONA_ID_KEY,
        persisted_prompt_key=_LAST_PROMPT_MODE_KEY,
        column_widths=(2, 1),
    )
    selected_persona = selection.persona
    prompt_mode = selection.prompt_mode
    changed_context = selection.changed

    def _reset_active_chat_context() -> None:
        reset_chat_context_state(
            chat_state,
            selected_persona.id,
            prompt_mode,
            chat_input_key,
            prompt_key,
            pending_key,
        )
        st.session_state.pop(edit_key, None)

    active_system_prompt = resolve_system_prompt(
        persona=selected_persona,
        mode=prompt_mode,
    )

    if changed_context:
        had_history = bool(chat_state["messages"])
        _reset_active_chat_context()
        if had_history:
            st.info("Chat history reset because the persona or system prompt changed.")

    chat_log = st.container()

    with chat_log:
        active_system_prompt = render_system_prompt(
            prompt_key,
            prompt_mode,
            active_system_prompt,
            on_save=lambda: reset_chat_context_state(
                chat_state,
                selected_persona.id,
                prompt_mode,
                chat_input_key,
                pending_key,
            ),
        )

    with probe_container:
        if tools.probe_enabled:
            from tabs.probe_ui import render_probe_inspector

            render_probe_inspector(
                context_key=context_key,
                model_name=model_name,
                remote=remote,
                active_system_prompt=active_system_prompt,
                chat_state=chat_state,
                enabled=True,
            )
        else:
            from utils.probe_overlay import clear_overlays

            clear_overlays(chat_state["messages"])

    render_chat_window(
        chat_log=chat_log,
        messages=chat_state["messages"],
        edit_key=edit_key,
        pending_key=pending_key,
    )

    _render_single_chat_footer(
        model_name=model_name,
        dataset_source=dataset_source,
        persona=selected_persona,
        prompt_mode=prompt_mode,
        system_prompt=active_system_prompt,
        chat_state=chat_state,
        generation=generation,
        export_key=export_key,
        reset_key=reset_key,
        on_reset=_reset_active_chat_context,
    )

    user_prompt = st.chat_input("Ask something...", key=chat_input_key)

    if user_prompt:
        chat_state["messages"].append({"role": "user", "content": user_prompt})
        st.session_state[pending_key] = "new_user_prompt"
        st.rerun()

    pending_action = cast(
        PendingChatAction | None,
        st.session_state.pop(pending_key, None),
    )
    if not pending_action:
        return

    _handle_single_chat_generation(
        remote=remote,
        model_name=model_name,
        chat_state=chat_state,
        active_system_prompt=active_system_prompt,
        generation=generation,
        pending_action=pending_action,
        chat_log=chat_log,
    )
