import streamlit as st

from state import chat_session_key, get_chat_state, reset_chat_context_state
from tabs.chat_ui import (
    GenerationConfig,
    generation_dict,
    render_chat_window,
    render_generation_settings,
    render_persona_prompt_controls,
    render_system_prompt,
)
from tabs.compare_chat import render_compare_mode
from tabs.probe_ui import render_probe_inspector
from utils.chat import (
    ChatReply,
    build_chat_messages,
    generate_chat_reply,
    resolve_system_prompt,
)
from utils.chat_export import save_chat_export
from utils.datasets import load_dataset
from utils.helpers import widget_key
from utils.runtime import cached_model

_LAST_PERSONA_ID_KEY = "chat:last_persona_id"
_LAST_PROMPT_MODE_KEY = "chat:last_prompt_mode"
_LAST_COMPARE_MODE_KEY = "chat:last_compare_mode"


def render_chat_tab(remote: bool, model_name: str, dataset_source: str) -> None:
    """Render the chat tab."""

    st.title("Chat")

    context_key = chat_session_key(model_name, dataset_source)
    chat_state = get_chat_state(model_name, remote, dataset_source)

    # Carry over persona / prompt selections across model or remote switches.
    if chat_state["persona_id"] is None:
        chat_state["persona_id"] = st.session_state.get(_LAST_PERSONA_ID_KEY)
        chat_state["prompt_mode"] = st.session_state.get(
            _LAST_PROMPT_MODE_KEY, "templated"
        )

    try:
        dataset, dataset_status = load_dataset(
            dataset_source,
            personas_file=st.session_state.get("extract__personas_file"),
            qa_file=st.session_state.get("extract__qa_file"),
        )
        st.caption(dataset_status)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.info("Check the selected dataset source or upload both JSONL files.")
        return

    personas = list(dataset)
    if not personas:
        st.warning("No personas found in the selected dataset.")
        st.info("Try a different dataset source or upload a non-empty personas file.")
        return

    generation: GenerationConfig = render_generation_settings(context_key, remote)
    probe_enabled = st.toggle(
        "Probe tools",
        value=False,
        key=widget_key(context_key, "probe_enabled"),
        help="Trace chat activations and run compatible `.pt` probes on tapped tokens.",
    )

    # ── Mode toggle ───────────────────────────────────────────────────────────
    compare_key = widget_key(context_key, "compare_mode")
    if compare_key not in st.session_state:
        st.session_state[compare_key] = st.session_state.get(
            _LAST_COMPARE_MODE_KEY, False
        )
    compare_mode = st.toggle(
        "Compare mode",
        key=compare_key,
        help="Side-by-side: send one message to two independent persona/prompt configurations.",
    )
    st.session_state[_LAST_COMPARE_MODE_KEY] = compare_mode

    if compare_mode:
        render_compare_mode(
            remote,
            model_name,
            context_key,
            dataset_source,
            personas,
            generation,
        )
        return

    # ── Single-chat mode ──────────────────────────────────────────────────────
    persona_select_key = widget_key(context_key, "persona_select")
    prompt_mode_select_key = widget_key(context_key, "system_prompt_select")
    prompt_key = widget_key(context_key, "custom_system_prompt")
    chat_input_key = widget_key(context_key, "chat_input")
    pending_key = widget_key(context_key, "pending_prompt")
    export_key = widget_key(context_key, "export_chat")
    reset_key = widget_key(context_key, "reset")
    edit_key = widget_key(context_key, "edit_idx")

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

    selected_persona, prompt_mode, changed_context = render_persona_prompt_controls(
        personas,
        chat_state["persona_id"],
        chat_state["prompt_mode"],
        persona_select_key,
        prompt_mode_select_key,
        column_widths=(2, 1),
    )
    st.session_state[_LAST_PERSONA_ID_KEY] = selected_persona.id
    st.session_state[_LAST_PROMPT_MODE_KEY] = prompt_mode

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
        )

    render_probe_inspector(
        context_key=context_key,
        model_name=model_name,
        remote=remote,
        active_system_prompt=active_system_prompt,
        chat_state=chat_state,
        enabled=probe_enabled,
    )

    render_chat_window(
        chat_log=chat_log,
        messages=chat_state["messages"],
        chat_state=chat_state,
        edit_key=edit_key,
        pending_key=pending_key,
    )

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
                    persona_id=selected_persona.id,
                    persona_name=getattr(selected_persona, "name", None),
                    prompt_mode=prompt_mode,
                    system_prompt=active_system_prompt,
                    messages=chat_state["messages"],
                    generation=generation_dict(generation),
                )
                st.toast("Exported", icon=":material/check:")
        with rst_col:
            if st.button(
                "",
                icon=":material/delete_sweep:",
                key=reset_key,
                help="Reset chat",
            ):
                _reset_active_chat_context()
                st.rerun()

    user_prompt = st.chat_input("Ask something...", key=chat_input_key)

    # Pass 1: user submitted — append message and rerun so it renders before generation.
    if user_prompt:
        chat_state["messages"].append({"role": "user", "content": user_prompt})
        st.session_state[pending_key] = "new_user_prompt"
        st.rerun()

    # Pass 2: message is already rendered above; now run generation.
    pending_action = st.session_state.pop(pending_key, None)
    if not pending_action:
        return

    messages = build_chat_messages(active_system_prompt, chat_state["messages"])

    with st.spinner("Generating reply..."):
        model = cached_model(model_name=model_name, remote=remote)
        try:
            reply: ChatReply = generate_chat_reply(
                model=model,
                messages=messages,
                remote=remote,
                past_key_values=chat_state["past_key_values"],
                **generation.to_generate_kwargs(),
            )
        except Exception as exc:
            with chat_log:
                st.error(f"Could not generate a reply: {exc}")
                st.info("Try a shorter prompt, reset the chat, or switch personas.")
            if pending_action == "new_user_prompt" and chat_state["messages"]:
                chat_state["messages"].pop()
            return

    chat_state["messages"].append({"role": "assistant", "content": reply.text})
    chat_state["past_key_values"] = reply.past_key_values if not remote else None
    st.rerun()
