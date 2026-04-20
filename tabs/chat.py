from typing import Any

import streamlit as st
from persona_data.synth_persona import PersonaData

from state import chat_session_key, get_chat_state, reset_chat_context_state
from utils.chat import ChatReply, generate_chat_reply, resolve_system_prompt
from utils.chat_export import save_chat_export
from utils.contrast import TokenContrast, render_contrast_html
from utils.datasets import load_dataset
from utils.helpers import (
    MODE_LABEL_TO_KEY,
    MODE_LABELS,
    VARIANT_LABELS,
    persona_label,
    widget_key,
)
from utils.runtime import cached_model

# ── Persistence keys for surviving model / remote switches ────────────────────
_LAST_PERSONA_ID_KEY = "chat:last_persona_id"
_LAST_PROMPT_MODE_KEY = "chat:last_prompt_mode"
_LAST_COMPARE_MODE_KEY = "chat:last_compare_mode"

# ── Generation defaults (single source of truth) ─────────────────────────────
_GEN_DEFAULTS = {
    "max_new_tokens": 256,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.0,
}

# ── Dialogs ───────────────────────────────────────────────────────────────────


@st.dialog("Edit", width="medium")
def _open_edit_dialog(
    *,
    msg_index: int,
    messages: list[dict[str, str]],
    chat_state: dict[str, object],
    pending_key: str,
) -> None:
    message = messages[msg_index]
    role = message["role"]

    n_after = len(messages) - msg_index - 1
    st.caption(
        f"**{role}**"
        + (
            f" — {n_after} subsequent {'message' if n_after == 1 else 'messages'} will be cleared"
            if n_after > 0
            else ""
        )
    )

    new_content = st.text_area(
        "Content",
        value=message["content"],
        height=320,
        label_visibility="collapsed",
    )

    save_col, cancel_col = st.columns(2)
    with save_col:
        if st.button("Save", type="primary", use_container_width=True):
            messages[msg_index]["content"] = new_content
            messages[msg_index].pop("_contrast", None)
            if role == "assistant":
                messages[msg_index]["_needs_contrast"] = True
            del messages[msg_index + 1 :]
            chat_state["past_key_values"] = None
            if role == "user":
                st.session_state[pending_key] = True
            st.rerun()
    with cancel_col:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("Edit system prompt", width="large")
def _open_system_prompt_dialog(*, prompt_key: str, current_value: str) -> None:
    new_value = st.text_area(
        "System prompt",
        value=current_value,
        height=320,
        label_visibility="collapsed",
    )
    save_col, cancel_col = st.columns(2)
    with save_col:
        if st.button("Save", type="primary", use_container_width=True):
            st.session_state[prompt_key] = new_value
            st.rerun()
    with cancel_col:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ── Message renderers ─────────────────────────────────────────────────────────


def render_chat_message(
    message: dict[str, str],
    show_contrast: bool = False,
) -> None:
    if not message.get("content"):
        return
    role = message["role"]
    tc: TokenContrast | None = message.get("_contrast") if show_contrast else None
    with st.chat_message(role):
        if tc is not None:
            st.html(render_contrast_html(tc))
        else:
            st.markdown(message["content"])


def _render_editable_message(
    message: dict[str, str],
    msg_index: int,
    messages: list[dict[str, str]],
    chat_state: dict[str, object],
    edit_key: str,
    pending_key: str,
    show_contrast: bool = False,
    column_ratio: tuple[int, int] = (25, 1),
) -> None:
    if not message.get("content"):
        return
    role = message["role"]
    tc: TokenContrast | None = message.get("_contrast") if show_contrast else None

    msg_col, edit_col = st.columns(
        list(column_ratio), gap="xsmall", vertical_alignment="center"
    )

    with msg_col:
        with st.chat_message(role):
            if tc is not None:
                st.html(render_contrast_html(tc))
            else:
                st.markdown(message["content"])
    with edit_col:
        if st.button(
            "", icon=":material/edit:", key=f"{edit_key}_edit_{msg_index}", help="Edit"
        ):
            _open_edit_dialog(
                msg_index=msg_index,
                messages=messages,
                chat_state=chat_state,
                pending_key=pending_key,
            )


def render_system_prompt(
    prompt_key: str,
    prompt_mode: str,
    active_system_prompt: str | None,
) -> str | None:
    if prompt_key not in st.session_state:
        st.session_state[prompt_key] = active_system_prompt or ""
    current = st.session_state.get(prompt_key) or ""
    with st.expander("System prompt"):
        st.markdown(current or "*empty*")
        if prompt_mode != "empty" and st.button(
            "Edit", icon=":material/edit:", key=f"{prompt_key}_edit"
        ):
            _open_system_prompt_dialog(prompt_key=prompt_key, current_value=current)
    return st.session_state.get(prompt_key) or None


def generation_dict(gen_kwargs: dict, advanced_generation: bool) -> dict[str, object]:
    return {
        "max_new_tokens": int(gen_kwargs["max_new_tokens"]),
        "advanced_generation": bool(advanced_generation),
        "use_sampling": bool(gen_kwargs["do_sample"]),
        "temperature": float(gen_kwargs["temperature"]),
        "top_p": float(gen_kwargs["top_p"]),
        "top_k": int(gen_kwargs["top_k"]),
        "repetition_penalty": float(gen_kwargs["repetition_penalty"]),
        "seed": gen_kwargs["seed"],
    }


def render_persona_prompt_controls(
    personas: list[PersonaData],
    current_persona_id: str | None,
    current_prompt_mode: str,
    persona_key: str,
    prompt_key: str,
    column_widths: tuple[int, int] = (3, 2),
) -> tuple[PersonaData, str, bool]:
    """Render persona and prompt selectors, returning the selected values."""

    p_col, m_col = st.columns(list(column_widths))
    with p_col:
        selected_index = next(
            (i for i, p in enumerate(personas) if p.id == current_persona_id), 0
        )
        selected_persona = st.selectbox(
            "Persona",
            options=personas,
            index=selected_index,
            format_func=persona_label,
            key=persona_key,
        )
    with m_col:
        current_label = VARIANT_LABELS.get(current_prompt_mode, "None")
        prompt_mode_label = st.selectbox(
            "Prompt",
            options=MODE_LABELS,
            index=MODE_LABELS.index(current_label),
            key=prompt_key,
        )
    prompt_mode = MODE_LABEL_TO_KEY[prompt_mode_label]
    changed = (
        current_persona_id != selected_persona.id or current_prompt_mode != prompt_mode
    )
    return selected_persona, prompt_mode, changed


def render_chat_window(
    *,
    chat_log: Any,
    messages: list[dict[str, str]],
    chat_state: dict[str, object] | None = None,
    edit_key: str | None = None,
    pending_key: str | None = None,
    show_contrast: bool = False,
    edit_column_ratio: tuple[int, int] = (25, 1),
) -> None:
    with chat_log:
        for i, message in enumerate(messages):
            if edit_key and pending_key:
                _render_editable_message(
                    message,
                    i,
                    messages,
                    chat_state,
                    edit_key,
                    pending_key,
                    show_contrast=show_contrast,
                    column_ratio=edit_column_ratio,
                )
            else:
                render_chat_message(message, show_contrast=show_contrast)


def build_chat_messages(
    system_prompt: str | None,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    return (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    ) + messages


# ── Main tab entry point ───────────────────────────────────────────────────────


def _render_generation_settings(context_key: str, remote: bool) -> tuple[dict, bool]:
    """Render the Advanced generation settings expander.

    Returns ``(gen_kwargs, advanced_generation)`` where ``advanced_generation``
    is True when any generation setting differs from its default.
    """
    with st.expander("Advanced", expanded=False):
        config_col1, config_col2 = st.columns([2, 1])
        with config_col1:
            max_new_tokens = st.slider(
                "Max new tokens",
                min_value=16,
                max_value=512,
                value=_GEN_DEFAULTS["max_new_tokens"],
                step=16,
                key=widget_key(context_key, "max_new_tokens"),
            )
        with config_col2:
            repetition_penalty = st.slider(
                "Repetition penalty",
                min_value=0.5,
                max_value=2.0,
                value=_GEN_DEFAULTS["repetition_penalty"],
                step=0.05,
                key=widget_key(context_key, "repetition_penalty"),
            )

        use_sampling = st.checkbox(
            "Random sampling",
            value=False,
            key=widget_key(context_key, "use_sampling"),
        )

        sampling_disabled = not use_sampling
        sampling_col1, sampling_col2, sampling_col3 = st.columns(3)
        with sampling_col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.01,
                max_value=2.0,
                value=_GEN_DEFAULTS["temperature"],
                step=0.01,
                disabled=sampling_disabled,
                key=widget_key(context_key, "temperature"),
            )
        with sampling_col2:
            top_p = st.slider(
                "Top-p",
                min_value=0.01,
                max_value=1.0,
                value=_GEN_DEFAULTS["top_p"],
                step=0.01,
                disabled=sampling_disabled,
                key=widget_key(context_key, "top_p"),
            )
        with sampling_col3:
            top_k = st.slider(
                "Top-k (0 = off)",
                min_value=0,
                max_value=100,
                value=_GEN_DEFAULTS["top_k"],
                step=1,
                disabled=sampling_disabled,
                key=widget_key(context_key, "top_k"),
            )

        seed_disabled = sampling_disabled or remote
        seed_enabled = st.checkbox(
            "Fix seed",
            value=False,
            disabled=seed_disabled,
            key=widget_key(context_key, "seed_enabled"),
        )
        if seed_enabled:
            seed = int(
                st.number_input(
                    "Seed",
                    min_value=0,
                    max_value=2_147_483_647,
                    value=0,
                    step=1,
                    disabled=seed_disabled,
                    key=widget_key(context_key, "seed"),
                )
            )
        else:
            seed = None

        if remote:
            st.caption("Seed is local-only and disabled for remote runs.")

    advanced_generation = (
        max_new_tokens != _GEN_DEFAULTS["max_new_tokens"]
        or use_sampling
        or temperature != _GEN_DEFAULTS["temperature"]
        or top_p != _GEN_DEFAULTS["top_p"]
        or top_k != _GEN_DEFAULTS["top_k"]
        or repetition_penalty != _GEN_DEFAULTS["repetition_penalty"]
        or seed is not None
    )

    do_sample = bool(use_sampling)
    generation_seed = seed if do_sample and seed is not None and not remote else None
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=generation_seed,
    )
    return gen_kwargs, advanced_generation


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

    gen_kwargs, advanced_generation = _render_generation_settings(context_key, remote)

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
        from tabs.compare_chat import render_compare_mode

        render_compare_mode(
            remote,
            model_name,
            context_key,
            dataset_source,
            personas,
            gen_kwargs,
            advanced_generation,
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
                    generation=generation_dict(gen_kwargs, advanced_generation),
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
        st.session_state[pending_key] = True
        st.rerun()

    # Pass 2: message is already rendered above; now run generation.
    if not st.session_state.pop(pending_key, False):
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
                **gen_kwargs,
            )
        except Exception as exc:
            with chat_log:
                st.error(f"Could not generate a reply: {exc}")
                st.info("Try a shorter prompt, reset the chat, or switch personas.")
            chat_state["messages"].pop()
            return

    chat_state["messages"].append({"role": "assistant", "content": reply.text})
    chat_state["past_key_values"] = reply.past_key_values if not remote else None
    st.rerun()
