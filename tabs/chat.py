from concurrent.futures import ThreadPoolExecutor
from typing import Any

import streamlit as st
from persona_data.synth_persona import PersonaData

from state import (
    _default_chat_state,
    chat_session_key,
    get_chat_state,
    reset_chat_state,
)
from utils.chat import ChatReply, generate_chat_reply, resolve_system_prompt
from utils.chat_export import save_chat_export
from utils.datasets import load_dataset
from utils.helpers import (
    MODE_LABEL_TO_KEY,
    MODE_LABELS,
    VARIANT_LABELS,
    VISIBLE_MESSAGE_COUNT,
    persona_label,
    widget_key,
)
from utils.runtime import cached_model

COLLAPSED_MESSAGE_CHAR_LIMIT = 500


def _render_collapsible_markdown(content: str) -> None:
    if len(content) <= COLLAPSED_MESSAGE_CHAR_LIMIT:
        st.markdown(content)
        return

    with st.expander(f"Show full text ({len(content)} chars)", expanded=False):
        st.markdown(content)


def _render_chat_message(message: dict[str, str]) -> None:
    if not message.get("content"):
        return
    with st.container(border=True):
        st.caption(message["role"])
        _render_collapsible_markdown(message["content"])


def _render_inline_system_prompt(
    prompt_key: str,
    prompt_mode: str,
    active_system_prompt: str | None,
    height: int = 200,
) -> str | None:
    """Render the system prompt as an always-editable text area at the top of the chat."""
    if prompt_mode == "empty":
        return active_system_prompt

    if prompt_key not in st.session_state:
        st.session_state[prompt_key] = active_system_prompt or ""

    with st.container(border=True):
        st.caption("System prompt")
        st.text_area(
            "system_prompt_edit",
            value=st.session_state[prompt_key],
            height=height,
            label_visibility="collapsed",
            key=prompt_key,
        )

    return st.session_state.get(prompt_key) or None


def _render_editable_message(
    message: dict[str, str],
    msg_index: int,
    messages: list[dict[str, str]],
    chat_state: dict[str, object],
    edit_key: str,
    pending_key: str,
) -> None:
    """Render a single message with an inline edit button."""
    if not message.get("content"):
        return

    is_editing = st.session_state.get(edit_key) == msg_index

    with st.container(border=True):
        st.caption(message["role"])
        if is_editing:
            new_content = st.text_area(
                "Edit",
                value=message["content"],
                height=100,
                label_visibility="collapsed",
                key=f"{edit_key}_msg_{msg_index}",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "Save", key=f"{edit_key}_msg_save_{msg_index}", type="primary"
                ):
                    messages[msg_index]["content"] = new_content
                    del messages[msg_index + 1 :]
                    chat_state["past_key_values"] = None
                    st.session_state[edit_key] = None
                    if message["role"] == "user":
                        st.session_state[pending_key] = True
                    st.rerun()
            with c2:
                if st.button("Cancel", key=f"{edit_key}_msg_cancel_{msg_index}"):
                    st.session_state[edit_key] = None
                    st.rerun()
        else:
            st.markdown(message["content"])
            if st.button("Edit", key=f"{edit_key}_msg_edit_{msg_index}"):
                st.session_state[edit_key] = msg_index
                st.rerun()


def _clear_chat_ui_state(*keys: str) -> None:
    for key in keys:
        st.session_state.pop(key, None)


def _reset_single_chat_context(
    model_name: str,
    dataset_source: str,
    chat_state: dict[str, object],
    persona_id: str,
    prompt_mode: str,
    *ui_keys: str,
) -> None:
    reset_chat_state(model_name, dataset_source)
    chat_state["persona_id"] = persona_id
    chat_state["prompt_mode"] = prompt_mode
    _clear_chat_ui_state(*ui_keys)


def _generation_dict(gen_kwargs: dict, advanced_generation: bool) -> dict[str, object]:
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


def _render_persona_prompt_controls(
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


def _render_chat_window(
    *,
    chat_log: Any,
    messages: list[dict[str, str]],
    show_all_key: str,
    show_all_btn_key: str,
    show_earlier_label: str,
    chat_state: dict[str, object] | None = None,
    edit_key: str | None = None,
    pending_key: str | None = None,
) -> Any:
    """Render the visible chat history inside one container."""

    with chat_log:
        if len(messages) > VISIBLE_MESSAGE_COUNT and not st.session_state.get(
            show_all_key, False
        ):
            hidden_count = len(messages) - VISIBLE_MESSAGE_COUNT
            if st.button(
                f"{show_earlier_label} ({hidden_count} hidden)",
                key=show_all_btn_key,
            ):
                st.session_state[show_all_key] = True
                st.rerun()
            visible_messages = messages[-VISIBLE_MESSAGE_COUNT:]
            index_offset = len(messages) - VISIBLE_MESSAGE_COUNT
        else:
            visible_messages = messages
            index_offset = 0

        for i, message in enumerate(visible_messages):
            actual_index = index_offset + i
            if edit_key and pending_key:
                _render_editable_message(
                    message, actual_index, messages, chat_state, edit_key, pending_key
                )
            else:
                _render_chat_message(message)

    return chat_log


def _build_chat_messages(
    system_prompt: str | None,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    return (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    ) + messages


def _save_chat_export_message(
    *,
    model_name: str,
    dataset_source: str,
    persona_id: str,
    persona_name: str | None,
    prompt_mode: str,
    system_prompt: str | None,
    messages: list[dict[str, str]],
    generation: dict[str, object],
    panel_label: str | None = None,
) -> str:
    export_path = save_chat_export(
        model_name=model_name,
        dataset_source=dataset_source,
        persona_id=persona_id,
        persona_name=persona_name,
        panel_label=panel_label,
        prompt_mode=prompt_mode,
        system_prompt=system_prompt,
        messages=messages,
        generation=generation,
    )
    return f"Saved chat export to {export_path}"


# ── Compare mode helpers ───────────────────────────────────────────────────────


def _panel_state(panel_key: str) -> dict:
    """Get or initialise compare-panel chat state stored in session_state."""
    if panel_key not in st.session_state:
        st.session_state[panel_key] = _default_chat_state()
    return st.session_state[panel_key]


def _render_compare_mode(
    remote: bool,
    model_name: str,
    context_key: str,
    dataset_source: str,
    personas: list[PersonaData],
    gen_kwargs: dict,
    advanced_generation: bool,
) -> None:
    """Render the full side-by-side comparison UI."""
    left_col, right_col = st.columns(2)

    def render_panel(side: str) -> tuple[dict[str, object], Any, str | None, str]:
        panel_key = widget_key(context_key, f"cmp_{side}")
        state = _panel_state(panel_key)
        prompt_key = widget_key(panel_key, "custom_prompt")
        show_all_key = widget_key(panel_key, "show_all")
        edit_key = widget_key(panel_key, "edit_idx")
        pending_regen_key = widget_key(panel_key, "pending_regen")

        selected_persona, prompt_mode, changed = _render_persona_prompt_controls(
            personas,
            state["persona_id"],
            state["prompt_mode"],
            widget_key(panel_key, "persona"),
            widget_key(panel_key, "prompt_mode"),
        )
        if changed:
            state["messages"] = []
            state["past_key_values"] = None
            state["persona_id"] = selected_persona.id
            state["prompt_mode"] = prompt_mode
            _clear_chat_ui_state(prompt_key, show_all_key)
            st.session_state.pop(edit_key, None)

        active_system_prompt = resolve_system_prompt(
            persona=selected_persona, mode=prompt_mode
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button(
                "Export chat", key=widget_key(panel_key, "export_chat"), width="stretch"
            ):
                st.success(
                    _save_chat_export_message(
                        model_name=model_name,
                        dataset_source=dataset_source,
                        persona_id=selected_persona.id,
                        persona_name=getattr(selected_persona, "name", None),
                        prompt_mode=prompt_mode,
                        system_prompt=active_system_prompt,
                        messages=state["messages"],
                        generation=_generation_dict(gen_kwargs, advanced_generation),
                        panel_label=side,
                    )
                )
        with btn_col2:
            if st.button(
                "Reset chat",
                key=widget_key(panel_key, "reset"),
                width="stretch",
                type="secondary",
            ):
                state["messages"] = []
                state["past_key_values"] = None
                _clear_chat_ui_state(prompt_key, show_all_key)
                st.session_state.pop(edit_key, None)
                st.rerun()

        chat_log = st.container()
        with chat_log:
            active_system_prompt = _render_inline_system_prompt(
                prompt_key,
                prompt_mode,
                active_system_prompt,
                height=150,
            )
        _render_chat_window(
            chat_log=chat_log,
            messages=state["messages"],
            show_all_key=show_all_key,
            show_all_btn_key=widget_key(panel_key, "show_all_btn"),
            show_earlier_label="Show earlier",
            chat_state=state,
            edit_key=edit_key,
            pending_key=pending_regen_key,
        )
        return state, chat_log, active_system_prompt, pending_regen_key

    with left_col:
        left_state, left_log, left_prompt, left_pending = render_panel("left")
    with right_col:
        right_state, right_log, right_prompt, right_pending = render_panel("right")

    panels = [
        (left_state, left_log, left_prompt, left_pending),
        (right_state, right_log, right_prompt, right_pending),
    ]

    # Handle per-panel regeneration triggered by message edits
    any_regen = any(st.session_state.get(p_pending) for _, _, _, p_pending in panels)
    if any_regen:
        model = cached_model(model_name=model_name, remote=remote)
        for panel_state, panel_log, panel_prompt, p_pending in panels:
            if not st.session_state.pop(p_pending, False):
                continue
            regen_messages = _build_chat_messages(panel_prompt, panel_state["messages"])
            with st.spinner("Regenerating..."):
                try:
                    result = generate_chat_reply(
                        model=model,
                        messages=regen_messages,
                        remote=remote,
                        past_key_values=panel_state["past_key_values"],
                        **gen_kwargs,
                    )
                except Exception as exc:
                    with panel_log:
                        st.error(f"Generation failed: {exc}")
                    panel_state["messages"].pop()
                    continue
            panel_state["messages"].append(
                {"role": "assistant", "content": result.text}
            )
            panel_state["past_key_values"] = (
                result.past_key_values if not remote else None
            )
            with panel_log:
                _render_chat_message({"role": "assistant", "content": result.text})
        st.rerun()

    user_prompt = st.chat_input(
        "Ask both...",
        key=widget_key(context_key, "cmp_input"),
    )
    if not user_prompt:
        return

    model = cached_model(model_name=model_name, remote=remote)

    for panel_state, panel_log, _panel_prompt, _p_pending in panels:
        panel_state["messages"].append({"role": "user", "content": user_prompt})
        with panel_log:
            _render_chat_message({"role": "user", "content": user_prompt})

    with st.spinner("Generating..."):
        if remote:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        generate_chat_reply,
                        model=model,
                        messages=_build_chat_messages(
                            panel_prompt, panel_state["messages"]
                        ),
                        remote=remote,
                        past_key_values=panel_state["past_key_values"],
                        **gen_kwargs,
                    )
                    for panel_state, _panel_log, panel_prompt, _p_pending in panels
                ]
                results: list[ChatReply | Exception] = []
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append(exc)
        else:
            results = []
            for panel_state, _panel_log, panel_prompt, _p_pending in panels:
                try:
                    results.append(
                        generate_chat_reply(
                            model=model,
                            messages=_build_chat_messages(
                                panel_prompt, panel_state["messages"]
                            ),
                            remote=remote,
                            past_key_values=panel_state["past_key_values"],
                            **gen_kwargs,
                        )
                    )
                except Exception as exc:
                    results.append(exc)

    for (panel_state, panel_log, _panel_prompt, _p_pending), result in zip(
        panels, results
    ):
        if isinstance(result, Exception):
            with panel_log:
                st.error(f"Generation failed: {result}")
            panel_state["messages"].pop()
            continue

        panel_state["messages"].append({"role": "assistant", "content": result.text})
        panel_state["past_key_values"] = result.past_key_values if not remote else None
        with panel_log:
            _render_chat_message({"role": "assistant", "content": result.text})

    # Rerun so the newly appended turns are redrawn through the editable history
    # renderer instead of only appearing in the one-off generation pass.
    st.rerun()


# ── Main tab entry point ───────────────────────────────────────────────────────


def _render_generation_settings(
    context_key: str, remote: bool
) -> tuple[dict, bool]:
    """Render the Advanced generation settings expander.

    Returns ``(gen_kwargs, advanced_generation)`` where ``advanced_generation``
    is True when any setting differs from its default.
    """
    with st.expander("Advanced", expanded=False):
        config_col1, config_col2 = st.columns([2, 1])
        with config_col1:
            max_new_tokens = st.slider(
                "Max new tokens",
                min_value=16,
                max_value=512,
                value=256,
                step=16,
                key=widget_key(context_key, "max_new_tokens"),
            )
        with config_col2:
            repetition_penalty = st.slider(
                "Repetition penalty",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
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
                value=1.0,
                step=0.01,
                disabled=sampling_disabled,
                key=widget_key(context_key, "temperature"),
            )
        with sampling_col2:
            top_p = st.slider(
                "Top-p",
                min_value=0.01,
                max_value=1.0,
                value=1.0,
                step=0.01,
                disabled=sampling_disabled,
                key=widget_key(context_key, "top_p"),
            )
        with sampling_col3:
            top_k = st.slider(
                "Top-k (0 = off)",
                min_value=0,
                max_value=100,
                value=50,
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
        max_new_tokens != 256
        or use_sampling
        or temperature != 1.0
        or top_p != 1.0
        or top_k != 50
        or repetition_penalty != 1.0
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
    compare_mode = st.toggle(
        "Compare mode",
        value=False,
        key=widget_key(context_key, "compare_mode"),
        help="Side-by-side: send one message to two independent persona/prompt configurations.",
    )

    if compare_mode:
        _render_compare_mode(
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
    show_all_key = widget_key(context_key, "show_all_messages")
    chat_input_key = widget_key(context_key, "chat_input")
    pending_key = widget_key(context_key, "pending_prompt")
    export_key = widget_key(context_key, "export_chat")
    reset_key = widget_key(context_key, "reset")
    edit_key = widget_key(context_key, "edit_idx")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_index = next(
            (i for i, p in enumerate(personas) if p.id == chat_state["persona_id"]),
            0,
        )
        selected_persona = st.selectbox(
            "Persona",
            options=personas,
            index=selected_index,
            format_func=persona_label,
            key=persona_select_key,
        )
    with col2:
        current_mode_label = VARIANT_LABELS.get(chat_state["prompt_mode"], "None")
        st.selectbox(
            "Prompt",
            options=MODE_LABELS,
            index=MODE_LABELS.index(current_mode_label),
            key=prompt_mode_select_key,
        )
        prompt_mode = MODE_LABEL_TO_KEY[st.session_state[prompt_mode_select_key]]

    active_system_prompt = resolve_system_prompt(
        persona=selected_persona,
        mode=prompt_mode,
    )

    changed_context = (
        chat_state["persona_id"] != selected_persona.id
        or chat_state["prompt_mode"] != prompt_mode
    )
    if changed_context:
        had_history = bool(chat_state["messages"])
        _reset_single_chat_context(
            model_name,
            dataset_source,
            chat_state,
            selected_persona.id,
            prompt_mode,
            chat_input_key,
            show_all_key,
            prompt_key,
            pending_key,
        )
        st.session_state.pop(edit_key, None)
        if had_history:
            st.info("Chat history reset because the persona or system prompt changed.")

    chat_log = st.container()

    with chat_log:
        active_system_prompt = _render_inline_system_prompt(
            prompt_key,
            prompt_mode,
            active_system_prompt,
            height=200,
        )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Export chat", key=export_key, width="stretch"):
            st.success(
                _save_chat_export_message(
                    model_name=model_name,
                    dataset_source=dataset_source,
                    persona_id=selected_persona.id,
                    persona_name=getattr(selected_persona, "name", None),
                    prompt_mode=prompt_mode,
                    system_prompt=active_system_prompt,
                    messages=chat_state["messages"],
                    generation=_generation_dict(gen_kwargs, advanced_generation),
                )
            )
    with action_col2:
        if st.button("Reset chat", key=reset_key, width="stretch", type="secondary"):
            _reset_single_chat_context(
                model_name,
                dataset_source,
                chat_state,
                selected_persona.id,
                prompt_mode,
                chat_input_key,
                show_all_key,
                prompt_key,
                pending_key,
            )
            st.session_state.pop(edit_key, None)
            st.rerun()

    _render_chat_window(
        chat_log=chat_log,
        messages=chat_state["messages"],
        show_all_key=show_all_key,
        show_all_btn_key=widget_key(context_key, "show_all_btn"),
        show_earlier_label="Show earlier messages",
        chat_state=chat_state,
        edit_key=edit_key,
        pending_key=pending_key,
    )

    user_prompt = st.chat_input(
        "Ask something...",
        key=chat_input_key,
    )

    # Pass 1: user submitted — append message and rerun so it renders before generation.
    if user_prompt:
        chat_state["messages"].append({"role": "user", "content": user_prompt})
        st.session_state[pending_key] = True
        st.rerun()

    # Pass 2: message is already rendered above; now run generation.
    if not st.session_state.pop(pending_key, False):
        return

    messages = _build_chat_messages(active_system_prompt, chat_state["messages"])

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

    save_chat_export(
        model_name=model_name,
        dataset_source=dataset_source,
        persona_id=selected_persona.id,
        persona_name=getattr(selected_persona, "name", None),
        prompt_mode=prompt_mode,
        system_prompt=active_system_prompt,
        messages=chat_state["messages"],
        generation=_generation_dict(gen_kwargs, advanced_generation),
    )
    st.rerun()
