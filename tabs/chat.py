import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import streamlit as st

from state import chat_session_key, get_chat_state, reset_chat_state
from utils.chat import ChatReply, generate_chat_reply, resolve_system_prompt
from utils.chat_export import save_chat_export
from utils.datasets import load_dataset
from utils.helpers import (
    MODE_LABEL_TO_KEY,
    MODE_LABELS,
    VARIANT_LABELS,
    persona_label,
    widget_key,
)
from utils.runtime import cached_model

_VISIBLE_MESSAGE_COUNT = 5
_model_lock = threading.Lock()


def _render_chat_message(message: dict[str, str]) -> None:
    if not message.get("content"):
        return
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def _clear_chat_ui_state(*keys: str) -> None:
    for key in keys:
        st.session_state.pop(key, None)


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


# ── Compare mode helpers ───────────────────────────────────────────────────────


def _panel_state(panel_key: str) -> dict:
    """Get or initialise compare-panel chat state stored in session_state."""
    if panel_key not in st.session_state:
        st.session_state[panel_key] = {
            "messages": [],
            "persona_id": None,
            "prompt_mode": "templated",
            "past_key_values": None,
        }
    return st.session_state[panel_key]


def _render_compare_panel(
    side: str,
    context_key: str,
    personas: list,
    remote: bool,
    model_name: str,
    dataset_source: str,
    gen_kwargs: dict,
    advanced_generation: bool,
) -> dict:
    """Render persona/prompt controls + chat log for one compare panel.

    Returns a dict with keys needed by the generation step:
      panel_key, state, active_system_prompt, selected_persona, chat_log
    """
    panel_key = widget_key(context_key, f"cmp_{side}")
    state = _panel_state(panel_key)

    # ── Per-panel selectors ──────────────────────────────────────────────────
    p_col, m_col = st.columns([3, 2])
    with p_col:
        selected_index = next(
            (i for i, p in enumerate(personas) if p.id == state["persona_id"]), 0
        )
        selected_persona = st.selectbox(
            "Persona",
            options=personas,
            index=selected_index,
            format_func=persona_label,
            key=widget_key(panel_key, "persona"),
        )
    with m_col:
        current_label = VARIANT_LABELS.get(state["prompt_mode"], "None")
        prompt_mode_label = st.selectbox(
            "Prompt",
            options=MODE_LABELS,
            index=MODE_LABELS.index(current_label),
            key=widget_key(panel_key, "prompt_mode"),
        )
    prompt_mode = MODE_LABEL_TO_KEY[prompt_mode_label]

    # Reset state when persona or mode changes.
    changed = (
        state["persona_id"] != selected_persona.id
        or state["prompt_mode"] != prompt_mode
    )
    if changed:
        state["messages"] = []
        state["past_key_values"] = None
        state["persona_id"] = selected_persona.id
        state["prompt_mode"] = prompt_mode
        _clear_chat_ui_state(
            widget_key(panel_key, "custom_prompt"),
            widget_key(panel_key, "show_all"),
        )

    # ── System prompt ────────────────────────────────────────────────────────
    active_system_prompt = resolve_system_prompt(
        persona=selected_persona, mode=prompt_mode
    )
    custom_prompt_key = widget_key(panel_key, "custom_prompt")
    if prompt_mode != "empty":
        if custom_prompt_key not in st.session_state:
            st.session_state[custom_prompt_key] = active_system_prompt
        with st.expander("Edit prompt", expanded=False):
            active_system_prompt = (
                st.text_area(
                    "prompt",
                    key=custom_prompt_key,
                    height=150,
                    label_visibility="collapsed",
                )
                or None
            )

    export_success_message: str | None = None
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button(
            "Export chat",
            key=widget_key(panel_key, "export_chat"),
            use_container_width=True,
        ):
            export_path = save_chat_export(
                model_name=model_name,
                dataset_source=dataset_source,
                persona_id=selected_persona.id,
                persona_name=getattr(selected_persona, "name", None),
                panel_label=side,
                prompt_mode=prompt_mode,
                system_prompt=active_system_prompt,
                messages=state["messages"],
                generation=_generation_dict(gen_kwargs, advanced_generation),
            )
            export_success_message = f"Saved chat export to {export_path}"
    with action_col2:
        if st.button(
            "Reset chat",
            key=widget_key(panel_key, "reset"),
            use_container_width=True,
            type="secondary",
        ):
            state["messages"] = []
            state["past_key_values"] = None
            _clear_chat_ui_state(
                widget_key(panel_key, "custom_prompt"),
                widget_key(panel_key, "show_all"),
            )
            st.rerun()

    if export_success_message:
        st.success(export_success_message)

    # ── Message history ──────────────────────────────────────────────────────
    show_all_key = widget_key(panel_key, "show_all")
    messages = state["messages"]
    if len(messages) > _VISIBLE_MESSAGE_COUNT and not st.session_state.get(
        show_all_key, False
    ):
        hidden_count = len(messages) - _VISIBLE_MESSAGE_COUNT
        if st.button(
            f"Show earlier ({hidden_count} hidden)",
            key=widget_key(panel_key, "show_all_btn"),
        ):
            st.session_state[show_all_key] = True
            st.rerun()
        visible = messages[-_VISIBLE_MESSAGE_COUNT:]
    else:
        visible = messages

    chat_log = st.container()
    with chat_log:
        for msg in visible:
            _render_chat_message(msg)

    return {
        "panel_key": panel_key,
        "state": state,
        "active_system_prompt": active_system_prompt,
        "selected_persona": selected_persona,
        "chat_log": chat_log,
    }


def _generate_for_panel(
    panel: dict,
    model,
    remote: bool,
    gen_kwargs: dict,
) -> ChatReply:
    """Run generate_chat_reply for one compare panel. Thread-safe."""
    messages = []
    if panel["active_system_prompt"]:
        messages.append({"role": "system", "content": panel["active_system_prompt"]})
    messages.extend(panel["state"]["messages"])

    ctx = nullcontext() if remote else _model_lock
    with ctx:
        return generate_chat_reply(
            model=model,
            messages=messages,
            remote=remote,
            past_key_values=panel["state"]["past_key_values"],
            **gen_kwargs,
        )


def _render_compare_mode(
    remote: bool,
    model_name: str,
    context_key: str,
    dataset_source: str,
    personas: list,
    gen_kwargs: dict,
    advanced_generation: bool,
) -> None:
    """Render the full side-by-side comparison UI."""
    left_col, right_col = st.columns(2)

    with left_col:
        left = _render_compare_panel(
            "left",
            context_key,
            personas,
            remote,
            model_name,
            dataset_source,
            gen_kwargs,
            advanced_generation,
        )
    with right_col:
        right = _render_compare_panel(
            "right",
            context_key,
            personas,
            remote,
            model_name,
            dataset_source,
            gen_kwargs,
            advanced_generation,
        )

    user_prompt = st.chat_input(
        "Ask both...",
        key=widget_key(context_key, "cmp_input"),
    )
    if not user_prompt:
        return

    model = cached_model(model_name=model_name, remote=remote)
    panels = [(left, left_col), (right, right_col)]

    for panel, col in panels:
        panel["state"]["messages"].append({"role": "user", "content": user_prompt})
        with col:
            with panel["chat_log"]:
                _render_chat_message({"role": "user", "content": user_prompt})

    # Generate both responses in parallel (remote: truly concurrent; local: serialised via lock).
    with st.spinner("Generating..."):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_generate_for_panel, panel, model, remote, gen_kwargs)
                for panel, col in panels
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(exc)

    for (panel, col), result in zip(panels, results):
        if isinstance(result, Exception):
            with col:
                with panel["chat_log"]:
                    st.error(f"Generation failed: {result}")
            panel["state"]["messages"].pop()
            continue

        panel["state"]["messages"].append({"role": "assistant", "content": result.text})
        panel["state"]["past_key_values"] = (
            result.past_key_values if not remote else None
        )
        with col:
            with panel["chat_log"]:
                _render_chat_message({"role": "assistant", "content": result.text})


# ── Main tab entry point ───────────────────────────────────────────────────────


def render_chat_tab(remote: bool, model_name: str, dataset_source: str) -> None:
    """Render the chat tab."""

    st.title("Chat")

    context_key = chat_session_key(model_name, dataset_source)
    chat_state = get_chat_state(model_name, remote, dataset_source)
    try:
        dataset, dataset_status = load_dataset(dataset_source)
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

    # ── Generation settings ───────────────────────────────────────────────────
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
        prompt_mode_label = st.selectbox(
            "Prompt",
            options=MODE_LABELS,
            index=MODE_LABELS.index(current_mode_label),
            key=prompt_mode_select_key,
        )
        prompt_mode = MODE_LABEL_TO_KEY[prompt_mode_label]

    active_system_prompt = resolve_system_prompt(
        persona=selected_persona,
        mode=prompt_mode,
    )

    chat_input_key = widget_key(context_key, "chat_input")
    show_all_key = widget_key(context_key, "show_all_messages")
    custom_prompt_key = widget_key(context_key, "custom_system_prompt")
    pending_key = widget_key(context_key, "pending_prompt")
    export_success_message: str | None = None

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Reset chat", use_container_width=True, type="secondary"):
            reset_chat_state(model_name, remote, dataset_source)
            _clear_chat_ui_state(
                chat_input_key,
                show_all_key,
                custom_prompt_key,
                pending_key,
            )
            st.rerun()
    with action_col2:
        if st.button("Export chat", use_container_width=True):
            export_path = save_chat_export(
                model_name=model_name,
                dataset_source=dataset_source,
                persona_id=selected_persona.id,
                persona_name=getattr(selected_persona, "name", None),
                prompt_mode=prompt_mode,
                system_prompt=active_system_prompt,
                messages=chat_state["messages"],
                generation=_generation_dict(gen_kwargs, advanced_generation),
            )
            export_success_message = f"Saved chat export to {export_path}"

    if export_success_message:
        st.success(export_success_message)

    changed_context = (
        chat_state["persona_id"] != selected_persona.id
        or chat_state["prompt_mode"] != prompt_mode
    )
    if changed_context:
        had_history = bool(chat_state["messages"])
        chat_state["persona_id"] = selected_persona.id
        chat_state["prompt_mode"] = prompt_mode
        reset_chat_state(model_name, remote, dataset_source)
        _clear_chat_ui_state(
            chat_input_key,
            show_all_key,
            custom_prompt_key,
            pending_key,
        )
        if had_history:
            st.info("Chat history reset because the persona or system prompt changed.")

    chat_log = st.container()

    with chat_log:
        # System prompt as first item in conversation — collapsed by default, editable.
        if prompt_mode != "empty":
            if custom_prompt_key not in st.session_state:
                st.session_state[custom_prompt_key] = active_system_prompt
            with st.expander("Edit prompt", expanded=False):
                active_system_prompt = (
                    st.text_area(
                        "Prompt",
                        key=custom_prompt_key,
                        height=200,
                        label_visibility="collapsed",
                    )
                    or None
                )

        # Collapse older messages, show only the most recent ones.
        messages = chat_state["messages"]
        if len(messages) > _VISIBLE_MESSAGE_COUNT and not st.session_state.get(
            show_all_key, False
        ):
            hidden_count = len(messages) - _VISIBLE_MESSAGE_COUNT
            if st.button(
                f"Show earlier messages ({hidden_count} hidden)",
                key=widget_key(context_key, "show_all_btn"),
            ):
                st.session_state[show_all_key] = True
                st.rerun()
            visible_messages = messages[-_VISIBLE_MESSAGE_COUNT:]
        else:
            visible_messages = messages

        for message in visible_messages:
            _render_chat_message(message)

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

    messages = []
    if active_system_prompt:
        messages.append({"role": "system", "content": active_system_prompt})
    messages.extend(chat_state["messages"])

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
