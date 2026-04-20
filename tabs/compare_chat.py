import streamlit as st
from nnterp import StandardizedTransformer
from persona_data.synth_persona import PersonaData

from state import default_chat_state, reset_chat_context_state
from utils.chat import ChatReply, generate_chat_reply, resolve_system_prompt
from utils.chat_export import save_chat_export
from utils.contrast import compute_contrast, compute_contrast_pair
from utils.helpers import persona_label, widget_key
from utils.runtime import cached_model

from .chat import (
    build_chat_messages,
    generation_dict,
    render_chat_message,
    render_chat_window,
    render_persona_prompt_controls,
    render_system_prompt,
)


def _panel_state(panel_key: str) -> dict[str, object]:
    """Get or initialise compare-panel chat state stored in session_state."""
    if panel_key not in st.session_state:
        st.session_state[panel_key] = default_chat_state()
    return st.session_state[panel_key]


def _reset_compare_panel(
    panel_state: dict,
    edit_key: str,
    persona_id: str,
    prompt_mode: str,
    *ui_keys: str,
) -> None:
    reset_chat_context_state(panel_state, persona_id, prompt_mode, *ui_keys)
    st.session_state.pop(edit_key, None)


def _generate_panel_reply(
    *,
    model: StandardizedTransformer,
    remote: bool,
    panel_state: dict[str, object],
    panel_prompt: str | None,
    gen_kwargs: dict,
) -> ChatReply:
    return generate_chat_reply(
        model=model,
        messages=build_chat_messages(panel_prompt, panel_state["messages"]),
        remote=remote,
        past_key_values=panel_state["past_key_values"],
        **gen_kwargs,
    )


def render_compare_mode(
    remote: bool,
    model_name: str,
    context_key: str,
    dataset_source: str,
    personas: list[PersonaData],
    gen_kwargs: dict,
    advanced_generation: bool,
) -> None:
    """Render the full side-by-side comparison UI."""
    model: StandardizedTransformer | None = None

    def _get_model() -> StandardizedTransformer:
        nonlocal model
        if model is None:
            model = cached_model(model_name=model_name, remote=remote)
        return model

    contrast_key = widget_key(context_key, "token_contrast")
    contrast_enabled = st.toggle(
        "Token contrast",
        value=False,
        key=contrast_key,
        help=(
            "Color each generated token by how characteristic it is of each persona. "
            "Red = more likely under the left persona, blue = more likely under the right. "
            "Requires four extra forward passes after each turn (batched into one "
            "remote session when running on NDIF)."
        ),
    )

    left_col, right_col = st.columns(2)
    left_panel_key = widget_key(context_key, "cmp_left")
    right_panel_key = widget_key(context_key, "cmp_right")
    left_prompt_key = widget_key(left_panel_key, "custom_prompt")
    right_prompt_key = widget_key(right_panel_key, "custom_prompt")
    left_edit_key = widget_key(left_panel_key, "edit_idx")
    right_edit_key = widget_key(right_panel_key, "edit_idx")
    left_pending_key = widget_key(left_panel_key, "pending_regen")
    right_pending_key = widget_key(right_panel_key, "pending_regen")

    def render_panel(side: str) -> tuple[dict, object, str | None, str, PersonaData]:
        panel_key = widget_key(context_key, f"cmp_{side}")
        state = _panel_state(panel_key)

        # Carry over persona / prompt selections across model or remote switches.
        persist_persona_key = f"chat:last_cmp_{side}_persona"
        persist_prompt_key = f"chat:last_cmp_{side}_prompt"
        if state["persona_id"] is None:
            state["persona_id"] = st.session_state.get(persist_persona_key)
            state["prompt_mode"] = st.session_state.get(persist_prompt_key, "templated")

        prompt_key = widget_key(panel_key, "custom_prompt")
        edit_key = widget_key(panel_key, "edit_idx")
        pending_regen_key = widget_key(panel_key, "pending_regen")

        selected_persona, prompt_mode, changed = render_persona_prompt_controls(
            personas,
            state["persona_id"],
            state["prompt_mode"],
            widget_key(panel_key, "persona"),
            widget_key(panel_key, "prompt_mode"),
        )
        st.session_state[persist_persona_key] = selected_persona.id
        st.session_state[persist_prompt_key] = prompt_mode

        if changed:
            reset_chat_context_state(
                state,
                selected_persona.id,
                prompt_mode,
                prompt_key,
                pending_regen_key,
            )
            st.session_state.pop(edit_key, None)

        active_system_prompt = resolve_system_prompt(
            persona=selected_persona, mode=prompt_mode
        )

        chat_log = st.container()
        with chat_log:
            active_system_prompt = render_system_prompt(
                prompt_key,
                prompt_mode,
                active_system_prompt,
            )
        return (
            state,
            chat_log,
            active_system_prompt,
            pending_regen_key,
            selected_persona,
        )

    with left_col:
        left_state, left_log, left_prompt, left_pending, left_persona = render_panel(
            "left"
        )
    with right_col:
        right_state, right_log, right_prompt, right_pending, right_persona = (
            render_panel("right")
        )

    panels = [
        (
            left_state,
            left_log,
            left_prompt,
            left_pending,
            left_edit_key,
            left_persona,
        ),
        (
            right_state,
            right_log,
            right_prompt,
            right_pending,
            right_edit_key,
            right_persona,
        ),
    ]

    # Handle per-panel regeneration triggered by message edits
    regen_panels = [
        (panel_state, panel_log, panel_prompt)
        for panel_state, panel_log, panel_prompt, p_pending, _panel_edit_key, _ in panels
        if st.session_state.pop(p_pending, False)
    ]
    if regen_panels:
        model = _get_model()

        results: list[ChatReply | Exception] = []
        with st.spinner("Regenerating..."):
            for panel_state, _panel_log, panel_prompt in regen_panels:
                try:
                    results.append(
                        _generate_panel_reply(
                            model=model,
                            remote=remote,
                            panel_state=panel_state,
                            panel_prompt=panel_prompt,
                            gen_kwargs=gen_kwargs,
                        )
                    )
                except Exception as exc:
                    results.append(exc)

        for (panel_state, panel_log, _panel_prompt), result in zip(
            regen_panels, results
        ):
            if isinstance(result, Exception):
                with panel_log:
                    st.error(f"Generation failed: {result}")
                panel_state["messages"].pop()
                continue
            panel_state["messages"].append(
                {"role": "assistant", "content": result.text}
            )
            panel_state["past_key_values"] = (
                result.past_key_values if not remote else None
            )
        st.rerun()

    # Recompute contrast for assistant messages that were edited in place.
    if contrast_enabled:
        pending_edits: list[tuple[int, int]] = [
            (panel_idx, msg_idx)
            for panel_idx, (panel_state, *_rest) in enumerate(panels)
            for msg_idx, msg in enumerate(panel_state["messages"])
            if msg.get("_needs_contrast") and msg.get("role") == "assistant"
        ]
        if pending_edits:
            model = _get_model()
            label_a = persona_label(left_persona)
            label_b = persona_label(right_persona)
            with st.spinner("Recomputing token contrast…"):
                for panel_idx, msg_idx in pending_edits:
                    panel_state = panels[panel_idx][0]
                    msg = panel_state["messages"][msg_idx]
                    if msg_idx >= len(left_state["messages"]) or msg_idx >= len(
                        right_state["messages"]
                    ):
                        msg.pop("_needs_contrast", None)
                        continue
                    context_a = build_chat_messages(
                        left_prompt, left_state["messages"][:msg_idx]
                    )
                    context_b = build_chat_messages(
                        right_prompt, right_state["messages"][:msg_idx]
                    )
                    try:
                        response_ids = model.tokenizer(
                            msg["content"],
                            add_special_tokens=False,
                            return_tensors="pt",
                        ).input_ids[0]
                        tc = compute_contrast(
                            model=model,
                            context_a=context_a,
                            context_b=context_b,
                            response_ids=response_ids,
                            label_a=label_a,
                            label_b=label_b,
                            remote=remote,
                        )
                        if tc is not None:
                            msg["_contrast"] = tc
                    except Exception as exc:
                        st.warning(f"Token contrast recompute failed: {exc}")
                    msg.pop("_needs_contrast", None)
            st.rerun()

    for (
        panel_state,
        panel_log,
        _panel_prompt,
        panel_pending,
        panel_edit_key,
        _,
    ) in panels:
        render_chat_window(
            chat_log=panel_log,
            messages=panel_state["messages"],
            chat_state=panel_state,
            edit_key=panel_edit_key,
            pending_key=panel_pending,
            show_contrast=contrast_enabled,
            edit_column_ratio=(10, 1),
        )

    footer = st.container()
    reset_menu_nonce_key = widget_key(context_key, "cmp_reset_menu_nonce")
    if reset_menu_nonce_key not in st.session_state:
        st.session_state[reset_menu_nonce_key] = 0
    with footer:
        exp_col, rst_col, _spacer = st.columns([0.5, 0.5, 10], gap="xsmall")
        with exp_col:
            if st.button(
                "",
                icon=":material/download:",
                key=widget_key(context_key, "cmp_export"),
                help="Export both chats",
            ):
                for side, panel_state, panel_prompt, panel_persona in (
                    ("left", left_state, left_prompt, left_persona),
                    ("right", right_state, right_prompt, right_persona),
                ):
                    save_chat_export(
                        model_name=model_name,
                        dataset_source=dataset_source,
                        persona_id=panel_persona.id,
                        persona_name=getattr(panel_persona, "name", None),
                        prompt_mode=panel_state["prompt_mode"],
                        system_prompt=panel_prompt,
                        messages=panel_state["messages"],
                        generation=generation_dict(gen_kwargs, advanced_generation),
                        panel_label=side,
                    )
                st.toast("Exported", icon=":material/check:")
        with rst_col:
            popover_key = widget_key(
                context_key,
                "cmp_reset_menu",
                str(st.session_state[reset_menu_nonce_key]),
            )
            with st.popover(
                "",
                icon=":material/delete_sweep:",
                help="Reset chat",
                key=popover_key,
            ):
                if st.button(
                    "Reset left",
                    key=widget_key(context_key, "cmp_reset_left"),
                ):
                    _reset_compare_panel(
                        left_state,
                        left_edit_key,
                        left_persona.id,
                        left_state["prompt_mode"],
                        left_prompt_key,
                        left_pending_key,
                    )
                    st.session_state[reset_menu_nonce_key] += 1
                    st.rerun()
                if st.button(
                    "Reset right",
                    key=widget_key(context_key, "cmp_reset_right"),
                ):
                    _reset_compare_panel(
                        right_state,
                        right_edit_key,
                        right_persona.id,
                        right_state["prompt_mode"],
                        right_prompt_key,
                        right_pending_key,
                    )
                    st.session_state[reset_menu_nonce_key] += 1
                    st.rerun()
                if st.button(
                    "Reset both",
                    key=widget_key(context_key, "cmp_reset_both"),
                    type="primary",
                ):
                    _reset_compare_panel(
                        left_state,
                        left_edit_key,
                        left_persona.id,
                        left_state["prompt_mode"],
                        left_prompt_key,
                        left_pending_key,
                    )
                    _reset_compare_panel(
                        right_state,
                        right_edit_key,
                        right_persona.id,
                        right_state["prompt_mode"],
                        right_prompt_key,
                        right_pending_key,
                    )
                    st.session_state[reset_menu_nonce_key] += 1
                    st.rerun()

    user_prompt = st.chat_input(
        "Ask both...",
        key=widget_key(context_key, "cmp_input"),
    )

    if not user_prompt:
        return

    model = cached_model(model_name=model_name, remote=remote)

    for panel_state, panel_log, _panel_prompt, _p_pending, _panel_edit_key, _ in panels:
        panel_state["messages"].append({"role": "user", "content": user_prompt})
        with panel_log:
            render_chat_message({"role": "user", "content": user_prompt})

    # Snapshot contexts before the new assistant turn is appended (needed for contrast).
    pre_gen_contexts = [
        build_chat_messages(panel_prompt, panel_state["messages"])
        for panel_state, _panel_log, panel_prompt, _p_pending, _panel_edit_key, _ in panels
    ]

    results: list[ChatReply | Exception] = []
    with st.spinner("Generating..."):
        # Keep compare-mode generation sequential so both panels use the same
        # model/session state safely.
        for (
            panel_state,
            _panel_log,
            panel_prompt,
            _p_pending,
            _panel_edit_key,
            _,
        ) in panels:
            try:
                results.append(
                    _generate_panel_reply(
                        model=model,
                        remote=remote,
                        panel_state=panel_state,
                        panel_prompt=panel_prompt,
                        gen_kwargs=gen_kwargs,
                    )
                )
            except Exception as exc:
                results.append(exc)

    valid_results: list[ChatReply | None] = []
    for (
        panel_state,
        panel_log,
        _panel_prompt,
        _p_pending,
        _panel_edit_key,
        _,
    ), result in zip(panels, results):
        if isinstance(result, Exception):
            with panel_log:
                st.error(f"Generation failed: {result}")
            panel_state["messages"].pop()
            valid_results.append(None)
            continue

        panel_state["messages"].append({"role": "assistant", "content": result.text})
        panel_state["past_key_values"] = result.past_key_values if not remote else None
        valid_results.append(result)

    # Compute contrastive token coloring when both panels succeeded.
    if (
        contrast_enabled
        and len(valid_results) == 2
        and all(r is not None and r.generated_ids is not None for r in valid_results)
    ):
        with st.spinner("Computing token contrast…"):
            try:
                tc_a, tc_b = compute_contrast_pair(
                    model=model,
                    context_a=pre_gen_contexts[0],
                    context_b=pre_gen_contexts[1],
                    response_ids_a=valid_results[0].generated_ids,
                    response_ids_b=valid_results[1].generated_ids,
                    label_a=persona_label(left_persona),
                    label_b=persona_label(right_persona),
                    remote=remote,
                )
                if tc_a is not None:
                    left_state["messages"][-1]["_contrast"] = tc_a
                if tc_b is not None:
                    right_state["messages"][-1]["_contrast"] = tc_b
            except Exception as exc:
                st.warning(f"Token contrast failed: {exc}")

    # Rerun so the newly appended turns are redrawn through the editable history
    # renderer instead of only appearing in the one-off generation pass.
    st.rerun()
