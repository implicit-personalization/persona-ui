from typing import Any, NamedTuple

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


class ComparePanel(NamedTuple):
    side: str
    state: dict[str, object]
    log: Any
    prompt: str | None
    persona: PersonaData
    prompt_key: str
    edit_key: str
    pending_key: str


def _reset_compare_panel(panel: ComparePanel) -> None:
    reset_chat_context_state(
        panel.state,
        panel.persona.id,
        panel.state["prompt_mode"],
        panel.prompt_key,
        panel.pending_key,
    )
    st.session_state.pop(panel.edit_key, None)


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

    def render_panel(side: str) -> ComparePanel:
        panel_key = widget_key(context_key, f"cmp_{side}")
        if panel_key not in st.session_state:
            st.session_state[panel_key] = default_chat_state()
        state = st.session_state[panel_key]

        prompt_key = widget_key(panel_key, "custom_prompt")
        edit_key = widget_key(panel_key, "edit_idx")
        pending_key = widget_key(panel_key, "pending_regen")

        # Carry over persona / prompt selections across model or remote switches.
        persist_persona_key = f"chat:last_cmp_{side}_persona"
        persist_prompt_key = f"chat:last_cmp_{side}_prompt"
        if state["persona_id"] is None:
            state["persona_id"] = st.session_state.get(persist_persona_key)
            state["prompt_mode"] = st.session_state.get(persist_prompt_key, "templated")

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
                state, selected_persona.id, prompt_mode, prompt_key, pending_key
            )
            st.session_state.pop(edit_key, None)

        active_system_prompt = resolve_system_prompt(
            persona=selected_persona, mode=prompt_mode
        )

        chat_log = st.container()
        with chat_log:
            active_system_prompt = render_system_prompt(
                prompt_key, prompt_mode, active_system_prompt
            )
        return ComparePanel(
            side=side,
            state=state,
            log=chat_log,
            prompt=active_system_prompt,
            persona=selected_persona,
            prompt_key=prompt_key,
            edit_key=edit_key,
            pending_key=pending_key,
        )

    left_col, right_col = st.columns(2)
    with left_col:
        left = render_panel("left")
    with right_col:
        right = render_panel("right")
    panels: list[ComparePanel] = [left, right]

    # Handle per-panel regeneration triggered by message edits
    regen_panels = [p for p in panels if st.session_state.pop(p.pending_key, False)]
    if regen_panels:
        model = _get_model()

        results: list[ChatReply | Exception] = []
        with st.spinner("Regenerating..."):
            for panel in regen_panels:
                try:
                    results.append(
                        _generate_panel_reply(
                            model=model,
                            remote=remote,
                            panel_state=panel.state,
                            panel_prompt=panel.prompt,
                            gen_kwargs=gen_kwargs,
                        )
                    )
                except Exception as exc:
                    results.append(exc)

        for panel, result in zip(regen_panels, results):
            if isinstance(result, Exception):
                with panel.log:
                    st.error(f"Generation failed: {result}")
                panel.state["messages"].pop()
                continue
            panel.state["messages"].append(
                {"role": "assistant", "content": result.text}
            )
            panel.state["past_key_values"] = (
                result.past_key_values if not remote else None
            )
        st.rerun()

    # Recompute contrast for assistant messages that were edited in place.
    if contrast_enabled:
        pending_edits: list[tuple[int, int]] = [
            (panel_idx, msg_idx)
            for panel_idx, panel in enumerate(panels)
            for msg_idx, msg in enumerate(panel.state["messages"])
            if msg.get("_needs_contrast") and msg.get("role") == "assistant"
        ]
        if pending_edits:
            model = _get_model()
            label_a = persona_label(left.persona)
            label_b = persona_label(right.persona)
            with st.spinner("Recomputing token contrast…"):
                for panel_idx, msg_idx in pending_edits:
                    panel = panels[panel_idx]
                    msg = panel.state["messages"][msg_idx]
                    if msg_idx >= len(left.state["messages"]) or msg_idx >= len(
                        right.state["messages"]
                    ):
                        msg.pop("_needs_contrast", None)
                        continue
                    context_a = build_chat_messages(
                        left.prompt, left.state["messages"][:msg_idx]
                    )
                    context_b = build_chat_messages(
                        right.prompt, right.state["messages"][:msg_idx]
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

    for panel in panels:
        render_chat_window(
            chat_log=panel.log,
            messages=panel.state["messages"],
            chat_state=panel.state,
            edit_key=panel.edit_key,
            pending_key=panel.pending_key,
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
                for panel in panels:
                    save_chat_export(
                        model_name=model_name,
                        dataset_source=dataset_source,
                        persona_id=panel.persona.id,
                        persona_name=getattr(panel.persona, "name", None),
                        prompt_mode=panel.state["prompt_mode"],
                        system_prompt=panel.prompt,
                        messages=panel.state["messages"],
                        generation=generation_dict(gen_kwargs),
                        panel_label=panel.side,
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
                for panel in panels:
                    if st.button(
                        f"Reset {panel.side}",
                        key=widget_key(context_key, f"cmp_reset_{panel.side}"),
                    ):
                        _reset_compare_panel(panel)
                        st.session_state[reset_menu_nonce_key] += 1
                        st.rerun()
                if st.button(
                    "Reset both",
                    key=widget_key(context_key, "cmp_reset_both"),
                    type="primary",
                ):
                    for panel in panels:
                        _reset_compare_panel(panel)
                    st.session_state[reset_menu_nonce_key] += 1
                    st.rerun()

    user_prompt = st.chat_input(
        "Ask both...",
        key=widget_key(context_key, "cmp_input"),
    )

    if not user_prompt:
        return

    model = cached_model(model_name=model_name, remote=remote)

    for panel in panels:
        panel.state["messages"].append({"role": "user", "content": user_prompt})
        with panel.log:
            render_chat_message({"role": "user", "content": user_prompt})

    # Snapshot contexts before the new assistant turn is appended (needed for contrast).
    pre_gen_contexts = [
        build_chat_messages(panel.prompt, panel.state["messages"]) for panel in panels
    ]

    results: list[ChatReply | Exception] = []
    with st.spinner("Generating..."):
        # Sequential generation keeps both panels using model/session state safely.
        for panel in panels:
            try:
                results.append(
                    _generate_panel_reply(
                        model=model,
                        remote=remote,
                        panel_state=panel.state,
                        panel_prompt=panel.prompt,
                        gen_kwargs=gen_kwargs,
                    )
                )
            except Exception as exc:
                results.append(exc)

    valid_results: list[ChatReply | None] = []
    for panel, result in zip(panels, results):
        if isinstance(result, Exception):
            with panel.log:
                st.error(f"Generation failed: {result}")
            panel.state["messages"].pop()
            valid_results.append(None)
            continue

        panel.state["messages"].append({"role": "assistant", "content": result.text})
        panel.state["past_key_values"] = result.past_key_values if not remote else None
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
                    label_a=persona_label(left.persona),
                    label_b=persona_label(right.persona),
                    remote=remote,
                )
                if tc_a is not None:
                    left.state["messages"][-1]["_contrast"] = tc_a
                if tc_b is not None:
                    right.state["messages"][-1]["_contrast"] = tc_b
            except Exception as exc:
                st.warning(f"Token contrast failed: {exc}")

    # Rerun so the newly appended turns are redrawn through the editable history
    # renderer instead of only appearing in the one-off generation pass.
    st.rerun()
