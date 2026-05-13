from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import streamlit as st

from state import ChatState, default_chat_state, reset_chat_context_state
from tabs.chat_shared import (
    generate_chat_reply_result,
    hydrate_chat_state,
    render_chat_selection,
)
from utils.chat import ChatReply, build_chat_messages, resolve_system_prompt
from utils.chat_export import save_chat_export
from utils.contrast import compute_contrast, compute_contrast_pair
from utils.helpers import persona_label, session_key, widget_key
from utils.runtime import cached_model

from .chat_ui import (
    GenerationConfig,
    render_chat_message,
    render_chat_window,
    render_system_prompt,
)

if TYPE_CHECKING:
    from nnterp import StandardizedTransformer
    from persona_data.synth_persona import PersonaData


@dataclass(frozen=True)
class ComparePanel:
    side: str
    state: ChatState
    log: Any
    prompt: str | None
    persona: PersonaData
    prompt_key: str
    edit_key: str
    pending_key: str


def _get_compare_state(context_key: str, side: str) -> tuple[str, ChatState]:
    panel_key = widget_key(context_key, f"cmp_{side}")
    if panel_key not in st.session_state:
        st.session_state[panel_key] = default_chat_state()
    return panel_key, st.session_state[panel_key]


def _reset_compare_panel(panel: ComparePanel) -> None:
    reset_chat_context_state(
        panel.state,
        panel.persona.id,
        panel.state["prompt_mode"],
        panel.prompt_key,
        panel.pending_key,
    )
    st.session_state.pop(panel.edit_key, None)


def _render_compare_panel(
    *,
    context_key: str,
    side: str,
    personas: list[PersonaData],
) -> ComparePanel:
    panel_key, state = _get_compare_state(context_key, side)

    prompt_key = widget_key(panel_key, "custom_prompt")
    edit_key = widget_key(panel_key, "edit_idx")
    pending_key = widget_key(panel_key, "pending_regen")

    persist_persona_key = session_key("chat", f"last_cmp_{side}_persona")
    persist_prompt_key = session_key("chat", f"last_cmp_{side}_prompt")
    hydrate_chat_state(
        state,
        persisted_persona_key=persist_persona_key,
        persisted_prompt_key=persist_prompt_key,
    )

    selection = render_chat_selection(
        personas,
        state["persona_id"],
        state["prompt_mode"],
        widget_key(panel_key, "persona"),
        widget_key(panel_key, "prompt_mode"),
        persisted_persona_key=persist_persona_key,
        persisted_prompt_key=persist_prompt_key,
    )
    selected_persona = selection.persona
    prompt_mode = selection.prompt_mode
    changed = selection.changed

    if changed:
        reset_chat_context_state(
            state,
            selected_persona.id,
            prompt_mode,
            prompt_key,
            pending_key,
        )
        st.session_state.pop(edit_key, None)

    active_system_prompt = resolve_system_prompt(
        persona=selected_persona,
        mode=prompt_mode,
    )

    chat_log = st.container()
    with chat_log:
        active_system_prompt = render_system_prompt(
            prompt_key,
            prompt_mode,
            active_system_prompt,
            on_save=lambda: reset_chat_context_state(
                state,
                selected_persona.id,
                prompt_mode,
                pending_key,
            ),
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


def _generate_panels(
    *,
    model: StandardizedTransformer,
    remote: bool,
    panels: list[ComparePanel],
    generation: GenerationConfig,
    spinner_label: str,
) -> list[ChatReply | Exception]:
    results: list[ChatReply | Exception] = []
    with st.spinner(spinner_label):
        for panel in panels:
            reply, error = generate_chat_reply_result(
                model=model,
                messages=build_chat_messages(panel.prompt, panel.state["messages"]),
                remote=remote,
                generation=generation,
            )
            results.append(reply if error is None else error)
    return results


def _apply_panel_results(
    *,
    panels: list[ComparePanel],
    results: list[ChatReply | Exception],
    rollback_user_on_error: bool,
) -> list[ChatReply | None]:
    valid_results: list[ChatReply | None] = []
    for panel, result in zip(panels, results, strict=True):
        if isinstance(result, Exception):
            with panel.log:
                st.error(f"Generation failed: {result}")
            if rollback_user_on_error and panel.state["messages"]:
                panel.state["messages"].pop()
            valid_results.append(None)
            continue

        panel.state["messages"].append({"role": "assistant", "content": result.text})
        valid_results.append(result)
    return valid_results


def _pending_contrast_edits(panels: list[ComparePanel]) -> list[tuple[int, int]]:
    return [
        (panel_idx, msg_idx)
        for panel_idx, panel in enumerate(panels)
        for msg_idx, msg in enumerate(panel.state["messages"])
        if msg.get("_needs_contrast") and msg.get("role") == "assistant"
    ]


def _recompute_pending_contrast(
    *,
    model: StandardizedTransformer,
    remote: bool,
    panels: list[ComparePanel],
) -> bool:
    pending_edits = _pending_contrast_edits(panels)
    if not pending_edits:
        return False

    left, right = panels
    label_a = persona_label(left.persona)
    label_b = persona_label(right.persona)
    with st.spinner("Recomputing token contrast..."):
        for panel_idx, msg_idx in pending_edits:
            panel = panels[panel_idx]
            msg = panel.state["messages"][msg_idx]
            if msg_idx >= len(left.state["messages"]) or msg_idx >= len(
                right.state["messages"]
            ):
                msg.pop("_needs_contrast", None)
                continue

            context_a = build_chat_messages(
                left.prompt,
                left.state["messages"][:msg_idx],
            )
            context_b = build_chat_messages(
                right.prompt,
                right.state["messages"][:msg_idx],
            )
            try:
                response_ids = model.tokenizer(
                    msg["content"],
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids[0]
                contrast = compute_contrast(
                    model=model,
                    context_a=context_a,
                    context_b=context_b,
                    response_ids=response_ids,
                    label_a=label_a,
                    label_b=label_b,
                    remote=remote,
                )
                if contrast is not None:
                    msg["_contrast"] = contrast
            except Exception as exc:
                st.warning(f"Token contrast recompute failed: {exc}")
            msg.pop("_needs_contrast", None)
    return True


def _render_compare_history(
    *,
    panels: list[ComparePanel],
    contrast_enabled: bool,
) -> None:
    for panel in panels:
        render_chat_window(
            chat_log=panel.log,
            messages=panel.state["messages"],
            edit_key=panel.edit_key,
            pending_key=panel.pending_key,
            show_contrast=contrast_enabled,
            edit_column_ratio=(10, 1),
        )


def _render_compare_footer(
    *,
    context_key: str,
    model_name: str,
    dataset_source: str,
    panels: list[ComparePanel],
    generation: GenerationConfig,
) -> None:
    # Bumping this nonce after a reset gives the popover a fresh widget key,
    # which forces Streamlit to re-mount it closed (popovers don't auto-close
    # on click).
    reset_menu_nonce_key = widget_key(context_key, "cmp_reset_menu_nonce")
    if reset_menu_nonce_key not in st.session_state:
        st.session_state[reset_menu_nonce_key] = 0

    footer = st.container()
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
                        generation=generation.to_export_dict(),
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


def _append_user_prompt(panels: list[ComparePanel], user_prompt: str) -> None:
    for panel in panels:
        panel.state["messages"].append({"role": "user", "content": user_prompt})
        with panel.log:
            render_chat_message({"role": "user", "content": user_prompt})


def _compute_new_reply_contrast(
    *,
    model: StandardizedTransformer,
    remote: bool,
    panels: list[ComparePanel],
    pre_gen_contexts: list[list[dict[str, str]]],
    results: list[ChatReply | None],
) -> None:
    if len(results) != 2 or any(
        result is None or result.generated_ids is None for result in results
    ):
        return

    left, right = panels
    with st.spinner("Computing token contrast..."):
        try:
            left_contrast, right_contrast = compute_contrast_pair(
                model=model,
                context_a=pre_gen_contexts[0],
                context_b=pre_gen_contexts[1],
                response_ids_a=results[0].generated_ids,
                response_ids_b=results[1].generated_ids,
                label_a=persona_label(left.persona),
                label_b=persona_label(right.persona),
                remote=remote,
            )
            if left_contrast is not None:
                left.state["messages"][-1]["_contrast"] = left_contrast
            if right_contrast is not None:
                right.state["messages"][-1]["_contrast"] = right_contrast
        except Exception as exc:
            st.warning(f"Token contrast failed: {exc}")


def _render_compare_panels(
    *,
    context_key: str,
    personas: list[PersonaData],
) -> list[ComparePanel]:
    left_col, right_col = st.columns(2)
    with left_col:
        left = _render_compare_panel(
            context_key=context_key,
            side="left",
            personas=personas,
        )
    with right_col:
        right = _render_compare_panel(
            context_key=context_key,
            side="right",
            personas=personas,
        )
    return [left, right]


def render_compare_mode(
    remote: bool,
    model_name: str,
    context_key: str,
    dataset_source: str,
    personas: list[PersonaData],
    generation: GenerationConfig,
    *,
    contrast_enabled: bool,
) -> None:
    """Render the full side-by-side comparison UI."""

    panels = _render_compare_panels(context_key=context_key, personas=personas)

    regen_panels = [
        panel for panel in panels if st.session_state.pop(panel.pending_key, False)
    ]
    if regen_panels:
        results = _generate_panels(
            model=cached_model(model_name=model_name),
            remote=remote,
            panels=regen_panels,
            generation=generation,
            spinner_label="Regenerating...",
        )
        _apply_panel_results(
            panels=regen_panels,
            results=results,
            rollback_user_on_error=False,
        )
        st.rerun()

    if contrast_enabled and _recompute_pending_contrast(
        model=cached_model(model_name=model_name),
        remote=remote,
        panels=panels,
    ):
        st.rerun()

    _render_compare_history(panels=panels, contrast_enabled=contrast_enabled)
    _render_compare_footer(
        context_key=context_key,
        model_name=model_name,
        dataset_source=dataset_source,
        panels=panels,
        generation=generation,
    )

    user_prompt = st.chat_input(
        "Ask both...",
        key=widget_key(context_key, "cmp_input"),
    )
    if not user_prompt:
        return

    _append_user_prompt(panels, user_prompt)
    pre_gen_contexts = [
        build_chat_messages(panel.prompt, panel.state["messages"]) for panel in panels
    ]
    model = cached_model(model_name=model_name)
    results = _generate_panels(
        model=model,
        remote=remote,
        panels=panels,
        generation=generation,
        spinner_label="Generating...",
    )
    valid_results = _apply_panel_results(
        panels=panels,
        results=results,
        rollback_user_on_error=True,
    )
    if contrast_enabled:
        _compute_new_reply_contrast(
            model=model,
            remote=remote,
            panels=panels,
            pre_gen_contexts=pre_gen_contexts,
            results=valid_results,
        )

    st.rerun()
