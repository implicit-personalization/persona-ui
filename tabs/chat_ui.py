from dataclasses import asdict, dataclass
from typing import Any

import streamlit as st
from persona_data.synth_persona import PersonaData

from utils.contrast import TokenContrast, render_contrast_html
from utils.helpers import (
    CHAT_PROMPT_MODE_LABEL_TO_KEY,
    CHAT_PROMPT_MODE_LABELS,
    VARIANT_LABELS,
    persona_label,
    widget_key,
)

GENERATION_DEFAULTS = {
    "max_new_tokens": 256,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.0,
}

_LAST_GEN_PREFIX = "chat:last_gen:"


def _persisted_key(context_key: str, name: str, default) -> str:
    """Per-context widget key, seeded from the last cross-context value."""
    last_key = f"{_LAST_GEN_PREFIX}{name}"
    key = widget_key(context_key, name)
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get(last_key, default)
    return key


def _remember(name: str, value) -> None:
    st.session_state[f"{_LAST_GEN_PREFIX}{name}"] = value


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    seed: int | None

    def to_generate_kwargs(self) -> dict[str, object]:
        return asdict(self)

    def to_export_dict(self) -> dict[str, object]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "use_sampling": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ChatTools:
    probe_enabled: bool
    compare_mode: bool
    token_contrast: bool


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
    suffix = (
        f" - {n_after} subsequent {'message' if n_after == 1 else 'messages'} will be cleared"
        if n_after > 0
        else ""
    )
    st.caption(f"**{role}**{suffix}")

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
                st.session_state[pending_key] = "regenerate_after_edit"
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


def render_advanced_settings(
    context_key: str,
    remote: bool,
    *,
    last_compare_mode_key: str,
    last_probe_enabled_key: str = "",
    last_token_contrast_key: str = "",
) -> tuple[GenerationConfig, ChatTools]:
    """Render the Advanced expander: tool toggles + generation settings."""
    with st.expander("Advanced", expanded=False):
        st.caption("Tools")

        compare_key = widget_key(context_key, "compare_mode")
        if compare_key not in st.session_state:
            st.session_state[compare_key] = st.session_state.get(
                last_compare_mode_key, False
            )

        probe_key = widget_key(context_key, "probe_enabled")
        if probe_key not in st.session_state:
            st.session_state[probe_key] = st.session_state.get(
                last_probe_enabled_key, False
            )

        token_contrast_key = widget_key(context_key, "token_contrast")
        if token_contrast_key not in st.session_state:
            st.session_state[token_contrast_key] = st.session_state.get(
                last_token_contrast_key, False
            )

        tools_col1, tools_col2, tools_col3 = st.columns(3)
        with tools_col1:
            probe_enabled = st.toggle(
                "Probe tools",
                key=probe_key,
                help="Trace chat activations and run compatible `.pt` probes on tapped tokens.",
            )
        with tools_col2:
            compare_mode = st.toggle(
                "Compare mode",
                key=compare_key,
                help="Side-by-side: send one message to two independent persona/prompt configurations.",
            )
        with tools_col3:
            token_contrast = st.toggle(
                "Token contrast",
                key=token_contrast_key,
                disabled=not compare_mode,
                help=(
                    "Color each generated token by how characteristic it is of each persona. "
                    "Red = more likely under the left persona, blue = more likely under the "
                    "right. Requires up to four extra scoring passes after each turn. "
                    "Available only in Compare mode."
                ),
            )
        st.session_state[last_compare_mode_key] = compare_mode
        if last_probe_enabled_key:
            st.session_state[last_probe_enabled_key] = probe_enabled
        if last_token_contrast_key:
            st.session_state[last_token_contrast_key] = token_contrast

        st.divider()
        st.caption("Generation")
        generation = _render_generation_fragment(context_key, remote)

    tools = ChatTools(
        probe_enabled=probe_enabled,
        compare_mode=compare_mode,
        token_contrast=token_contrast and compare_mode,
    )
    return generation, tools


@st.fragment
def _render_generation_fragment(context_key: str, remote: bool) -> GenerationConfig:
    """Render generation sliders inside a fragment so tweaks don't full-rerun."""
    config_col1, config_col2 = st.columns([2, 1])
    with config_col1:
        max_new_tokens = st.slider(
            "Max new tokens",
            min_value=16,
            max_value=512,
            step=16,
            key=_persisted_key(
                context_key, "max_new_tokens", GENERATION_DEFAULTS["max_new_tokens"]
            ),
        )
    with config_col2:
        repetition_penalty = st.slider(
            "Repetition penalty",
            min_value=0.5,
            max_value=2.0,
            step=0.05,
            key=_persisted_key(
                context_key,
                "repetition_penalty",
                GENERATION_DEFAULTS["repetition_penalty"],
            ),
        )

    use_sampling = st.checkbox(
        "Random sampling",
        key=_persisted_key(context_key, "use_sampling", False),
    )

    sampling_disabled = not use_sampling
    sampling_col1, sampling_col2, sampling_col3 = st.columns(3)
    with sampling_col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.01,
            max_value=2.0,
            step=0.01,
            disabled=sampling_disabled,
            key=_persisted_key(
                context_key, "temperature", GENERATION_DEFAULTS["temperature"]
            ),
        )
    with sampling_col2:
        top_p = st.slider(
            "Top-p",
            min_value=0.01,
            max_value=1.0,
            step=0.01,
            disabled=sampling_disabled,
            key=_persisted_key(context_key, "top_p", GENERATION_DEFAULTS["top_p"]),
        )
    with sampling_col3:
        top_k = st.slider(
            "Top-k (0 = off)",
            min_value=0,
            max_value=100,
            step=1,
            disabled=sampling_disabled,
            key=_persisted_key(context_key, "top_k", GENERATION_DEFAULTS["top_k"]),
        )

    seed_disabled = sampling_disabled or remote
    seed_enabled = st.checkbox(
        "Fix seed",
        disabled=seed_disabled,
        key=_persisted_key(context_key, "seed_enabled", False),
    )
    seed = None
    if seed_enabled:
        seed = int(
            st.number_input(
                "Seed",
                min_value=0,
                max_value=2_147_483_647,
                step=1,
                disabled=seed_disabled,
                key=_persisted_key(context_key, "seed", 0),
            )
        )

    if remote:
        st.caption("Seed is local-only and disabled for remote runs.")

    for name, value in (
        ("max_new_tokens", max_new_tokens),
        ("repetition_penalty", repetition_penalty),
        ("use_sampling", use_sampling),
        ("temperature", temperature),
        ("top_p", top_p),
        ("top_k", top_k),
        ("seed_enabled", seed_enabled),
    ):
        _remember(name, value)

    do_sample = bool(use_sampling)
    return GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        seed=seed if do_sample and seed is not None and not remote else None,
    )


def render_chat_message(
    message: dict[str, str],
    show_contrast: bool = False,
) -> None:
    if not message.get("content"):
        return
    contrast: TokenContrast | None = message.get("_contrast") if show_contrast else None
    with st.chat_message(message["role"]):
        if contrast is not None:
            st.html(render_contrast_html(contrast))
        else:
            st.markdown(message["content"])


def render_chat_window(
    *,
    chat_log: Any,
    messages: list[dict[str, str]],
    chat_state: dict[str, object],
    edit_key: str,
    pending_key: str,
    show_contrast: bool = False,
    edit_column_ratio: tuple[int, int] = (25, 1),
) -> None:
    with chat_log:
        for i, message in enumerate(messages):
            if not message.get("content"):
                continue
            msg_col, edit_col = st.columns(
                list(edit_column_ratio), gap="xsmall", vertical_alignment="center"
            )
            with msg_col:
                render_chat_message(message, show_contrast=show_contrast)
            with edit_col:
                if st.button(
                    "",
                    icon=":material/edit:",
                    key=f"{edit_key}_edit_{i}",
                    help="Edit",
                ):
                    _open_edit_dialog(
                        msg_index=i,
                        messages=messages,
                        chat_state=chat_state,
                        pending_key=pending_key,
                    )


def _assistant_first(personas: list[PersonaData]) -> list[PersonaData]:
    def is_assistant(persona: PersonaData) -> bool:
        persona_id = str(getattr(persona, "id", "")).strip().lower()
        persona_name = str(getattr(persona, "name", "")).strip().lower()
        return persona_id == "assistant" or persona_name == "assistant"

    return sorted(personas, key=lambda persona: 0 if is_assistant(persona) else 1)


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
        persona_options = _assistant_first(personas)
        selected_index = next(
            (i for i, p in enumerate(persona_options) if p.id == current_persona_id),
            0,
        )
        selected_persona = st.selectbox(
            "Persona",
            options=persona_options,
            index=selected_index,
            format_func=persona_label,
            key=persona_key,
        )
    with m_col:
        current_label = VARIANT_LABELS[current_prompt_mode]
        prompt_mode_label = st.selectbox(
            "Prompt",
            options=CHAT_PROMPT_MODE_LABELS,
            index=CHAT_PROMPT_MODE_LABELS.index(current_label),
            key=prompt_key,
        )
    prompt_mode = CHAT_PROMPT_MODE_LABEL_TO_KEY[prompt_mode_label]
    changed = (
        current_persona_id != selected_persona.id or current_prompt_mode != prompt_mode
    )
    return selected_persona, prompt_mode, changed
