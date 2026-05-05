import streamlit as st
import torch

from utils.chat import build_chat_messages
from utils.helpers import widget_key
from utils.probe_trace import ConversationTrace, trace_conversation, vectorize_token
from utils.probes import (
    DEFAULT_PROBE_REPO,
    LoadedProbe,
    ProbeRunResult,
    list_probe_files,
    load_probe,
    load_probe_from_bytes,
    parse_probe_filename,
)
from utils.runtime import cached_model


def _token_button_label(index: int, token: str) -> str:
    display = token.encode("unicode_escape").decode("ascii") or "<empty>"
    if len(display) > 18:
        display = display[:15] + "..."
    return f"{index}: {display}"


def _render_probe_results(result: ProbeRunResult, probe: LoadedProbe) -> None:
    top_k = min(5, int(result.probabilities.numel()))
    if top_k == 0:
        st.warning("Probe returned an empty output tensor.")
        return

    top_values, top_indices = torch.topk(result.probabilities, k=top_k)
    label = result.predicted_label or str(result.predicted_index)
    st.success(f"Prediction: `{label}` (index {result.predicted_index})")
    st.caption(
        f"Input dim {result.input_dim}; output dim {int(result.logits.numel())}; "
        f"probe type {probe.model_type}"
    )

    lines = []
    for rank, (idx, prob) in enumerate(
        zip(top_indices.tolist(), top_values.tolist(), strict=True),
        start=1,
    ):
        item_label = (
            probe.labels[idx]
            if 0 <= idx < len(probe.labels) and probe.labels[idx] is not None
            else str(idx)
        )
        logit = float(result.logits[idx].item())
        lines.append(
            f"- {rank}. `{item_label}` (index {idx}) - "
            f"prob `{prob:.4f}`, logit `{logit:.4f}`"
        )
    st.markdown("\n".join(lines))


def _load_probe_from_controls(context_key: str) -> LoadedProbe | None:
    source = st.radio(
        "Probe source",
        options=("Hugging Face repo", "Upload .pt"),
        horizontal=True,
        key=widget_key(context_key, "probe_source"),
    )

    if source == "Upload .pt":
        uploaded = st.file_uploader(
            "Probe file",
            type=["pt"],
            key=widget_key(context_key, "probe_upload"),
        )
        if uploaded is None:
            return None
        return load_probe_from_bytes(uploaded.name, uploaded.getvalue())

    repo_id = st.text_input(
        "Probe repo",
        value=DEFAULT_PROBE_REPO,
        key=widget_key(context_key, "probe_repo"),
    )
    if not repo_id.strip():
        return None

    probe_files = list_probe_files(repo_id.strip())
    if not probe_files:
        st.warning("No `.pt` probe files were found in that repo.")
        return None

    selected_file = st.selectbox(
        "Probe file",
        options=probe_files,
        format_func=lambda filename: parse_probe_filename(filename).label,
        key=widget_key(context_key, "probe_file"),
    )
    return load_probe(repo_id.strip(), selected_file)


def _render_token_buttons(trace: ConversationTrace, context_key: str) -> int:
    selected_key = widget_key(
        context_key,
        "probe_selected_token",
        trace.prompt_hash[:12],
    )
    selected = int(st.session_state.get(selected_key, trace.n_tokens - 1))
    selected = max(0, min(selected, trace.n_tokens - 1))

    window_size = st.slider(
        "Token window",
        min_value=8,
        max_value=min(96, max(8, trace.n_tokens)),
        value=min(32, max(8, trace.n_tokens)),
        step=8,
        key=widget_key(context_key, "probe_token_window", trace.prompt_hash[:12]),
    )
    center = st.slider(
        "Window center",
        min_value=0,
        max_value=trace.n_tokens - 1,
        value=selected,
        key=widget_key(context_key, "probe_token_center", trace.prompt_hash[:12]),
    )
    start = max(0, center - window_size // 2)
    end = min(trace.n_tokens, start + window_size)
    start = max(0, end - window_size)

    cols = st.columns(8)
    for offset, token_index in enumerate(range(start, end)):
        col = cols[offset % len(cols)]
        token = trace.tokens[token_index]
        if col.button(
            _token_button_label(token_index, token),
            key=widget_key(
                context_key,
                "probe_token",
                trace.prompt_hash[:12],
                str(token_index),
            ),
            type="primary" if token_index == selected else "secondary",
            help=token.encode("unicode_escape").decode("ascii"),
        ):
            selected = token_index
            st.session_state[selected_key] = token_index

    st.caption(
        f"Selected token {selected}: "
        f"`{trace.tokens[selected].encode('unicode_escape').decode('ascii')}`"
    )
    return selected


def _model_dimensions(model: object) -> tuple[int, int]:
    config = getattr(model, "config", None)
    hidden_size = getattr(model, "hidden_size", None) or getattr(
        config, "hidden_size", None
    )
    num_layers = (
        getattr(model, "num_layers", None)
        or getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None)
    )
    if hidden_size is None or num_layers is None:
        raise ValueError("Could not read hidden_size and num_layers from the model.")
    return int(hidden_size), int(num_layers)


def render_probe_inspector(
    *,
    context_key: str,
    model_name: str,
    remote: bool,
    active_system_prompt: str | None,
    chat_state: dict[str, object],
    enabled: bool,
) -> None:
    if not enabled:
        return

    with st.expander("Probe Inspector", expanded=False):
        if not chat_state["messages"]:
            st.info("Add at least one message before tracing probe activations.")
            return

        try:
            probe = _load_probe_from_controls(context_key)
        except Exception as exc:
            st.error(f"Could not load probe: {exc}")
            return
        if probe is None:
            return

        with st.spinner("Loading model metadata..."):
            model = cached_model(model_name=model_name, remote=remote)
        try:
            hidden_size, num_layers = _model_dimensions(model)
        except Exception as exc:
            st.error(str(exc))
            return

        layer = probe.layer
        if layer is None:
            layer = int(
                st.number_input(
                    "Layer",
                    min_value=0,
                    max_value=max(0, num_layers - 1),
                    value=min(15, max(0, num_layers - 1)),
                    step=1,
                    key=widget_key(context_key, "probe_layer"),
                )
            )

        location = probe.location
        if location is None:
            location = st.selectbox(
                "Activation location",
                options=("post_reasoning", "pre_reasoning"),
                key=widget_key(context_key, "probe_location"),
            )

        st.caption(
            f"Probe layer {layer}; {location}; input dim {probe.input_dim}; "
            f"model hidden size {hidden_size}"
        )
        if not 0 <= layer < num_layers:
            st.error(f"Probe layer {layer} is outside the model's {num_layers} layers.")
            return
        if probe.input_dim != hidden_size:
            st.warning(
                "This probe input dim does not match a single-token activation "
                "for the active model."
            )
            return

        trace_key = widget_key(context_key, "probe_trace_enabled")
        if st.button(
            "Trace conversation",
            key=widget_key(context_key, "probe_trace"),
            use_container_width=True,
        ):
            st.session_state[trace_key] = True
        if not st.session_state.get(trace_key, False):
            return

        messages = build_chat_messages(active_system_prompt, chat_state["messages"])
        with st.spinner("Tracing conversation..."):
            trace = trace_conversation(
                model=model,
                model_name=model_name,
                messages=messages,
                layer=layer,
                location=location,
                remote=remote,
            )

        st.caption(
            f"Cached {trace.n_tokens} tokens from layer {trace.layer}; "
            f"prompt hash `{trace.prompt_hash[:10]}`"
        )
        if trace.n_tokens == 0:
            st.warning("The traced conversation produced no tokens.")
            return

        selected_token = _render_token_buttons(trace, context_key)
        try:
            vector = vectorize_token(trace, token_index=selected_token)
            result = probe.run(vector.vector)
        except Exception as exc:
            st.error(f"Probe execution failed: {exc}")
            return

        st.caption(
            f"Vectorization {vector.mode}; token {vector.token_index}; "
            f"vector dim {int(vector.vector.shape[0])}"
        )
        _render_probe_results(result, probe)
