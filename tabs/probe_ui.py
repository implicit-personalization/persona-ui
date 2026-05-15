from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch

from utils.chat import build_chat_messages
from utils.helpers import session_key, widget_key
from utils.probe_overlay import (
    attach_overlays,
    build_classification_overlays,
    build_regression_overlays,
    clear_overlays,
)
from utils.probe_trace import ConversationTrace, trace_conversation
from utils.probes import (
    DEFAULT_LOCAL_PROBE_DIR,
    DEFAULT_PROBE_REPO,
    LoadedProbe,
    list_local_probe_files,
    list_probe_files,
    load_local_probe,
    load_probe,
    load_probe_from_bytes,
    model_probe_dir_name,
    parse_probe_filename,
)
from utils.runtime import cached_model

_LAST_SOURCE_KEY = session_key("probe", "last_source")
_LAST_LOCAL_FILE_KEY = session_key("probe", "last_local_file")
_LAST_HUB_FILE_KEY = session_key("probe", "last_hub_file")

_PROBE_SOURCES = ("Local artifact", "Hugging Face repo", "Upload .pt")


# ---------------------------------------------------------------------------
# Probe selection
# ---------------------------------------------------------------------------


def _probe_label(filename: str) -> str:
    metadata = parse_probe_filename(filename)
    prefix = f"{metadata.model_name} / " if metadata.model_name else ""
    return f"{prefix}{metadata.label}"


def _model_compatible_files(files: list[str], model_name: str) -> list[str]:
    model_dir = model_probe_dir_name(model_name)
    compatible = [
        filename
        for filename in files
        if Path(filename).parts and Path(filename).parts[0] == model_dir
    ]
    return compatible or files


def _default_file(files: list[str], remembered: str | None) -> str:
    if remembered and remembered in files:
        return remembered
    return files[0]


def _render_probe_selector(
    *, context_key: str, model_name: str
) -> LoadedProbe | None:
    """Inline source + file selector. Returns the loaded probe or None."""
    source_key = widget_key(context_key, "probe_source")
    if source_key not in st.session_state:
        st.session_state[source_key] = st.session_state.get(
            _LAST_SOURCE_KEY, _PROBE_SOURCES[0]
        )
    source = st.segmented_control(
        "Probe source",
        options=_PROBE_SOURCES,
        key=source_key,
        label_visibility="collapsed",
    )
    source = source or _PROBE_SOURCES[0]
    st.session_state[_LAST_SOURCE_KEY] = source

    if source == "Local artifact":
        return _render_local_probe(context_key=context_key, model_name=model_name)
    if source == "Hugging Face repo":
        return _render_hub_probe(context_key=context_key, model_name=model_name)
    return _render_upload_probe(context_key=context_key)


def _render_local_probe(
    *, context_key: str, model_name: str
) -> LoadedProbe | None:
    root_dir = st.text_input(
        "Probe directory",
        value=st.session_state.get(
            widget_key(context_key, "probe_local_dir"), DEFAULT_LOCAL_PROBE_DIR
        ),
        key=widget_key(context_key, "probe_local_dir"),
    )
    files = list_local_probe_files(root_dir.strip())
    if not files:
        st.warning("No probe files found in that directory.")
        return None
    files = _model_compatible_files(files, model_name)
    default = _default_file(files, st.session_state.get(_LAST_LOCAL_FILE_KEY))
    selected = st.selectbox(
        "Probe",
        options=files,
        index=files.index(default),
        format_func=_probe_label,
        key=widget_key(context_key, "probe_local_file"),
    )
    st.session_state[_LAST_LOCAL_FILE_KEY] = selected
    try:
        return load_local_probe(root_dir.strip(), selected)
    except Exception as exc:
        st.error(f"Could not load probe: {exc}")
        return None


def _render_hub_probe(
    *, context_key: str, model_name: str
) -> LoadedProbe | None:
    repo_id = st.text_input(
        "Probe repo",
        value=st.session_state.get(
            widget_key(context_key, "probe_repo"), DEFAULT_PROBE_REPO
        ),
        key=widget_key(context_key, "probe_repo"),
    )
    if not repo_id.strip():
        return None
    files = list_probe_files(repo_id.strip())
    if not files:
        st.warning("No probe files found in that repo.")
        return None
    files = _model_compatible_files(files, model_name)
    default = _default_file(files, st.session_state.get(_LAST_HUB_FILE_KEY))
    selected = st.selectbox(
        "Probe",
        options=files,
        index=files.index(default),
        format_func=_probe_label,
        key=widget_key(context_key, "probe_hub_file"),
    )
    st.session_state[_LAST_HUB_FILE_KEY] = selected
    try:
        return load_probe(repo_id.strip(), selected)
    except Exception as exc:
        st.error(f"Could not load probe: {exc}")
        return None


def _render_upload_probe(*, context_key: str) -> LoadedProbe | None:
    uploaded = st.file_uploader(
        "Upload probe (.pt)",
        type=["pt"],
        key=widget_key(context_key, "probe_upload"),
    )
    if uploaded is None:
        return None
    try:
        return load_probe_from_bytes(uploaded.name, uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not load probe: {exc}")
        return None


# ---------------------------------------------------------------------------
# Probe card + target validation
# ---------------------------------------------------------------------------


def _render_probe_card(probe: LoadedProbe) -> None:
    parts: list[str] = []
    if probe.attribute_name:
        parts.append(f"**{probe.attribute_name}**")
    parts.append(f"layer `{probe.layer if probe.layer is not None else '?'}`")
    parts.append(f"kind `{probe.model_type}`")
    if probe.feature_space:
        parts.append(f"`{probe.feature_space}`")
    if probe.location:
        parts.append(f"`{probe.location}`")
    classes = (
        ", ".join(label for label in probe.labels if label)
        or f"{len(probe.labels)} classes"
    )
    parts.append(f"classes: {classes}")
    st.markdown(" &nbsp;·&nbsp; ".join(parts))


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


def _resolve_target(
    *, probe: LoadedProbe, context_key: str, num_layers: int
) -> tuple[int, str]:
    layer = probe.layer
    if layer is None:
        layer = int(
            st.number_input(
                "Layer (probe did not specify one)",
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
            "Activation location (probe did not specify one)",
            options=("post_reasoning", "pre_reasoning"),
            key=widget_key(context_key, "probe_location"),
        )
    return layer, location


def _validate(
    *, probe: LoadedProbe, layer: int, num_layers: int, hidden_size: int
) -> bool:
    if not 0 <= layer < num_layers:
        st.error(f"Probe layer {layer} is outside the model's {num_layers} layers.")
        return False
    if probe.input_dim != hidden_size:
        st.warning(
            f"Probe input dim ({probe.input_dim}) does not match the model's hidden "
            f"size ({hidden_size}). Predictions will not be meaningful."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Cached batched probe forward
# ---------------------------------------------------------------------------


def _classification_predictions(
    probe: LoadedProbe, activations: torch.Tensor, cache_key: str
) -> tuple[torch.Tensor, torch.Tensor]:
    full_key = widget_key("probe_predictions", cache_key, str(id(probe)))
    cached = st.session_state.get(full_key)
    if cached is not None:
        return cached
    _, probs, predicted = probe.run_batch(activations)
    st.session_state[full_key] = (probs, predicted)
    return probs, predicted


def _regression_values(
    probe: LoadedProbe, activations: torch.Tensor, cache_key: str
) -> torch.Tensor:
    full_key = widget_key("probe_values", cache_key, str(id(probe)))
    cached = st.session_state.get(full_key)
    if cached is not None:
        return cached
    values = probe.predict_batch(activations)
    st.session_state[full_key] = values
    return values


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _has_assistant_message(messages: list[dict]) -> bool:
    return any(m.get("role") == "assistant" and m.get("content") for m in messages)


def _apply_overlays(
    *, probe: LoadedProbe, trace: ConversationTrace, messages: list[dict]
) -> bool:
    if probe.is_regression:
        values = _regression_values(probe, trace.activations, trace.cache_key)
        overlays = build_regression_overlays(
            trace=trace,
            values=values,
            labels=probe.labels,
            attribute_name=probe.attribute_name,
        )
    else:
        probs, predicted = _classification_predictions(
            probe, trace.activations, trace.cache_key
        )
        binary = probs.shape[1] == 1 or (
            probs.shape[1] == 2 and len(probe.labels) == 2
        )
        overlays = build_classification_overlays(
            trace=trace,
            probs=probs,
            predicted=predicted,
            labels=probe.labels,
            binary=binary,
            attribute_name=probe.attribute_name,
        )
    attach_overlays(messages, overlays)
    return bool(overlays)


def render_probe_inspector(
    *,
    context_key: str,
    model_name: str,
    remote: bool,
    active_system_prompt: str | None,
    chat_state: dict[str, object],
    enabled: bool,
) -> None:
    messages: list[dict] = chat_state["messages"]  # type: ignore[assignment]
    if not enabled:
        clear_overlays(messages)
        return

    status_key = widget_key(context_key, "probe_status")
    sig_key = widget_key(context_key, "probe_scored_sig")

    def _conversation_sig() -> int:
        return hash(
            tuple(
                (m.get("role"), m.get("content"))
                for m in messages
                if m.get("content")
            )
        )

    def _reset() -> None:
        clear_overlays(messages)
        st.session_state.pop(status_key, None)
        st.session_state.pop(sig_key, None)

    with st.expander("Probe", expanded=True):
        if not _has_assistant_message(messages):
            _reset()
            st.caption("Probe overlay shows up after the first assistant reply.")
            return

        probe = _render_probe_selector(
            context_key=context_key, model_name=model_name
        )
        if probe is None:
            _reset()
            return
        _render_probe_card(probe)

        model = cached_model(model_name=model_name)
        try:
            hidden_size, num_layers = _model_dimensions(model)
        except Exception as exc:
            _reset()
            st.error(str(exc))
            return

        layer, location = _resolve_target(
            probe=probe, context_key=context_key, num_layers=num_layers
        )
        if not _validate(
            probe=probe, layer=layer, num_layers=num_layers, hidden_size=hidden_size
        ):
            _reset()
            return

        # The probe scores via a separate forward pass over the whole
        # conversation, so it's fully decoupled from generation: pick or switch
        # probes any time and score on demand. Gate that pass behind a button
        # instead of re-running it on every Streamlit rerun. Overlays live on
        # the message dicts, so they persist across reruns until refreshed.
        run = st.button(
            "Run probe",
            type="primary",
            key=widget_key(context_key, "probe_run"),
            help="Score the current conversation with the selected probe.",
        )
        if not run:
            status = st.session_state.get(status_key)
            if not status:
                st.caption("Press **Run probe** to score the conversation.")
            elif st.session_state.get(sig_key) != _conversation_sig():
                # Conversation changed since it was scored: drop the now-stale
                # overlay so it can't paint over edited/new text.
                clear_overlays(messages)
                st.caption("Conversation changed — press **Run probe** to refresh.")
            else:
                st.caption(f"{status} · press **Run probe** to refresh.")
            return

        chat_messages = build_chat_messages(active_system_prompt, messages)
        with st.spinner("Tracing conversation..."):
            try:
                trace = trace_conversation(
                    model=model,
                    model_name=model_name,
                    messages=chat_messages,
                    layer=layer,
                    location=location,
                    remote=remote,
                )
            except Exception as exc:
                _reset()
                st.error(f"Trace failed: {exc}")
                return

        if not trace.assistant_spans:
            _reset()
            st.warning(
                "Could not locate assistant tokens in the traced sequence, so "
                "the overlay can't be aligned to message bodies."
            )
            return

        try:
            applied = _apply_overlays(probe=probe, trace=trace, messages=messages)
        except Exception as exc:
            _reset()
            st.error(f"Probe execution failed: {exc}")
            return

        if not applied:
            _reset()
            return

        n_body = sum(
            sum(1 for i in range(s, e) if not bool(trace.is_special[i].item()))
            for s, e in trace.assistant_spans
        )
        kind = "regression" if probe.is_regression else "classification"
        status = (
            f"{kind} · {len(trace.assistant_spans)} assistant message(s) · "
            f"{n_body} body tokens · layer {trace.layer} · {trace.location}"
        )
        st.session_state[status_key] = status
        st.session_state[sig_key] = _conversation_sig()
        st.caption(status)
