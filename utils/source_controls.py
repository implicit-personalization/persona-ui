from __future__ import annotations

from pathlib import Path

import streamlit as st
from persona_data.environment import get_artifacts_dir
from persona_vectors.extraction import MaskStrategy

from utils.analysis_sources import (
    DEFAULT_COMPARE_MODEL,
    DEFAULT_HUB_REPO,
    SOURCE_HUB,
    SOURCE_LOCAL,
    SOURCES,
    Store,
    activation_store_cached,
    hub_models_by_mask_strategy,
    local_model_matches,
    local_model_options_cached,
)
from utils.helpers import widget_key
from utils.selection_controls import remembered_segmented_control

_SHARED_SOURCE_KEY = "source:last_source"
_SHARED_HUB_REPO_KEY = "source:hub_repo"
_SHARED_HUB_MODEL_KEY = "source:hub_model"
_SHARED_LOCAL_ROOT_KEY = "source:local_root"
_SHARED_LOCAL_MODEL_KEY = "source:local_model"


def render_source_select(
    *,
    widget_scope: str,
    last_source_key: str | None = None,
) -> str:
    key = widget_key(widget_scope, "source")
    if last_source_key is not None and last_source_key not in st.session_state:
        shared_source = st.session_state.get(_SHARED_SOURCE_KEY)
        if shared_source is not None:
            st.session_state[last_source_key] = shared_source
    selected = remembered_segmented_control(
        "Source",
        options=SOURCES,
        key=key,
        remember_key=last_source_key or _SHARED_SOURCE_KEY,
        default=SOURCE_HUB,
        label_visibility="collapsed",
    )
    st.session_state[_SHARED_SOURCE_KEY] = selected
    if last_source_key is not None:
        st.session_state[last_source_key] = selected
    return selected


def _render_hub_model_select(
    *,
    state_prefix: str,
    widget_scope: str,
    repo_id: str,
    mask_strategy: MaskStrategy,
    model_label: str,
    fallback_help: str,
    selection_help: str,
) -> str:
    fallback_key = f"{state_prefix}:hub_model_fallback"
    fallback_model = st.session_state.get(
        fallback_key,
        st.session_state.get(_SHARED_HUB_MODEL_KEY, DEFAULT_COMPARE_MODEL),
    )
    try:
        models_by_strategy = hub_models_by_mask_strategy(repo_id)
    except Exception as exc:
        st.warning(f"Could not load Hub configs for `{repo_id}`: {exc}")
        model = st.text_input(
            model_label,
            value=fallback_model,
            key=fallback_key,
            help=fallback_help,
        )
        st.session_state[_SHARED_HUB_MODEL_KEY] = model
        return model

    model_options = models_by_strategy.get(mask_strategy, [])
    if not model_options:
        st.warning(
            f"No Hub vector configs found for `{mask_strategy.value}` in `{repo_id}`."
        )
        model = st.text_input(
            model_label,
            value=fallback_model,
            key=fallback_key,
            help=fallback_help,
        )
        st.session_state[_SHARED_HUB_MODEL_KEY] = model
        return model

    select_key = widget_key(widget_scope, "hub_model", repo_id, mask_strategy.value)
    previous_model = st.session_state.get(
        select_key,
        st.session_state.get(_SHARED_HUB_MODEL_KEY, fallback_model),
    )
    default_model = (
        previous_model if previous_model in model_options else model_options[0]
    )
    selected = st.selectbox(
        model_label,
        options=model_options,
        index=model_options.index(default_model),
        key=select_key,
        help=selection_help,
    )
    st.session_state[fallback_key] = selected
    st.session_state[_SHARED_HUB_MODEL_KEY] = selected
    return selected


def _render_local_model_select(
    *,
    state_prefix: str,
    artifacts_root: str,
    mask_strategy: MaskStrategy,
    allow_custom_toggle: bool,
    model_label: str,
) -> str:
    fallback_key = f"{state_prefix}:local_model"
    fallback_model = st.session_state.get(
        fallback_key,
        st.session_state.get(_SHARED_LOCAL_MODEL_KEY, DEFAULT_COMPARE_MODEL),
    )
    model_options = local_model_options_cached(artifacts_root, mask_strategy.value)
    if not model_options:
        model = st.text_input(model_label, value=fallback_model, key=fallback_key)
        st.session_state[_SHARED_LOCAL_MODEL_KEY] = model
        return model

    if allow_custom_toggle:
        custom = st.toggle(
            "Custom local model",
            value=False,
            key=f"{state_prefix}:local_model_custom_enabled",
            help="Enter a model id/path manually instead of choosing from activation directories.",
        )
        if custom:
            model = st.text_input("Local model", value=fallback_model, key=fallback_key)
            st.session_state[_SHARED_LOCAL_MODEL_KEY] = model
            return model

    select_key = f"{state_prefix}:local_model_select"
    previous_model = st.session_state.get(
        select_key,
        st.session_state.get(_SHARED_LOCAL_MODEL_KEY, fallback_model),
    )
    if not any(local_model_matches(previous_model, option) for option in model_options):
        previous_model = fallback_model
    default_model = next(
        (
            option
            for option in model_options
            if local_model_matches(option, previous_model)
        ),
        model_options[0],
    )
    selected = st.selectbox(
        model_label,
        options=model_options,
        index=model_options.index(default_model),
        key=select_key,
        help="Models discovered under the selected artifacts root.",
    )
    st.session_state[fallback_key] = selected
    st.session_state[_SHARED_LOCAL_MODEL_KEY] = selected
    return selected


def render_store_select(
    source: str,
    mask_strategy: MaskStrategy,
    *,
    state_prefix: str,
    widget_scope: str,
    artifacts_root_key: str,
    model_label: str = "Model",
    local_model_label: str = "Model",
    allow_custom_local_model: bool = False,
    repo_help: str | None = None,
    fallback_help: str = "Model id to use if Hub config discovery is unavailable.",
) -> Store:
    if source == SOURCE_HUB:
        repo_key = f"{state_prefix}:hub_repo"
        repo = st.text_input(
            "Hub repo",
            value=st.session_state.get(
                repo_key,
                st.session_state.get(_SHARED_HUB_REPO_KEY, DEFAULT_HUB_REPO),
            ),
            key=repo_key,
            help=repo_help,
        )
        st.session_state[_SHARED_HUB_REPO_KEY] = repo
        model_name = _render_hub_model_select(
            state_prefix=state_prefix,
            widget_scope=widget_scope,
            repo_id=repo,
            mask_strategy=mask_strategy,
            model_label=model_label,
            fallback_help=fallback_help,
            selection_help="Models with vectors in the selected Hub repo and mask strategy.",
        )
        return activation_store_cached(
            SOURCE_HUB, repo, model_name, mask_strategy.value
        )

    root = st.text_input(
        "Artifacts root",
        value=st.session_state.get(
            _SHARED_LOCAL_ROOT_KEY,
            str(get_artifacts_dir() / "activations"),
        ),
        key=artifacts_root_key,
    )
    root = str(Path(root).expanduser())
    st.session_state[_SHARED_LOCAL_ROOT_KEY] = root
    model_name = _render_local_model_select(
        state_prefix=state_prefix,
        artifacts_root=root,
        mask_strategy=mask_strategy,
        allow_custom_toggle=allow_custom_local_model,
        model_label=local_model_label,
    )
    return activation_store_cached(SOURCE_LOCAL, root, model_name, mask_strategy.value)
