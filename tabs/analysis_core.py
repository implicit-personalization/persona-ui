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
from utils.helpers import (
    ANALYSIS_HELP_TEXT,
    ANALYSIS_MODES,
    prompt_variant_label,
    widget_key,
)

from tabs.analysis._shared import _render_mask_strategy_select
from tabs.analysis._state import (
    _DEFAULT_PERSONA_LIMITS,
    _LAST_PROJECTION_DIMS_KEY,
    _LAST_SIMILARITY_PERSONAS_KEY,
    _LAST_SOURCE_KEY,
)
from tabs.analysis.cosine import _render_cosine_similarity
from tabs.analysis.dendrogram import _render_dendrogram_analysis
from tabs.analysis.layered import _render_layered_figure_analysis


def _render_source_select() -> str:
    last_source = st.session_state.get(_LAST_SOURCE_KEY, SOURCE_HUB)
    source = st.segmented_control(
        "Source",
        options=SOURCES,
        default=last_source if last_source in SOURCES else SOURCE_HUB,
        key=widget_key("load", "source"),
        label_visibility="collapsed",
    )
    if source is None:
        source = SOURCE_HUB
    st.session_state[_LAST_SOURCE_KEY] = source
    return source


def _render_hub_model_select(
    repo_id: str,
    mask_strategy: MaskStrategy,
) -> str:
    fallback_model = st.session_state.get(
        "analysis:hub_model_fallback",
        DEFAULT_COMPARE_MODEL,
    )
    try:
        models_by_strategy = hub_models_by_mask_strategy(repo_id)
    except Exception as exc:
        st.warning(f"Could not load Hub configs for `{repo_id}`: {exc}")
        return st.text_input(
            "Hub model",
            value=fallback_model,
            key="analysis:hub_model_fallback",
            help="Analysis-only model id to use if Hub config discovery is unavailable.",
        )

    model_options = models_by_strategy.get(mask_strategy, [])
    if not model_options:
        st.warning(
            f"No Hub vector configs found for `{mask_strategy.value}` in `{repo_id}`."
        )
        return st.text_input(
            "Hub model",
            value=fallback_model,
            key="analysis:hub_model_fallback",
            help="Analysis-only model id to use for this Hub repo.",
        )

    previous_model = st.session_state.get(
        widget_key("load", "hub_model", repo_id, mask_strategy.value),
        fallback_model,
    )
    default_model = (
        previous_model if previous_model in model_options else model_options[0]
    )

    return st.selectbox(
        "Hub model",
        options=model_options,
        index=model_options.index(default_model),
        key=widget_key("load", "hub_model", repo_id, mask_strategy.value),
        help="Models with vectors in the selected Hub repo and mask strategy.",
    )


def _render_local_model_select(
    artifacts_root: str,
    mask_strategy: MaskStrategy,
) -> str:
    fallback_model = st.session_state.get("analysis:local_model", DEFAULT_COMPARE_MODEL)
    model_options = local_model_options_cached(artifacts_root, mask_strategy.value)
    if not model_options:
        return st.text_input(
            "Local model",
            value=fallback_model,
            key="analysis:local_model",
            help="Analysis-only local model id or path.",
        )

    custom = st.toggle(
        "Custom local model",
        value=False,
        key="analysis:local_model_custom_enabled",
        help="Enter a model id/path manually instead of choosing from activation directories.",
    )
    if custom:
        return st.text_input(
            "Local model",
            value=fallback_model,
            key="analysis:local_model",
            help="Analysis-only local model id or path.",
        )

    previous_model = st.session_state.get("analysis:local_model_select", fallback_model)
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
        "Local model",
        options=model_options,
        index=model_options.index(default_model),
        key="analysis:local_model_select",
        help="Models discovered under the selected artifacts root.",
    )
    st.session_state["analysis:local_model"] = selected
    return selected


def _build_store(source: str, mask_strategy: MaskStrategy) -> Store:
    if source == SOURCE_HUB:
        repo = st.text_input(
            "Hub repo",
            value=st.session_state.get("analysis:hub_repo", DEFAULT_HUB_REPO),
            key="analysis:hub_repo",
            help="Hugging Face dataset published by `scripts/push_to_hf.py`.",
        )
        hub_model_name = _render_hub_model_select(repo, mask_strategy)
        return activation_store_cached(
            SOURCE_HUB,
            repo,
            hub_model_name,
            mask_strategy.value,
        )
    artifacts_root = st.text_input(
        "Artifacts root",
        value=str(get_artifacts_dir() / "activations"),
        key="analysis:artifacts_root",
    )
    artifacts_root = str(Path(artifacts_root).expanduser())
    local_model_name = _render_local_model_select(artifacts_root, mask_strategy)
    return activation_store_cached(
        SOURCE_LOCAL,
        artifacts_root,
        local_model_name,
        mask_strategy.value,
    )


def render_analysis_tab() -> None:
    """Render the analysis tab."""

    st.title("Analysis")
    st.caption(
        "Analyse persona vectors by cosine similarity, PCA, UMAP, Isomap, or hierarchical clustering."
    )

    source = _render_source_select()

    analysis_mode = st.segmented_control(
        "Analysis mode",
        options=ANALYSIS_MODES,
        default=ANALYSIS_MODES[0],
        key=widget_key("load", "analysis_mode"),
        label_visibility="collapsed",
    )
    if analysis_mode is None:
        analysis_mode = ANALYSIS_MODES[0]
    st.caption(ANALYSIS_HELP_TEXT[analysis_mode])

    with st.expander("Source settings", expanded=True):
        mask_strategy = _render_mask_strategy_select(analysis_mode)
        store = _build_store(source, mask_strategy)

    if analysis_mode == "Cosine similarity":
        _render_cosine_similarity(store, mask_strategy)
        return
    if analysis_mode == "Similarity matrix":
        _render_layered_figure_analysis(
            store,
            mask_strategy,
            scope="similarity_matrix",
            figure_kind="similarity",
            button_label="Generate similarity matrix",
            title_fn=lambda v: (
                f"Centered similarity - {prompt_variant_label(v)} - persona vectors"
            ),
            include_pair_trajectories=True,
            remember_key=_LAST_SIMILARITY_PERSONAS_KEY,
            default_count_limit=_DEFAULT_PERSONA_LIMITS["similarity"],
        )
        return

    if analysis_mode == "Dendrogram":
        _render_dendrogram_analysis(store, mask_strategy)
        return

    dim_options = ["2D", "3D"]
    dim_key = widget_key("load", "projection_dims", analysis_mode)
    remembered_dim = st.session_state.get(
        dim_key,
        st.session_state.get(_LAST_PROJECTION_DIMS_KEY, "2D"),
    )
    if remembered_dim not in dim_options:
        remembered_dim = "2D"
    dimension_choice = st.segmented_control(
        "Projection dimensions",
        options=dim_options,
        default=remembered_dim,
        key=dim_key,
        label_visibility="collapsed",
    )
    if dimension_choice is not None:
        st.session_state[_LAST_PROJECTION_DIMS_KEY] = dimension_choice
    n_components = 3 if dimension_choice == "3D" else 2
    dim_suffix = "" if n_components == 2 else " (3D)"
    _render_layered_figure_analysis(
        store,
        mask_strategy,
        scope=f"{analysis_mode.lower()}{'_3d' if n_components == 3 else ''}",
        figure_kind=analysis_mode.lower(),
        button_label=f"Generate {analysis_mode}{dim_suffix} projection",
        title_fn=lambda v: (
            f"{analysis_mode}{dim_suffix} - {prompt_variant_label(v)} - persona vectors"
        ),
        n_components=n_components,
        default_count_limit=_DEFAULT_PERSONA_LIMITS[analysis_mode.lower()],
    )
