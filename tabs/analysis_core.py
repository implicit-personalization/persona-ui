import streamlit as st

from tabs.analysis._shared import _render_mask_strategy_select
from tabs.analysis._state import (
    _DEFAULT_PERSONA_LIMITS,
    _LAST_PROJECTION_DIMS_KEY,
    _LAST_SIMILARITY_PERSONAS_KEY,
    _LAST_SOURCE_KEY,
    _MAX_PERSONA_COUNTS,
)
from tabs.analysis.cosine import _render_cosine_similarity
from tabs.analysis.dendrogram import _render_dendrogram_analysis
from tabs.analysis.layered import _render_layered_figure_analysis
from utils.helpers import (
    ANALYSIS_HELP_TEXT,
    ANALYSIS_MODES,
    prompt_variant_label,
    widget_key,
)
from utils.source_controls import render_source_select, render_store_select


def render_analysis_tab() -> None:
    """Render the analysis tab."""

    st.title("Analysis")
    st.caption(
        "Analyse persona vectors by cosine similarity, PCA, UMAP, Isomap, or hierarchical clustering."
    )

    source = render_source_select(widget_scope="load", last_source_key=_LAST_SOURCE_KEY)

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
        store = render_store_select(
            source,
            mask_strategy,
            state_prefix="analysis",
            widget_scope="load",
            artifacts_root_key="analysis:artifacts_root",
            model_label="Hub model",
            local_model_label="Local model",
            allow_custom_local_model=True,
            repo_help="Hugging Face dataset published by `scripts/push_to_hf.py`.",
            fallback_help="Analysis-only model id to use if Hub config discovery is unavailable.",
        )

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
            max_count_limit=_MAX_PERSONA_COUNTS["similarity"],
            allow_specific_personas=True,
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
