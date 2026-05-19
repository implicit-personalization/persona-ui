import gc
from copy import deepcopy

import plotly.graph_objects as go
import streamlit as st
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import plot_persona_dendrogram
from plotly.subplots import make_subplots

from tabs.analysis._shared import (
    _load_persona_options,
    _load_variant_vectors,
    _plotly_chart,
    _render_layer_frame_controls,
    _render_persona_select_controls,
    _render_save_buttons,
    _select_artifact_personas,
)
from tabs.analysis._state import (
    _DEFAULT_PERSONA_LIMITS,
    _MAX_PERSONA_COUNTS,
    _clear_old_figure_states,
    _filename,
    _persona_names_state_key,
    _personas_empty_message,
    _store_figure_state,
)
from utils.analysis_sources import (
    Store,
    available_variants,
    store_cache_parts,
    store_id,
    store_layers_cached,
)
from utils.helpers import personas_fingerprint, prompt_variant_label, widget_key

_LAST_DENDRO_PERSONAS_KEY = "analysis:last_personas:dendro"
_DENDRO_LINKAGE_OPTIONS = ["ward", "complete", "average", "single"]


def _comparison_dendrogram_figure(
    fig_a: go.Figure,
    fig_b: go.Figure,
    *,
    title_a: str,
    title_b: str,
) -> go.Figure:
    """Merge two layered dendrograms so one slider drives both panels."""
    combined = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(title_a, title_b),
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )
    for trace in fig_a.data:
        combined.add_trace(deepcopy(trace), row=1, col=1)
    for trace in fig_b.data:
        combined.add_trace(deepcopy(trace), row=1, col=2)

    frames: list[go.Frame] = []
    for frame_a, frame_b in zip(fig_a.frames, fig_b.frames, strict=True):
        right_data = []
        for trace in frame_b.data:
            copied = deepcopy(trace)
            copied.update(xaxis="x2", yaxis="y2")
            right_data.append(copied)
        frame_xaxis = frame_a.layout.xaxis.to_plotly_json()
        frame_xaxis2 = frame_b.layout.xaxis.to_plotly_json()
        frame_xaxis2["matches"] = None
        frame_xaxis2["anchor"] = "y2"
        frame_yaxis = frame_a.layout.yaxis.to_plotly_json()
        frame_yaxis2 = frame_b.layout.yaxis.to_plotly_json()
        frame_yaxis2["matches"] = "y"
        frame_yaxis2["anchor"] = "x2"
        frames.append(
            go.Frame(
                name=frame_a.name,
                data=[*deepcopy(frame_a.data), *right_data],
                layout={
                    "title": {"text": f"Dendrogram comparison - Layer {frame_a.name}"},
                    "xaxis": frame_xaxis,
                    "xaxis2": frame_xaxis2,
                    "yaxis": frame_yaxis,
                    "yaxis2": frame_yaxis2,
                },
            )
        )

    y_ranges = [
        fig_a.layout.yaxis.range,
        fig_b.layout.yaxis.range,
    ]
    max_y = max(float(axis_range[1]) for axis_range in y_ranges if axis_range)
    first_layer = fig_a.frames[0].name if fig_a.frames else ""
    combined.frames = frames
    combined.update_layout(
        title={
            "text": f"Dendrogram comparison - Layer {first_layer}",
            "font": {"size": 24},
            "y": 0.98,
            "yanchor": "top",
        },
        template="plotly_white",
        height=750,
        margin=dict(t=140, b=260),
        updatemenus=fig_a.layout.updatemenus,
        sliders=fig_a.layout.sliders,
    )
    left_xaxis = fig_a.layout.xaxis.to_plotly_json()
    right_xaxis = fig_b.layout.xaxis.to_plotly_json()
    right_xaxis["matches"] = None
    right_xaxis["anchor"] = "y2"
    combined.update_layout(xaxis=left_xaxis, xaxis2=right_xaxis)
    combined.update_xaxes(tickangle=-45, automargin=True)
    combined.update_yaxes(
        title_text=fig_a.layout.yaxis.title.text,
        range=[0.0, max_y],
        automargin=True,
    )
    return combined


def _render_dendrogram_analysis(
    store: Store,
    mask_strategy: MaskStrategy,
) -> None:
    variants = available_variants(store, mask_strategy)
    if not variants:
        st.info("No variants with saved vectors for this model.")
        return

    with st.expander("Variant selection", expanded=True):
        col1, col2 = st.columns(2)
        default_a = "biography" if "biography" in variants else variants[0]
        default_b_idx = (
            variants.index("templated")
            if "templated" in variants
            else min(1, len(variants) - 1)
        )
        with col1:
            variant_a = st.selectbox(
                "Variant A",
                options=variants,
                index=variants.index(default_a),
                format_func=prompt_variant_label,
                key=widget_key("load", "dendro_variant_a", store_id(store)),
            )
        with col2:
            variant_b = st.selectbox(
                "Variant B",
                options=variants,
                index=default_b_idx,
                format_func=prompt_variant_label,
                key=widget_key("load", "dendro_variant_b", store_id(store)),
            )

    shared_variants = list(dict.fromkeys([variant_a, variant_b]))

    select_specific = st.toggle(
        "Select specific personas",
        value=False,
        key=widget_key("load", "dendro_select_mode", store_id(store)),
        help="Search and select specific personas instead of using the first N.",
    )

    if select_specific:
        empty_message = _personas_empty_message(shared_variants)
        options = _load_persona_options(
            store,
            shared_variants,
            mask_strategy,
            empty_message=empty_message,
        )
        if options is None:
            st.session_state.pop(
                _persona_names_state_key(f"dendro:{store_id(store)}"), None
            )
            return
        persona_ids = _render_persona_select_controls(
            options,
            widget_scope=f"dendro:{store_id(store)}",
            max_selections=_MAX_PERSONA_COUNTS["dendro"],
        )
        if not persona_ids:
            return
    else:
        persona_ids = _select_artifact_personas(
            store,
            shared_variants,
            mask_strategy,
            widget_scope=f"dendro:{store_id(store)}",
            remember_key=_LAST_DENDRO_PERSONAS_KEY,
            default_count_limit=_DEFAULT_PERSONA_LIMITS["dendro"],
            max_count_limit=_MAX_PERSONA_COUNTS["dendro"],
        )
        if not persona_ids:
            return

    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        layered_mode = st.toggle(
            "Per-layer animated",
            value=False,
            key=widget_key("load", "dendro_layered", store_id(store)),
            help="Animated dendrogram with one frame per layer instead of averaging all layers.",
        )
    with col_opts2:
        linkage = st.selectbox(
            "Linkage",
            options=_DENDRO_LINKAGE_OPTIONS,
            index=0,
            key=widget_key("load", "dendro_linkage", store_id(store)),
        )

    selected_layers: list[int] | None = None
    if layered_mode:
        source, location, model_name = store_cache_parts(store)
        layer_options = store_layers_cached(
            source,
            location,
            model_name,
            mask_strategy.value,
            tuple(shared_variants),
            tuple(persona_ids),
        )
        if not layer_options:
            st.info("No shared layers are available for the selected personas.")
            return
        selected_layers = _render_layer_frame_controls(store, "dendro", layer_options)

    persona_key = personas_fingerprint(persona_ids)
    fig_key = widget_key(
        "load",
        "dendro_fig_state",
        store_id(store),
        store.model_name,
        mask_strategy.value,
        variant_a,
        variant_b,
        persona_key,
        str(layered_mode),
        linkage,
        "_".join(map(str, selected_layers or [])),
    )
    _clear_old_figure_states(fig_key)

    if st.button(
        "Generate dendrograms",
        type="primary",
        key=widget_key(
            "load", "dendro_btn", store_id(store), variant_a, variant_b, persona_key
        ),
    ):
        progress = st.progress(0, text="Loading first variant vectors…")
        try:
            progress.progress(15, text="Loading variant vectors…")
            by_variant = _load_variant_vectors(
                store,
                shared_variants,
                mask_strategy,
                persona_ids,
            )
            samples_a = by_variant[variant_a]
            progress.progress(40, text="Building first dendrogram…")
            fig_a = plot_persona_dendrogram(
                samples_a,
                layered=layered_mode,
                layers=selected_layers,
                linkage=linkage,
                title=f"Dendrogram — {prompt_variant_label(variant_a)}",
            )
            fig_a.update_layout(height=750)
            fig_b = None
            if variant_a != variant_b:
                progress.progress(60, text="Building second dendrogram…")
                samples_b = by_variant[variant_b]
                progress.progress(75, text="Building second dendrogram…")
                fig_b = plot_persona_dendrogram(
                    samples_b,
                    layered=layered_mode,
                    layers=selected_layers,
                    linkage=linkage,
                    title=f"Dendrogram — {prompt_variant_label(variant_b)}",
                )
                fig_b.update_layout(height=750)
                del samples_b
            del samples_a
            comparison_fig = None
            if fig_b is not None and layered_mode:
                comparison_fig = _comparison_dendrogram_figure(
                    fig_a,
                    fig_b,
                    title_a=prompt_variant_label(variant_a),
                    title_b=prompt_variant_label(variant_b),
                )
            progress.progress(90, text="Storing figure state…")
            _store_figure_state(
                fig_key,
                (
                    None if comparison_fig is not None else fig_a,
                    None if comparison_fig is not None else fig_b,
                    comparison_fig,
                    len(persona_ids),
                    variant_a,
                    variant_b,
                ),
            )
            progress.progress(100, text="Done.")
        except Exception as exc:
            st.error(f"Could not build dendrogram: {exc}")
            st.session_state.pop(fig_key, None)
        finally:
            gc.collect()
            progress.empty()

    if fig_key in st.session_state:
        saved = st.session_state[fig_key]
        fig_a, fig_b, comparison_fig, n_personas, va, vb = saved
        if comparison_fig is not None:
            _plotly_chart(comparison_fig)
        elif fig_b is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(prompt_variant_label(va))
                _plotly_chart(fig_a)
            with col_b:
                st.subheader(prompt_variant_label(vb))
                _plotly_chart(fig_b)
        else:
            _plotly_chart(fig_a)

        figs = (
            [comparison_fig]
            if comparison_fig is not None
            else [fig_a] + ([fig_b] if fig_b else [])
        )
        filenames = (
            [_filename("dendro_compare", store.model_name, mask_strategy.value, va, vb)]
            if comparison_fig is not None
            else [
                _filename("dendro", store.model_name, mask_strategy.value, va),
                *(
                    [_filename("dendro", store.model_name, mask_strategy.value, vb)]
                    if fig_b
                    else []
                ),
            ]
        )
        _render_save_buttons(figs, filenames, "dendro")
        st.success(f"Generated dendrogram(s) for {n_personas} persona(s).")
