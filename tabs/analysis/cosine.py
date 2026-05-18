from itertools import combinations

import streamlit as st
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import plot_layer_similarity

from utils.analysis_sources import Store, available_variants, store_id
from utils.helpers import personas_fingerprint, prompt_variant_label, widget_key

from tabs.analysis._shared import (
    _load_variant_vectors,
    _plotly_chart,
    _release_vector_memory,
    _render_save_buttons,
    _select_artifact_personas,
)
from tabs.analysis._state import (
    _LAST_COSINE_PERSONAS_KEY,
    CosineSelection,
    _clear_old_figure_states,
    _filename,
    _store_figure_state,
)


def _render_cosine_selection(
    store: Store,
    mask_strategy: MaskStrategy,
) -> CosineSelection | None:
    variants = available_variants(store, mask_strategy)
    if len(variants) < 2:
        st.info("Need at least two variants with saved vectors for cosine comparison.")
        return None

    with st.expander("Vector selection", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            variant_a = st.selectbox(
                "Variant A",
                options=variants,
                index=0,
                format_func=prompt_variant_label,
                key=widget_key("load", "variant_a", store_id(store)),
            )
        with col2:
            variant_b = st.selectbox(
                "Variant B",
                options=variants,
                index=min(1, len(variants) - 1),
                format_func=prompt_variant_label,
                key=widget_key("load", "variant_b", store_id(store)),
            )

        if variant_a == variant_b:
            st.warning("Choose two different variants to compare.")
            return None

        persona_ids = _select_artifact_personas(
            store,
            [variant_a, variant_b],
            mask_strategy,
            widget_scope=f"cosine:{store_id(store)}",
            remember_key=_LAST_COSINE_PERSONAS_KEY,
        )
    if not persona_ids:
        return None
    return CosineSelection(
        variants=variants,
        variant_a=variant_a,
        variant_b=variant_b,
        persona_ids=persona_ids,
        persona_key=personas_fingerprint(persona_ids),
    )


def _build_cosine_figures(
    store: Store,
    mask_strategy: MaskStrategy,
    selection: CosineSelection,
) -> tuple[object, object | None, int, int] | None:
    try:
        by_variant = _load_variant_vectors(
            store,
            selection.variants,
            mask_strategy,
            persona_ids=selection.persona_ids,
        )
        samples_a = by_variant[selection.variant_a]
        samples_b = by_variant[selection.variant_b]
    except Exception as exc:
        st.error(f"Could not load vectors: {exc}")
        return None

    labels = samples_a.labels
    display_traces = [
        (
            label,
            samples_a.vectors[index],
            samples_b.vectors[index],
        )
        for index, label in enumerate(labels)
    ]
    fig = plot_layer_similarity(
        display_traces,
        title=(
            f"{prompt_variant_label(selection.variant_a)} vs "
            f"{prompt_variant_label(selection.variant_b)}"
        ),
        show=False,
    )

    pair_traces = []
    pair_errors = []
    for left, right in combinations(selection.variants, 2):
        try:
            left_samples = by_variant[left]
            right_samples = by_variant[right]
            pair_traces.append(
                (
                    f"{prompt_variant_label(left)} vs {prompt_variant_label(right)}",
                    left_samples.vectors.mean(dim=0),
                    right_samples.vectors.mean(dim=0),
                )
            )
        except Exception as exc:
            pair_errors.append(f"{left} vs {right}: {exc}")
            continue

    for err in pair_errors:
        st.warning(f"Skipped pair trace: `{err}`")
    pair_fig = (
        plot_layer_similarity(
            pair_traces,
            title="Variant-pair cosine similarity averaged over selected personas",
            show=False,
        )
        if pair_traces
        else None
    )
    return fig, pair_fig, len(display_traces), len(pair_traces)


def _render_cosine_similarity(
    store: Store,
    mask_strategy: MaskStrategy,
) -> None:
    selection = _render_cosine_selection(store, mask_strategy)
    if selection is None:
        return

    cosine_fig_key = widget_key(
        "load",
        "cosine_fig_state",
        store_id(store),
        store.model_name,
        mask_strategy.value,
        selection.variant_a,
        selection.variant_b,
        selection.persona_key,
    )
    filename = _filename(
        "analysis",
        "cosine",
        store.model_name,
        mask_strategy.value,
        selection.variant_a,
        selection.variant_b,
    )
    pairs_filename = _filename(
        "analysis",
        "cosine_pairs",
        store.model_name,
        mask_strategy.value,
        "_".join(selection.variants),
    )
    _clear_old_figure_states(cosine_fig_key)

    if st.button(
        "Compare vectors",
        type="primary",
        key=widget_key(
            "load",
            "analysis_vectors",
            store_id(store),
            store.model_name,
            mask_strategy.value,
            selection.variant_a,
            selection.variant_b,
            selection.persona_key,
        ),
    ):
        progress = st.progress(0, text="Loading activation vectors…")
        try:
            progress.progress(15, text="Loading activation vectors…")
            figures = _build_cosine_figures(store, mask_strategy, selection)
            if figures is None:
                st.session_state.pop(cosine_fig_key, None)
                return
            progress.progress(90, text="Storing figure state…")
            _store_figure_state(cosine_fig_key, figures)
            progress.progress(100, text="Done.")
        finally:
            _release_vector_memory()
            progress.empty()

    if cosine_fig_key in st.session_state:
        fig, pair_fig, n_traces, n_pair_traces = st.session_state[cosine_fig_key]
        _plotly_chart(fig)
        figs = [fig]
        filenames = [filename]
        if pair_fig is not None:
            st.subheader("Variant pairs")
            _plotly_chart(pair_fig)
            figs.append(pair_fig)
            filenames.append(pairs_filename)
        _render_save_buttons(figs, filenames, "cosine")
        st.success(f"Loaded {n_traces} personas for cosine comparison.")
        if pair_fig is not None:
            st.caption(f"Generated {n_pair_traces} averaged variant-pair trace(s).")
