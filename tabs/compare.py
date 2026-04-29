from itertools import combinations

import streamlit as st
import torch
from persona_data.environment import get_artifacts_dir
from persona_data.prompts import BASELINE_PERSONA_ID
from persona_vectors.analysis import load_persona_mean_samples
from persona_vectors.artifacts import (
    PERSONA_VARIANTS,
    ActivationStore,
)
from persona_vectors.artifacts import list_layers as list_available_layers
from persona_vectors.artifacts import list_personas as list_available_personas
from persona_vectors.artifacts import load_mean_activations, load_persona_names
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import (
    build_layered_figure,
    build_pair_similarity_figure,
    plot_layer_similarity,
    save_plot_html,
    save_plot_png,
)

from utils.helpers import (
    ANALYSIS_HELP_TEXT,
    ANALYSIS_MODES,
    persona_display_label,
    prompt_variant_label,
    slugify,
    widget_key,
)


def _filename(*parts: str) -> str:
    return "__".join(slugify(part) for part in parts if part)


_list_layers_cached = st.cache_data(show_spinner=False)(list_available_layers)

# Keep compare-tab selection state separate so projection defaults do not
# overwrite cosine similarity defaults.
_LAST_COSINE_PERSONAS_KEY = "compare:last_personas:cosine"
_LAST_PROJECTION_PERSONAS_KEY = "compare:last_personas:projection"
_LAST_MASK_STRATEGY_KEY = "compare:last_mask_strategy"


def _select_artifact_personas(
    store: ActivationStore,
    variants: list[str],
    mask_strategy: MaskStrategy,
    *,
    widget_scope: str,
    remember_key: str,
    default_all: bool = False,
) -> tuple[list[str], dict[str, str]]:
    persona_options = list_available_personas(
        store.root_dir,
        store.model_name,
        variants,
        mask_strategy=mask_strategy,
    )
    persona_names = load_persona_names(
        store.root_dir,
        store.model_name,
        variants,
        persona_options,
        mask_strategy=mask_strategy,
    )
    if not persona_options:
        if len(variants) > 1:
            st.info(
                "No personas have saved activations for all selected variants. Run extraction for both variants first."
            )
        else:
            st.info("No personas found for this model yet. Run extraction first.")
        return [], persona_names

    last_personas: list[str] = st.session_state.get(remember_key, [])
    default_personas = [p for p in last_personas if p in persona_options]
    if not default_personas:
        default_personas = persona_options if default_all else persona_options[:1]

    persona_key = widget_key(
        "load",
        "personas",
        widget_scope,
        store.model_name,
        mask_strategy.value,
        *variants,
    )

    def _remember_personas() -> None:
        st.session_state[remember_key] = [
            persona_id
            for persona_id in st.session_state.get(persona_key, [])
            if persona_id in persona_options
        ]

    persona_ids = st.multiselect(
        "Personas",
        options=persona_options,
        default=default_personas,
        format_func=lambda persona_id: persona_display_label(
            persona_id, persona_names.get(persona_id)
        ),
        key=persona_key,
        on_change=_remember_personas,
    )
    return persona_ids, persona_names


def _render_save_buttons(
    figs: list[object],
    filenames: list[str],
    key_suffix: str,
) -> None:
    """Render Save HTML / Save PNG column buttons for one or more figures."""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save HTML", key=widget_key("load", "save_html", key_suffix)):
            paths = [save_plot_html(fig, fn) for fig, fn in zip(figs, filenames)]
            st.success(f"Saved {len(paths)} HTML file(s) to `artifacts/plots`.")
    with col2:
        if st.button("Save PNG", key=widget_key("load", "save_png", key_suffix)):
            try:
                paths = [save_plot_png(fig, fn) for fig, fn in zip(figs, filenames)]
                st.success(f"Saved {len(paths)} PNG file(s) to `artifacts/plots`.")
            except Exception as exc:
                st.error(f"Could not save PNG: {exc}")


def _render_mask_strategy_select(scope: str) -> MaskStrategy:
    last_strategy = st.session_state.get(
        _LAST_MASK_STRATEGY_KEY,
        MaskStrategy.ANSWER_MEAN.value,
    )
    strategies = list(MaskStrategy)
    default_index = next(
        (
            idx
            for idx, strategy in enumerate(strategies)
            if strategy.value == last_strategy
        ),
        0,
    )
    selected = st.selectbox(
        "Mask strategy",
        options=strategies,
        index=default_index,
        format_func=lambda strategy: strategy.value.replace("_", " ").title(),
        key=widget_key("load", "mask_strategy", scope),
        help="Which extracted activation artifact set to load.",
    )
    st.session_state[_LAST_MASK_STRATEGY_KEY] = selected.value
    return selected


def _render_cosine_similarity(
    store: ActivationStore,
    mask_strategy: MaskStrategy,
) -> None:
    if len(PERSONA_VARIANTS) < 2:
        st.info("Need at least two non-baseline variants for cosine comparison.")
        return

    col1, col2 = st.columns(2)
    with col1:
        variant_a = st.selectbox(
            "Variant A",
            options=PERSONA_VARIANTS,
            index=0,
            format_func=prompt_variant_label,
            key=widget_key("load", "variant_a"),
        )
    with col2:
        variant_b = st.selectbox(
            "Variant B",
            options=PERSONA_VARIANTS,
            index=min(1, len(PERSONA_VARIANTS) - 1),
            format_func=prompt_variant_label,
            key=widget_key("load", "variant_b"),
        )

    if variant_a == variant_b:
        st.warning("Choose two different variants to compare.")
        return

    persona_ids, _ = _select_artifact_personas(
        store,
        [variant_a, variant_b],
        mask_strategy,
        widget_scope="cosine",
        remember_key=_LAST_COSINE_PERSONAS_KEY,
    )
    if not persona_ids:
        return

    cosine_fig_key = widget_key(
        "load",
        "cosine_fig_state",
        store.model_name,
        mask_strategy.value,
        variant_a,
        variant_b,
    )
    filename = _filename(
        "compare",
        "cosine",
        store.model_name,
        mask_strategy.value,
        variant_a,
        variant_b,
    )
    pairs_filename = _filename(
        "compare",
        "cosine_pairs",
        store.model_name,
        mask_strategy.value,
        "_".join(PERSONA_VARIANTS),
    )

    if st.button("Compare vectors", type="primary"):
        traces, loaded_names, errors = load_mean_activations(
            store.root_dir,
            store.model_name,
            persona_ids,
            variant_a,
            variant_b,
            mask_strategy=mask_strategy,
        )

        if errors:
            for err in errors:
                st.error(f"Failed to load vectors: `{err}`")
        if not traces:
            st.error("No personas loaded successfully.")
            st.info(
                "Check that extraction has been run for both variants and selected personas."
            )
            st.session_state.pop(cosine_fig_key, None)
            return

        display_traces = [
            (
                persona_display_label(persona_id, loaded_names.get(persona_id)),
                short,
                long,
            )
            for persona_id, short, long in traces
        ]
        fig = plot_layer_similarity(
            display_traces,
            title=f"{prompt_variant_label(variant_a)} vs {prompt_variant_label(variant_b)}",
            show=False,
        )

        pair_traces = []
        pair_errors: list[str] = []
        for left, right in combinations(PERSONA_VARIANTS, 2):
            pair_data, _, pair_load_errors = load_mean_activations(
                store.root_dir,
                store.model_name,
                persona_ids,
                left,
                right,
                mask_strategy=mask_strategy,
            )
            pair_errors.extend(pair_load_errors)
            if not pair_data:
                continue
            pair_traces.append(
                (
                    f"{prompt_variant_label(left)} vs {prompt_variant_label(right)}",
                    torch.stack([short for _, short, _ in pair_data]).mean(dim=0),
                    torch.stack([long for _, _, long in pair_data]).mean(dim=0),
                )
            )

        if pair_errors:
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
        st.session_state[cosine_fig_key] = (fig, pair_fig, len(traces), len(pair_traces))

    if cosine_fig_key in st.session_state:
        fig, pair_fig, n_traces, n_pair_traces = st.session_state[cosine_fig_key]
        st.plotly_chart(fig, width="stretch")
        figs = [fig]
        filenames = [filename]
        if pair_fig is not None:
            st.subheader("Variant pairs")
            st.plotly_chart(pair_fig, width="stretch")
            figs.append(pair_fig)
            filenames.append(pairs_filename)
        _render_save_buttons(figs, filenames, "cosine")
        st.success(f"Loaded {n_traces} personas for cosine comparison.")
        if pair_fig is not None:
            st.caption(f"Generated {n_pair_traces} averaged variant-pair trace(s).")


def _select_single_variant_samples(
    store: ActivationStore,
    mask_strategy: MaskStrategy,
    scope: str,
) -> tuple[str, list[str], str, list[int]] | None:
    variant = st.selectbox(
        "Variant",
        options=PERSONA_VARIANTS,
        index=PERSONA_VARIANTS.index("biography")
        if "biography" in PERSONA_VARIANTS
        else 0,
        format_func=prompt_variant_label,
        key=widget_key("load", "variant", scope),
    )
    persona_ids, _ = _select_artifact_personas(
        store,
        [variant],
        mask_strategy,
        widget_scope=scope,
        remember_key=_LAST_PROJECTION_PERSONAS_KEY,
        default_all=True,
    )
    if not persona_ids:
        return None

    persona_key = "_".join(sorted(persona_ids))
    layer_options = _list_layers_cached(
        str(store.root_dir),
        store.model_name,
        [variant],
        persona_ids,
        mask_strategy=mask_strategy,
    )
    if not layer_options:
        st.info("No shared layers are available for the selected personas.")
        return None

    selected_layers = st.multiselect(
        "Layers",
        options=layer_options,
        default=layer_options,
        key=widget_key(
            "load",
            "layers",
            scope,
            store.model_name,
            mask_strategy.value,
            variant,
            persona_key,
        ),
    )
    if not selected_layers:
        st.info("Select at least one layer.")
        return None

    return variant, persona_ids, persona_key, selected_layers


def _baseline_available(
    store: ActivationStore,
    mask_strategy: MaskStrategy,
) -> bool:
    return BASELINE_PERSONA_ID in list_available_personas(
        store.root_dir,
        store.model_name,
        [BASELINE_PERSONA_ID],
        mask_strategy=mask_strategy,
        warn_missing=False,
    )


def _render_baseline_reference_toggle(
    store: ActivationStore,
    mask_strategy: MaskStrategy,
    scope: str,
) -> bool:
    available = _baseline_available(store, mask_strategy)
    return st.checkbox(
        "Include Assistant baseline reference",
        value=available,
        disabled=not available,
        key=widget_key("load", "include_baseline", scope, mask_strategy.value),
        help=(
            "Adds the single saved baseline artifact as one reference sample."
            if available
            else "Run Assistant baseline extraction first."
        ),
    )


def _render_similarity_matrix(
    store: ActivationStore,
    mask_strategy: MaskStrategy,
) -> None:
    selected = _select_single_variant_samples(
        store,
        mask_strategy,
        "similarity_matrix",
    )
    if selected is None:
        return
    variant, persona_ids, persona_key, selected_layers = selected
    include_baseline = _render_baseline_reference_toggle(
        store,
        mask_strategy,
        "similarity_matrix",
    )

    fig_key = widget_key(
        "load",
        "similarity_matrix_fig_state",
        store.model_name,
        mask_strategy.value,
        variant,
        "persona_mean",
        persona_key,
        BASELINE_PERSONA_ID if include_baseline else "no_baseline",
    )
    filename = _filename(
        "compare",
        "similarity_matrix",
        store.model_name,
        mask_strategy.value,
        variant,
        "persona_mean",
        persona_key,
        BASELINE_PERSONA_ID if include_baseline else "",
    )

    if st.button("Generate similarity matrix", type="primary"):
        try:
            samples = load_persona_mean_samples(
                store.root_dir,
                store.model_name,
                variant,
                mask_strategy=mask_strategy,
                persona_ids=persona_ids,
                include_baseline=include_baseline,
            )
            matrix_fig = build_layered_figure(
                samples,
                "similarity",
                layers=selected_layers,
                title=(
                    "Centered similarity - "
                    f"{prompt_variant_label(variant)} - personas averaged over questions"
                ),
            )
            trajectory_fig = build_pair_similarity_figure(
                samples,
                layers=selected_layers,
                title=(
                    "Pair similarity trajectories - "
                    f"{prompt_variant_label(variant)} - personas averaged over questions"
                ),
            )
            st.session_state[fig_key] = (
                matrix_fig,
                trajectory_fig,
                samples.vectors.shape[0],
            )
        except Exception as exc:
            st.error(f"Could not build similarity matrix: {exc}")
            st.session_state.pop(fig_key, None)

    if fig_key in st.session_state:
        matrix_fig, trajectory_fig, n_samples = st.session_state[fig_key]
        st.plotly_chart(matrix_fig, width="stretch")
        st.subheader("Pair trajectories")
        st.plotly_chart(trajectory_fig, width="stretch")
        _render_save_buttons(
            [matrix_fig, trajectory_fig],
            [filename, f"{filename}__pair_trajectories"],
            "similarity_matrix",
        )
        st.success(f"Loaded {n_samples} samples.")


def _render_embedding_analysis(
    store: ActivationStore,
    analysis_mode: str,
    mask_strategy: MaskStrategy,
) -> None:
    selected = _select_single_variant_samples(
        store,
        mask_strategy,
        analysis_mode.lower(),
    )
    if selected is None:
        return
    variant, persona_ids, persona_key, selected_layers = selected

    figure_kind = analysis_mode.lower()
    include_baseline = _render_baseline_reference_toggle(
        store,
        mask_strategy,
        analysis_mode.lower(),
    )

    fig_key = widget_key(
        "load",
        "embedding_fig_state",
        store.model_name,
        mask_strategy.value,
        figure_kind,
        variant,
        "persona_mean",
        persona_key,
        BASELINE_PERSONA_ID if include_baseline else "no_baseline",
    )
    filename = _filename(
        "compare",
        figure_kind,
        store.model_name,
        mask_strategy.value,
        variant,
        "persona_mean",
        persona_key,
        BASELINE_PERSONA_ID if include_baseline else "",
    )

    if st.button(f"Generate {analysis_mode} projection", type="primary"):
        try:
            samples = load_persona_mean_samples(
                store.root_dir,
                store.model_name,
                variant,
                mask_strategy=mask_strategy,
                persona_ids=persona_ids,
                include_baseline=include_baseline,
            )
            fig = build_layered_figure(
                samples,
                figure_kind,
                layers=selected_layers,
                title=(
                    f"{analysis_mode} - "
                    f"{prompt_variant_label(variant)} - Persona means"
                ),
            )
            st.session_state[fig_key] = (fig, samples.vectors.shape[0])
        except Exception as exc:
            st.error(f"Could not build {analysis_mode}: {exc}")
            st.session_state.pop(fig_key, None)

    if fig_key in st.session_state:
        fig, n_samples = st.session_state[fig_key]
        st.plotly_chart(fig, width="stretch")
        _render_save_buttons([fig], [filename], figure_kind)
        st.success(f"Loaded {n_samples} samples.")


def render_compare_tab(model_name: str) -> None:
    """Render the compare tab."""

    st.title("Compare")
    st.caption("Compare saved activations by cosine similarity, PCA, or UMAP.")

    st.subheader("Analysis")

    with st.expander("Advanced", expanded=False):
        artifacts_root = st.text_input(
            "Artifacts root",
            value=str(get_artifacts_dir() / "activations"),
        )

    store = ActivationStore(model_name, artifacts_root)

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
    mask_strategy = _render_mask_strategy_select(analysis_mode)

    if analysis_mode == "Cosine similarity":
        _render_cosine_similarity(store, mask_strategy)
        return
    if analysis_mode == "Similarity matrix":
        _render_similarity_matrix(store, mask_strategy)
        return

    _render_embedding_analysis(store, analysis_mode, mask_strategy)
