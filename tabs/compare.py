from collections.abc import Callable
from itertools import combinations
from dataclasses import dataclass

import streamlit as st
from persona_data.environment import get_artifacts_dir
from persona_vectors.analysis import (
    load_persona_mean_samples,
    load_variant_mean_samples,
)
from persona_vectors.artifacts import ActivationStore
from persona_vectors.artifacts import list_layers as list_available_layers
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


@dataclass(frozen=True)
class CosineSelection:
    variants: list[str]
    variant_a: str
    variant_b: str
    persona_ids: list[str]
    persona_key: str


def _select_artifact_personas(
    store: ActivationStore,
    variants: list[str],
    mask_strategy: MaskStrategy,
    *,
    widget_scope: str,
    remember_key: str,
    default_all: bool = False,
) -> tuple[list[str], dict[str, str]]:
    persona_options = store.list_personas(variants)
    persona_names = store.persona_names(
        persona_options,
        variants=variants,
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
    selected = st.selectbox(
        "Mask strategy",
        options=strategies,
        index=next(
            (
                idx
                for idx, strategy in enumerate(strategies)
                if strategy.value == last_strategy
            ),
            0,
        ),
        format_func=lambda strategy: strategy.value.replace("_", " ").title(),
        key=widget_key("load", "mask_strategy", scope),
        help="Which extracted activation artifact set to load.",
    )
    st.session_state[_LAST_MASK_STRATEGY_KEY] = selected.value
    return selected


def _render_cosine_selection(
    store: ActivationStore,
    mask_strategy: MaskStrategy,
) -> CosineSelection | None:
    variants = list(store.variants)
    if len(variants) < 2:
        st.info("Need at least two non-baseline variants for cosine comparison.")
        return None

    with st.expander("Vector selection", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            variant_a = st.selectbox(
                "Variant A",
                options=variants,
                index=0,
                format_func=prompt_variant_label,
                key=widget_key("load", "variant_a"),
            )
        with col2:
            variant_b = st.selectbox(
                "Variant B",
                options=variants,
                index=min(1, len(variants) - 1),
                format_func=prompt_variant_label,
                key=widget_key("load", "variant_b"),
            )

        if variant_a == variant_b:
            st.warning("Choose two different variants to compare.")
            return None

        persona_ids, _ = _select_artifact_personas(
            store,
            [variant_a, variant_b],
            mask_strategy,
            widget_scope="cosine",
            remember_key=_LAST_COSINE_PERSONAS_KEY,
        )
    if not persona_ids:
        return None
    return CosineSelection(
        variants=variants,
        variant_a=variant_a,
        variant_b=variant_b,
        persona_ids=persona_ids,
        persona_key="_".join(sorted(persona_ids)),
    )


def _build_cosine_figures(
    store: ActivationStore,
    selection: CosineSelection,
) -> tuple[object, object | None, int, int] | None:
    try:
        variant_samples = load_variant_mean_samples(
            store,
            [selection.variant_a, selection.variant_b],
            persona_ids=selection.persona_ids,
        )
    except Exception as exc:
        st.error(f"Could not load vectors: {exc}")
        return None

    labels = variant_samples[selection.variant_a].labels
    display_traces = [
        (
            label,
            variant_samples[selection.variant_a].vectors[index],
            variant_samples[selection.variant_b].vectors[index],
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
            pair_samples = (
                variant_samples
                if {left, right} == {selection.variant_a, selection.variant_b}
                else load_variant_mean_samples(
                    store,
                    [left, right],
                    persona_ids=selection.persona_ids,
                )
            )
            pair_traces.append(
                (
                    f"{prompt_variant_label(left)} vs {prompt_variant_label(right)}",
                    pair_samples[left].vectors.mean(dim=0),
                    pair_samples[right].vectors.mean(dim=0),
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
    store: ActivationStore,
    mask_strategy: MaskStrategy,
) -> None:
    selection = _render_cosine_selection(store, mask_strategy)
    if selection is None:
        return

    cosine_fig_key = widget_key(
        "load",
        "cosine_fig_state",
        store.model_name,
        mask_strategy.value,
        selection.variant_a,
        selection.variant_b,
        selection.persona_key,
    )
    filename = _filename(
        "compare",
        "cosine",
        store.model_name,
        mask_strategy.value,
        selection.variant_a,
        selection.variant_b,
    )
    pairs_filename = _filename(
        "compare",
        "cosine_pairs",
        store.model_name,
        mask_strategy.value,
        "_".join(selection.variants),
    )

    if st.button(
        "Compare vectors",
        type="primary",
        key=widget_key(
            "load",
            "compare_vectors",
            store.model_name,
            mask_strategy.value,
            selection.variant_a,
            selection.variant_b,
            selection.persona_key,
        ),
    ):
        figures = _build_cosine_figures(store, selection)
        if figures is None:
            st.session_state.pop(cosine_fig_key, None)
            return
        st.session_state[cosine_fig_key] = figures

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
    variants = list(store.variants)
    variant = st.selectbox(
        "Variant",
        options=variants,
        index=(
            variants.index("biography")
            if "biography" in variants
            else 0
        ),
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


def _render_layered_figure_analysis(
    store: ActivationStore,
    mask_strategy: MaskStrategy,
    *,
    scope: str,
    figure_kind: str,
    button_label: str,
    title_fn: Callable[[str], str],
    include_pair_trajectories: bool = False,
) -> None:
    """Render a single-variant layered analysis: select → button → figure(s).

    Used for similarity matrix, PCA, and UMAP. Set ``include_pair_trajectories``
    to add the pair-similarity-trajectory figure (similarity matrix only).
    """
    selected = _select_single_variant_samples(store, mask_strategy, scope)
    if selected is None:
        return
    variant, persona_ids, persona_key, selected_layers = selected

    fig_key = widget_key(
        "load",
        f"{scope}_fig_state",
        store.model_name,
        mask_strategy.value,
        figure_kind,
        variant,
        "persona_mean",
        persona_key,
    )
    filename = _filename(
        "compare",
        scope,
        store.model_name,
        mask_strategy.value,
        variant,
        "persona_mean",
        persona_key,
    )

    if st.button(button_label, type="primary"):
        try:
            samples = load_persona_mean_samples(
                store,
                variant,
                mask_strategy=mask_strategy,
                persona_ids=persona_ids,
            )
            main_fig = build_layered_figure(
                samples,
                figure_kind,
                layers=selected_layers,
                title=title_fn(variant),
            )
            extra_fig = (
                build_pair_similarity_figure(
                    samples,
                    layers=selected_layers,
                    title=(
                        "Pair similarity trajectories - "
                        f"{prompt_variant_label(variant)} - "
                        "persona mean activations"
                    ),
                )
                if include_pair_trajectories
                else None
            )
            st.session_state[fig_key] = (main_fig, extra_fig, samples.vectors.shape[0])
        except Exception as exc:
            st.error(f"Could not build figure: {exc}")
            st.session_state.pop(fig_key, None)

    if fig_key in st.session_state:
        main_fig, extra_fig, n_samples = st.session_state[fig_key]
        st.plotly_chart(main_fig, width="stretch")
        figs = [main_fig]
        filenames = [filename]
        if extra_fig is not None:
            st.subheader("Pair trajectories")
            st.plotly_chart(extra_fig, width="stretch")
            figs.append(extra_fig)
            filenames.append(f"{filename}__pair_trajectories")
        _render_save_buttons(figs, filenames, scope)
        st.success(f"Loaded {n_samples} samples.")


def render_compare_tab(model_name: str) -> None:
    """Render the compare tab."""

    st.title("Compare")
    st.caption("Compare saved activations by cosine similarity, PCA, or UMAP.")

    with st.expander("Artifact settings", expanded=False):
        artifacts_root = st.text_input(
            "Artifacts root",
            value=str(get_artifacts_dir() / "activations"),
        )

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
    with st.expander("Activation settings", expanded=False):
        mask_strategy = _render_mask_strategy_select(analysis_mode)
    store = ActivationStore(model_name, artifacts_root, mask_strategy=mask_strategy)

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
                "Centered similarity - "
                f"{prompt_variant_label(v)} - persona mean activations"
            ),
            include_pair_trajectories=True,
        )
        return

    _render_layered_figure_analysis(
        store,
        mask_strategy,
        scope=analysis_mode.lower(),
        figure_kind=analysis_mode.lower(),
        button_label=f"Generate {analysis_mode} projection",
        title_fn=lambda v: (
            f"{analysis_mode} - {prompt_variant_label(v)} - Persona means"
        ),
    )
