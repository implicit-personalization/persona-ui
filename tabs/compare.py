from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import streamlit as st
from persona_data.environment import get_artifacts_dir
from persona_data.synth_persona import BASELINE_PERSONA_ID
from persona_vectors.analysis import load_persona_vectors, load_variant_vectors
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import (
    build_layered_figure,
    build_pair_similarity_figure,
    plot_layer_similarity,
    plot_persona_dendrogram,
    save_plot_html,
)

from utils.compare_sources import (
    DEFAULT_COMPARE_MODEL,
    DEFAULT_HUB_REPO,
    SOURCE_HUB,
    SOURCE_LOCAL,
    SOURCES,
    Store,
    activation_store_cached,
    available_variants,
    hub_models_by_mask_strategy,
    local_model_matches,
    local_model_options_cached,
    persona_names_cached,
    personas_cached,
    store_cache_parts,
    store_id,
    store_layers_cached,
)
from utils.controls import render_mask_strategy_select
from utils.helpers import (
    ANALYSIS_HELP_TEXT,
    ANALYSIS_MODES,
    prompt_variant_label,
    slugify,
    widget_key,
)


def _filename(*parts: str) -> str:
    return "__".join(slugify(part) for part in parts if part)


# Keep compare-tab selection state separate so projection defaults do not
# overwrite cosine similarity defaults.
_LAST_COSINE_PERSONAS_KEY = "compare:last_personas:cosine"
_LAST_PROJECTION_PERSONAS_KEY = "compare:last_personas:projection"
_LAST_MASK_STRATEGY_KEY = "compare:last_mask_strategy"
_LAST_SOURCE_KEY = "compare:last_source"


def _is_assistant_persona(persona_id: str, persona_name: str | None = None) -> bool:
    persona_id_normalized = persona_id.strip().lower()
    persona_name_normalized = (persona_name or "").strip().lower()
    return (
        persona_id_normalized in {"assistant", BASELINE_PERSONA_ID.lower()}
        or persona_name_normalized == "assistant"
    )


@dataclass(frozen=True)
class CosineSelection:
    variants: list[str]
    variant_a: str
    variant_b: str
    persona_ids: list[str]
    persona_key: str


@dataclass(frozen=True)
class PersonaOptions:
    regular_ids: list[str]
    assistant_id: str | None


def _layers_for_variant(
    store: Store,
    variant: str,
    persona_ids: list[str],
    mask_strategy: MaskStrategy,
) -> list[int]:
    source, location, model_name = store_cache_parts(store)
    return store_layers_cached(
        source,
        location,
        model_name,
        mask_strategy.value,
        (variant,),
        tuple(persona_ids),
    )


def _load_persona_options(
    store: Store,
    variants: list[str],
    mask_strategy: MaskStrategy,
    *,
    empty_message: str,
) -> PersonaOptions | None:
    source, location, model_name = store_cache_parts(store)
    variant_key = tuple(variants)
    persona_ids = personas_cached(
        source,
        location,
        model_name,
        mask_strategy.value,
        variant_key,
    )
    if not persona_ids:
        st.info(empty_message)
        return None

    persona_names = persona_names_cached(
        source,
        location,
        model_name,
        mask_strategy.value,
        variant_key,
        tuple(persona_ids),
    )
    assistant_ids = [
        persona_id
        for persona_id in persona_ids
        if _is_assistant_persona(persona_id, persona_names.get(persona_id))
    ]
    assistant_id = next(
        (
            persona_id
            for persona_id in assistant_ids
            if persona_id == BASELINE_PERSONA_ID
        ),
        assistant_ids[0] if assistant_ids else None,
    )
    regular_ids = [
        persona_id for persona_id in persona_ids if persona_id not in assistant_ids
    ]
    if not regular_ids and assistant_id is None:
        st.info("No personas found for this model and variant.")
        return None
    return PersonaOptions(regular_ids=regular_ids, assistant_id=assistant_id)


def _seed_persona_memory(
    remember_key: str,
    options: PersonaOptions,
    *,
    default_all: bool,
) -> tuple[int, bool]:
    remembered_count_key = f"{remember_key}:count"
    remembered_assistant_key = f"{remember_key}:include_assistant"
    legacy_ids = st.session_state.get(remember_key, [])
    if isinstance(legacy_ids, list) and legacy_ids:
        st.session_state.setdefault(
            remembered_count_key,
            sum(persona_id in options.regular_ids for persona_id in legacy_ids),
        )
        st.session_state.setdefault(
            remembered_assistant_key,
            options.assistant_id in legacy_ids,
        )

    default_count = (
        len(options.regular_ids) if default_all else min(1, len(options.regular_ids))
    )
    remembered_count = int(st.session_state.get(remembered_count_key, default_count))
    persona_count = min(max(remembered_count, 0), len(options.regular_ids))
    include_assistant = bool(
        st.session_state.get(remembered_assistant_key, options.assistant_id is not None)
    )
    return persona_count, include_assistant


def _render_persona_count_controls(
    store: Store,
    variants: list[str],
    mask_strategy: MaskStrategy,
    widget_scope: str,
    options: PersonaOptions,
    *,
    default_count: int,
    include_assistant_default: bool,
) -> tuple[int, bool]:
    count_key = widget_key(
        "load",
        "persona_count",
        widget_scope,
        store.model_name,
        mask_strategy.value,
        *variants,
    )
    assistant_key = widget_key(
        "load",
        "include_assistant",
        widget_scope,
        store.model_name,
        mask_strategy.value,
        *variants,
    )

    if options.regular_ids:
        persona_count = st.slider(
            "Personas",
            min_value=0 if options.assistant_id is not None else 1,
            max_value=len(options.regular_ids),
            value=default_count,
            key=count_key,
            help="Use the first N available non-assistant personas.",
        )
    else:
        persona_count = 0
        st.caption("No non-assistant personas are available for this selection.")
    include_assistant = False
    if options.assistant_id is not None:
        include_assistant = st.checkbox(
            "Include Assistant persona",
            value=include_assistant_default,
            key=assistant_key,
        )
    return persona_count, include_assistant


def _select_artifact_personas(
    store: Store,
    variants: list[str],
    mask_strategy: MaskStrategy,
    *,
    widget_scope: str,
    remember_key: str,
    default_all: bool = False,
) -> list[str]:
    empty_message = (
        "No personas have vectors for all selected variants. "
        "Pick a single variant or change the source."
        if len(variants) > 1
        else "No personas found for this model and variant."
    )
    options = _load_persona_options(
        store,
        variants,
        mask_strategy,
        empty_message=empty_message,
    )
    if options is None:
        return []

    default_count, include_assistant_default = _seed_persona_memory(
        remember_key,
        options,
        default_all=default_all,
    )
    persona_count, include_assistant = _render_persona_count_controls(
        store,
        variants,
        mask_strategy,
        widget_scope,
        options,
        default_count=default_count,
        include_assistant_default=include_assistant_default,
    )

    persona_ids = options.regular_ids[:persona_count]
    if include_assistant and options.assistant_id is not None:
        persona_ids.append(options.assistant_id)

    remembered_count_key = f"{remember_key}:count"
    remembered_assistant_key = f"{remember_key}:include_assistant"
    st.session_state[remembered_count_key] = persona_count
    st.session_state[remembered_assistant_key] = include_assistant
    st.session_state[remember_key] = persona_ids

    if not persona_ids:
        st.info("Select at least one persona or include the Assistant persona.")
        return []

    regular_label = f"{persona_count} persona{'s' if persona_count != 1 else ''}"
    assistant_label = (
        " plus Assistant" if include_assistant and options.assistant_id else ""
    )
    st.caption(f"Using {regular_label}{assistant_label}.")
    return persona_ids


def _render_save_buttons(
    figs: list[object],
    filenames: list[str],
    key_suffix: str,
) -> None:
    """Render the Save HTML button for one or more figures."""
    if st.button("Save HTML", key=widget_key("load", "save_html", key_suffix)):
        try:
            paths = [save_plot_html(fig, fn) for fig, fn in zip(figs, filenames)]
            st.success(f"Saved {len(paths)} HTML file(s) to `artifacts/plots`.")
        except Exception as exc:
            st.error(f"Could not save HTML: {exc}")


def _render_mask_strategy_select(scope: str) -> MaskStrategy:
    return render_mask_strategy_select(
        key=widget_key("load", "mask_strategy", scope),
        last_key=_LAST_MASK_STRATEGY_KEY,
        help_text="Which extracted activation set to load.",
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
        persona_key="_".join(sorted(persona_ids)),
    )


def _build_cosine_figures(
    store: Store,
    selection: CosineSelection,
) -> tuple[object, object | None, int, int] | None:
    variant_sample_cache = {}

    def _load_pair(left: str, right: str):
        key = tuple(sorted((left, right)))
        if key not in variant_sample_cache:
            variant_sample_cache[key] = load_variant_vectors(
                store,
                [left, right],
                persona_ids=selection.persona_ids,
            )
        return variant_sample_cache[key]

    try:
        variant_samples = _load_pair(selection.variant_a, selection.variant_b)
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
            pair_samples = _load_pair(left, right)
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
            store_id(store),
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
    store: Store,
    mask_strategy: MaskStrategy,
    scope: str,
) -> tuple[str, list[str], str, list[int]] | None:
    variants = available_variants(store, mask_strategy)
    if not variants:
        st.info("No variants with saved vectors for this model.")
        return None
    variant = st.selectbox(
        "Variant",
        options=variants,
        index=variants.index("biography") if "biography" in variants else 0,
        format_func=prompt_variant_label,
        key=widget_key("load", "variant", scope, store_id(store)),
    )
    persona_ids = _select_artifact_personas(
        store,
        [variant],
        mask_strategy,
        widget_scope=f"{scope}:{store_id(store)}",
        remember_key=_LAST_PROJECTION_PERSONAS_KEY,
        default_all=True,
    )
    if not persona_ids:
        return None

    persona_key = "_".join(sorted(persona_ids))
    layer_options = _layers_for_variant(store, variant, persona_ids, mask_strategy)
    if not layer_options:
        st.info("No shared layers are available for the selected personas.")
        return None

    st.caption(f"Using all {len(layer_options)} available layer(s).")
    return variant, persona_ids, persona_key, layer_options


def _render_layered_figure_analysis(
    store: Store,
    mask_strategy: MaskStrategy,
    *,
    scope: str,
    figure_kind: str,
    button_label: str,
    title_fn: Callable[[str], str],
    include_pair_trajectories: bool = False,
    n_components: int = 2,
) -> None:
    """Render a single-variant layered analysis: select → button → figure(s).

    Used for similarity matrix, PCA, and UMAP. Set ``include_pair_trajectories``
    to add the pair-similarity-trajectory figure (similarity matrix only).
    """
    selected = _select_single_variant_samples(store, mask_strategy, scope)
    if selected is None:
        return
    variant, persona_ids, persona_key, selected_layers = selected

    n_clusters = None
    if figure_kind in {"pca", "umap"}:
        use_kmeans = st.toggle(
            "Color by K-means clusters",
            value=False,
            key=widget_key("load", "kmeans_enabled", scope, store_id(store)),
            help="Run K-means on persona vectors and color each persona by cluster.",
        )
        if use_kmeans:
            n_clusters = st.slider(
                "K (clusters)",
                min_value=2,
                max_value=min(10, len(persona_ids)),
                value=min(3, len(persona_ids)),
                key=widget_key("load", "kmeans_k", scope, store_id(store)),
            )

    fig_key = widget_key(
        "load",
        f"{scope}_fig_state",
        store_id(store),
        store.model_name,
        mask_strategy.value,
        figure_kind,
        str(n_components),
        str(n_clusters),
        variant,
        "persona_vector",
        persona_key,
    )
    filename = scope

    if st.button(button_label, type="primary"):
        try:
            samples = load_persona_vectors(
                store,
                variant,
                mask_strategy=mask_strategy,
                persona_ids=persona_ids,
            )
            build_kwargs = {}
            if figure_kind in {"umap", "pca"}:
                build_kwargs["n_components"] = n_components
                if n_clusters is not None:
                    build_kwargs["n_clusters"] = n_clusters
            main_fig = build_layered_figure(
                samples,
                figure_kind,
                layers=selected_layers,
                title=title_fn(variant),
                **build_kwargs,
            )
            if figure_kind in {"umap", "pca"}:
                main_fig.update_layout(height=700)
            extra_fig = (
                build_pair_similarity_figure(
                    samples,
                    layers=selected_layers,
                    title=(
                        "Pair similarity trajectories - "
                        f"{prompt_variant_label(variant)} - persona vectors"
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


_LAST_DENDRO_PERSONAS_KEY = "compare:last_personas:dendro"
_DENDRO_LINKAGE_OPTIONS = ["ward", "complete", "average", "single"]


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
        default_b_idx = variants.index("templated") if "templated" in variants else min(1, len(variants) - 1)
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
    persona_ids = _select_artifact_personas(
        store,
        shared_variants,
        mask_strategy,
        widget_scope=f"dendro:{store_id(store)}",
        remember_key=_LAST_DENDRO_PERSONAS_KEY,
        default_all=True,
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

    persona_key = "_".join(sorted(persona_ids))
    fig_key = widget_key(
        "load", "dendro_fig_state",
        store_id(store),
        store.model_name,
        mask_strategy.value,
        variant_a, variant_b,
        persona_key,
        str(layered_mode), linkage,
    )

    if st.button(
        "Generate dendrograms",
        type="primary",
        key=widget_key("load", "dendro_btn", store_id(store), variant_a, variant_b, persona_key),
    ):
        try:
            samples_a = load_persona_vectors(
                store, variant_a, mask_strategy=mask_strategy, persona_ids=persona_ids,
            )
            fig_a = plot_persona_dendrogram(
                samples_a,
                layered=layered_mode,
                linkage=linkage,
                title=f"Dendrogram — {prompt_variant_label(variant_a)}",
            )
            fig_a.update_layout(height=750)
            fig_b = None
            if variant_a != variant_b:
                samples_b = load_persona_vectors(
                    store, variant_b, mask_strategy=mask_strategy, persona_ids=persona_ids,
                )
                fig_b = plot_persona_dendrogram(
                    samples_b,
                    layered=layered_mode,
                    linkage=linkage,
                    title=f"Dendrogram — {prompt_variant_label(variant_b)}",
                )
                fig_b.update_layout(height=750)
            st.session_state[fig_key] = (fig_a, fig_b, len(persona_ids), variant_a, variant_b)
        except Exception as exc:
            st.error(f"Could not build dendrogram: {exc}")
            st.session_state.pop(fig_key, None)

    if fig_key in st.session_state:
        fig_a, fig_b, n_personas, va, vb = st.session_state[fig_key]
        if fig_b is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(prompt_variant_label(va))
                st.plotly_chart(fig_a, width="stretch")
            with col_b:
                st.subheader(prompt_variant_label(vb))
                st.plotly_chart(fig_b, width="stretch")
        else:
            st.plotly_chart(fig_a, width="stretch")

        figs = [fig_a] + ([fig_b] if fig_b else [])
        filenames = [
            _filename("dendro", store.model_name, mask_strategy.value, va),
            *([_filename("dendro", store.model_name, mask_strategy.value, vb)] if fig_b else []),
        ]
        _render_save_buttons(figs, filenames, "dendro")
        st.success(f"Generated dendrogram(s) for {n_personas} persona(s).")


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
        "compare:hub_model_fallback",
        DEFAULT_COMPARE_MODEL,
    )
    try:
        models_by_strategy = hub_models_by_mask_strategy(repo_id)
    except Exception as exc:
        st.warning(f"Could not load Hub configs for `{repo_id}`: {exc}")
        return st.text_input(
            "Hub model",
            value=fallback_model,
            key="compare:hub_model_fallback",
            help="Compare-only model id to use if Hub config discovery is unavailable.",
        )

    model_options = models_by_strategy.get(mask_strategy, [])
    if not model_options:
        st.warning(
            f"No Hub vector configs found for `{mask_strategy.value}` in `{repo_id}`."
        )
        return st.text_input(
            "Hub model",
            value=fallback_model,
            key="compare:hub_model_fallback",
            help="Compare-only model id to use for this Hub repo.",
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
    fallback_model = st.session_state.get("compare:local_model", DEFAULT_COMPARE_MODEL)
    model_options = local_model_options_cached(artifacts_root, mask_strategy.value)
    if not model_options:
        return st.text_input(
            "Local model",
            value=fallback_model,
            key="compare:local_model",
            help="Compare-only local model id or path.",
        )

    custom = st.toggle(
        "Custom local model",
        value=False,
        key="compare:local_model_custom_enabled",
        help="Enter a model id/path manually instead of choosing from activation directories.",
    )
    if custom:
        return st.text_input(
            "Local model",
            value=fallback_model,
            key="compare:local_model",
            help="Compare-only local model id or path.",
        )

    previous_model = st.session_state.get("compare:local_model_select", fallback_model)
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
        key="compare:local_model_select",
        help="Models discovered under the selected artifacts root.",
    )
    st.session_state["compare:local_model"] = selected
    return selected


def _build_store(source: str, mask_strategy: MaskStrategy) -> Store:
    if source == SOURCE_HUB:
        repo = st.text_input(
            "Hub repo",
            value=st.session_state.get("compare:hub_repo", DEFAULT_HUB_REPO),
            key="compare:hub_repo",
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
        key="compare:artifacts_root",
    )
    artifacts_root = str(Path(artifacts_root).expanduser())
    local_model_name = _render_local_model_select(artifacts_root, mask_strategy)
    return activation_store_cached(
        SOURCE_LOCAL,
        artifacts_root,
        local_model_name,
        mask_strategy.value,
    )


def render_compare_tab() -> None:
    """Render the analysis tab."""

    st.title("Analysis")
    st.caption("Analyse persona vectors by cosine similarity, PCA, UMAP, or hierarchical clustering.")

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
        )
        return

    if analysis_mode == "Dendrogram":
        _render_dendrogram_analysis(store, mask_strategy)
        return

    dimension_choice = st.segmented_control(
        "Projection dimensions",
        options=["2D", "3D"],
        default="2D",
        key=widget_key("load", "projection_dims", analysis_mode),
        label_visibility="collapsed",
    )
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
    )
