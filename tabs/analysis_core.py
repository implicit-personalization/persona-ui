import gc
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from persona_data.environment import get_artifacts_dir
from persona_data.synth_persona import BASELINE_PERSONA_ID
from persona_vectors.attributes import (
    DEFAULT_MAX_ATTRIBUTE_CATEGORIES,
    attribute_color_kwargs,
    attribute_display_label,
)
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import (
    build_layered_figure,
    build_pair_similarity_figure,
    build_similarity_figures,
    plot_layer_similarity,
    plot_persona_dendrogram,
    prepare_layered_projection_data,
    save_plot_html,
)

from utils.analysis_sources import (
    DEFAULT_COMPARE_MODEL,
    DEFAULT_HUB_REPO,
    SOURCE_HUB,
    SOURCE_LOCAL,
    SOURCES,
    Store,
    activation_store_cached,
    available_variants,
    hub_models_by_mask_strategy,
    load_persona_vectors_cached,
    load_variant_vectors_cached,
    local_model_matches,
    local_model_options_cached,
    persona_names_cached,
    personas_cached,
    release_hf_store_cache,
    store_cache_parts,
    store_id,
    store_layers_cached,
)
from utils.analysis_metadata import (
    synth_persona_attribute_names,
    synth_persona_dataset_cached,
)
from utils.controls import render_mask_strategy_select
from utils.helpers import (
    ANALYSIS_HELP_TEXT,
    ANALYSIS_MODES,
    personas_fingerprint,
    prompt_variant_label,
    slugify,
    widget_key,
)
from utils.theme import style_plotly_layer_controls


def _filename(*parts: str) -> str:
    return "__".join(slugify(part) for part in parts if part)


# Keep compare-tab selection state separate so projection defaults do not
# overwrite cosine similarity defaults.
_LAST_COSINE_PERSONAS_KEY = "analysis:last_personas:cosine"
_LAST_PROJECTION_PERSONAS_KEY = "analysis:last_personas:projection"
_LAST_SIMILARITY_PERSONAS_KEY = "analysis:last_personas:similarity"
_LAST_MASK_STRATEGY_KEY = "analysis:last_mask_strategy"
_LAST_SOURCE_KEY = "analysis:last_source"
_LAST_PROJECTION_VARIANT_KEY = "analysis:last_projection_variant"
_LAST_SIMILARITY_VARIANT_KEY = "analysis:last_similarity_variant"
_LAST_PROJECTION_COLOR_MODE_KEY = "analysis:last_projection_color_mode"
_LAST_PROJECTION_ATTRIBUTE_KEY = "analysis:last_projection_attribute"
_LAST_PROJECTION_CLUSTER_K_KEY = "analysis:last_projection_cluster_k"
_LAST_PROJECTION_CLUSTER_MODE_KEY = "analysis:last_projection_cluster_mode"
_LAST_PROJECTION_HIGHLIGHTS_KEY = "analysis:last_projection_highlights"
_LAST_PROJECTION_DIMS_KEY = "analysis:last_projection_dims"
_LAST_LAYER_FRAMES_KEY = "analysis:last_layer_frames"

_DEFAULT_LAYER_FRAMES = 16
_DEFAULT_PERSONA_LIMITS = {
    "similarity": 120,
    "pca": 500,
    "umap": 500,
    "isomap": 500,
    "dendro": 160,
}
_MAX_SIMILARITY_CELLS = 4_000_000
_MAX_PAIR_TRAJECTORY_TRACES = 500
_DEFAULT_GRAPH_NEIGHBORS = 5
_PROJECTION_KINDS = {"pca", "umap", "isomap"}
_CLUSTER_MODES = {
    "Mean across layers": "mean_across_layers",
    "First selected layer": "first_layer",
    "Per layer": "per_layer",
}
_PROJECTION_COLOR_MODES = ["Persona", "K-means clusters", "Persona attribute"]
_MAX_ATTRIBUTE_CATEGORIES = DEFAULT_MAX_ATTRIBUTE_CATEGORIES


def _synth_persona_dataset():
    return synth_persona_dataset_cached()


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
    persona_names: dict[str, str]


@dataclass(frozen=True)
class ProjectionColorConfig:
    color_mode: str = "Persona"
    n_clusters: int | None = None
    cluster_mode: str | None = None
    attribute_name: str | None = None
    highlight_persona_ids: tuple[str, ...] = ()
    highlight_persona_key: str = ""


@dataclass(frozen=True)
class LayeredFigureStateKeys:
    figure: str
    projection: str | None = None


_HIGHLIGHT_OTHER_LABEL = "Other"
_HIGHLIGHT_OTHER_COLOR = "rgba(148, 163, 184, 0.35)"


def _persona_names_state_key(widget_scope: str) -> str:
    return widget_key("load", "persona_names", widget_scope)


def _persona_display_label(persona_names: dict[str, str], persona_id: str) -> str:
    name = persona_names.get(persona_id, persona_id)
    return f"{name} ({persona_id})" if name != persona_id else persona_id


def _highlight_persona_groups(
    persona_ids: list[str],
    persona_names: dict[str, str],
    highlight_persona_ids: tuple[str, ...],
) -> list[str] | None:
    if not highlight_persona_ids:
        return None

    highlighted = set(highlight_persona_ids)
    return [
        (
            _persona_display_label(persona_names, persona_id)
            if persona_id in highlighted
            else _HIGHLIGHT_OTHER_LABEL
        )
        for persona_id in persona_ids
    ]


def _sequence_to_list(value: object) -> list[object] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return None


def _gray_out_unselected_personas(fig: go.Figure) -> None:
    def _gray_trace(trace: object) -> None:
        marker = getattr(trace, "marker", None)
        if marker is None:
            return

        colors = _sequence_to_list(getattr(marker, "color", None))
        labels = _sequence_to_list(getattr(trace, "customdata", None))
        if colors is not None and labels is not None and len(colors) == len(labels):
            trace.marker.color = [
                (
                    _HIGHLIGHT_OTHER_COLOR
                    if str(label) == _HIGHLIGHT_OTHER_LABEL
                    else color
                )
                for label, color in zip(labels, colors, strict=True)
            ]
            return

        if getattr(trace, "name", None) == _HIGHLIGHT_OTHER_LABEL:
            trace.marker.color = _HIGHLIGHT_OTHER_COLOR
            trace.opacity = 0.28

    for trace in fig.data:
        _gray_trace(trace)
    for frame in fig.frames:
        for trace in frame.data:
            _gray_trace(trace)


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


def _load_persona_vectors(
    store: Store,
    variant: str,
    mask_strategy: MaskStrategy,
    persona_ids: list[str],
):
    source, location, model_name = store_cache_parts(store)
    return load_persona_vectors_cached(
        source,
        location,
        model_name,
        mask_strategy.value,
        variant,
        tuple(persona_ids),
    )


def _load_variant_vectors(
    store: Store,
    variants: list[str] | tuple[str, ...],
    mask_strategy: MaskStrategy,
    persona_ids: list[str],
):
    source, location, model_name = store_cache_parts(store)
    return load_variant_vectors_cached(
        source,
        location,
        model_name,
        mask_strategy.value,
        tuple(variants),
        tuple(persona_ids),
    )


def _clear_old_load_states(current_key: str, suffix: str) -> None:
    for key in list(st.session_state):
        if key == current_key or not isinstance(key, str):
            continue
        parts = key.split("::", 2)
        if len(parts) >= 2 and parts[0] == "load" and parts[1].endswith(suffix):
            st.session_state.pop(key, None)


def _clear_old_figure_states(current_key: str) -> None:
    _clear_old_load_states(current_key, "_fig_state")


def _clear_old_projection_states(current_key: str) -> None:
    _clear_old_load_states(current_key, "_projection_state")


def _store_figure_state(key: str, value: object) -> None:
    _clear_old_figure_states(key)
    st.session_state[key] = value


def _seed_selectbox_key(
    *,
    key: str,
    remember_key: str,
    options: list[str],
    default: str,
) -> str:
    value = st.session_state.get(key, st.session_state.get(remember_key, default))
    if value not in options:
        value = default
    return value


def _remember_multiselect(
    *,
    key: str,
    remember_key: str,
    options: list[str],
) -> list[str]:
    remembered = st.session_state.get(key, st.session_state.get(remember_key, []))
    if not isinstance(remembered, list):
        remembered = []
    return [value for value in remembered if value in options]


def _release_vector_memory(store: Store, variants: list[str] | tuple[str, ...]) -> None:
    release_hf_store_cache(store, variants)
    gc.collect()


def _evenly_spaced_layers(layers: list[int], max_count: int) -> list[int]:
    if max_count >= len(layers):
        return layers
    if max_count <= 1:
        return [layers[0]]

    last = len(layers) - 1
    indices = [round(i * last / (max_count - 1)) for i in range(max_count)]
    return [layers[index] for index in dict.fromkeys(indices)]


def _render_layer_frame_controls(
    store: Store,
    scope: str,
    layers: list[int],
) -> list[int]:
    if len(layers) <= _DEFAULT_LAYER_FRAMES:
        st.caption(f"Using all {len(layers)} available layer(s).")
        return layers

    frame_count = st.slider(
        "Layer frames",
        min_value=2,
        max_value=len(layers),
        value=min(
            max(
                int(
                    st.session_state.get(
                        _LAST_LAYER_FRAMES_KEY,
                        _DEFAULT_LAYER_FRAMES,
                    )
                ),
                2,
            ),
            len(layers),
        ),
        key=widget_key("load", "layer_frames", scope, store_id(store)),
        help="Limit animated Plotly frames to keep browser and RAM usage bounded.",
    )
    st.session_state[_LAST_LAYER_FRAMES_KEY] = frame_count
    selected = _evenly_spaced_layers(layers, frame_count)
    st.caption(f"Using {len(selected)} of {len(layers)} layers.")
    return selected


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
    return PersonaOptions(
        regular_ids=regular_ids,
        assistant_id=assistant_id,
        persona_names=persona_names,
    )


def _seed_persona_memory(
    remember_key: str,
    options: PersonaOptions,
    *,
    default_all: bool,
    default_count_limit: int | None = None,
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

    if default_count_limit is not None:
        default_count = min(default_count_limit, len(options.regular_ids))
    elif default_all:
        default_count = len(options.regular_ids)
    else:
        default_count = min(1, len(options.regular_ids))
    remembered_count = int(st.session_state.get(remembered_count_key, default_count))
    persona_count = min(max(remembered_count, 0), len(options.regular_ids))
    include_assistant = bool(st.session_state.get(remembered_assistant_key, False))
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
    default_count_limit: int | None = None,
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
        st.session_state.pop(_persona_names_state_key(widget_scope), None)
        return []

    default_count, include_assistant_default = _seed_persona_memory(
        remember_key,
        options,
        default_all=default_all,
        default_count_limit=default_count_limit,
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
    st.session_state[_persona_names_state_key(widget_scope)] = options.persona_names

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
            _style_plotly_figures(figs)
            paths = [
                save_plot_html(fig, fn) for fig, fn in zip(figs, filenames, strict=True)
            ]
            st.success(f"Saved {len(paths)} HTML file(s) to `artifacts/plots`.")
        except Exception as exc:
            st.error(f"Could not save HTML: {exc}")


def _style_plotly_figures(figs: list[object]) -> None:
    base = st.get_option("theme.base")
    for fig in figs:
        if isinstance(fig, go.Figure):
            style_plotly_layer_controls(fig, base)


def _plotly_chart(fig: object) -> None:
    _style_plotly_figures([fig])
    st.plotly_chart(
        fig,
        width="stretch",
        config={"responsive": True, "displaylogo": False},
    )


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
        persona_key=personas_fingerprint(persona_ids),
    )


def _build_cosine_figures(
    store: Store,
    mask_strategy: MaskStrategy,
    selection: CosineSelection,
) -> tuple[object, object | None, int, int] | None:
    variant_sample_cache: dict[str, object] = {}

    def _load_variant(variant: str):
        if variant not in variant_sample_cache:
            samples = _load_variant_vectors(
                store,
                [variant],
                mask_strategy,
                persona_ids=selection.persona_ids,
            )
            variant_sample_cache[variant] = samples[variant]
        return variant_sample_cache[variant]

    try:
        samples_a = _load_variant(selection.variant_a)
        samples_b = _load_variant(selection.variant_b)
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
            left_samples = _load_variant(left)
            right_samples = _load_variant(right)
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
            _release_vector_memory(store, selection.variants)
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


def _select_single_variant_samples(
    store: Store,
    mask_strategy: MaskStrategy,
    scope: str,
    *,
    remember_key: str,
    variant_remember_key: str,
    default_count_limit: int,
) -> tuple[str, list[str], str, list[int]] | None:
    variants = available_variants(store, mask_strategy)
    if not variants:
        st.info("No variants with saved vectors for this model.")
        return None
    variant_key = widget_key("load", "variant", scope, store_id(store))
    default_variant = "biography" if "biography" in variants else variants[0]
    selected_variant = _seed_selectbox_key(
        key=variant_key,
        remember_key=variant_remember_key,
        options=variants,
        default=default_variant,
    )
    variant = st.selectbox(
        "Variant",
        options=variants,
        index=variants.index(selected_variant),
        format_func=prompt_variant_label,
        key=variant_key,
    )
    st.session_state[variant_remember_key] = variant
    persona_ids = _select_artifact_personas(
        store,
        [variant],
        mask_strategy,
        widget_scope=f"{scope}:{store_id(store)}",
        remember_key=remember_key,
        default_count_limit=default_count_limit,
    )
    if not persona_ids:
        return None

    persona_key = personas_fingerprint(persona_ids)
    layer_options = _layers_for_variant(store, variant, persona_ids, mask_strategy)
    if not layer_options:
        st.info("No shared layers are available for the selected personas.")
        return None

    selected_layers = _render_layer_frame_controls(store, scope, layer_options)
    return variant, persona_ids, persona_key, selected_layers


def _render_pair_trajectory_control(
    *,
    enabled: bool,
    persona_count: int,
    scope: str,
    store: Store,
) -> bool:
    if not enabled:
        return False
    pair_count = persona_count * (persona_count - 1) // 2
    if pair_count > _MAX_PAIR_TRAJECTORY_TRACES:
        st.caption(
            "Pair trajectories hidden because this selection would create "
            f"{pair_count:,} Plotly traces."
        )
        return False
    return st.checkbox(
        "Pair trajectories",
        value=False,
        key=widget_key("load", "pair_trajectories", scope, store_id(store)),
        help="Adds one line per persona pair. Keep this off for larger selections.",
    )


def _validate_layered_figure_size(
    figure_kind: str,
    persona_count: int,
    selected_layers: list[int],
) -> bool:
    if figure_kind != "similarity":
        return True
    similarity_cells = persona_count * persona_count * len(selected_layers)
    if similarity_cells <= _MAX_SIMILARITY_CELLS:
        return True
    st.error(
        "Reduce personas or layer frames before generating the similarity "
        f"matrix ({similarity_cells:,} cells selected)."
    )
    return False


def _render_projection_color_config(
    store: Store,
    scope: str,
    persona_ids: list[str],
) -> ProjectionColorConfig | None:
    widget_scope = f"{scope}:{store_id(store)}"
    persona_key = personas_fingerprint(persona_ids)
    persona_names = st.session_state.get(
        _persona_names_state_key(widget_scope),
        {},
    )
    color_mode_key = widget_key("load", "color_mode", scope, store_id(store))
    selected_color_mode = _seed_selectbox_key(
        key=color_mode_key,
        remember_key=_LAST_PROJECTION_COLOR_MODE_KEY,
        options=_PROJECTION_COLOR_MODES,
        default="Persona",
    )
    color_mode = st.selectbox(
        "Color by",
        options=_PROJECTION_COLOR_MODES,
        index=_PROJECTION_COLOR_MODES.index(selected_color_mode),
        key=color_mode_key,
    )
    st.session_state[_LAST_PROJECTION_COLOR_MODE_KEY] = color_mode
    if color_mode == "K-means clusters":
        max_clusters = min(10, len(persona_ids))
        if max_clusters < 2:
            st.info("Select at least two personas to use K-means coloring.")
            return None
        cluster_key = widget_key("load", "cluster_k", scope, store_id(store))
        default_clusters = min(3, len(persona_ids))
        if cluster_key not in st.session_state:
            st.session_state[cluster_key] = min(
                max(
                    int(
                        st.session_state.get(
                            _LAST_PROJECTION_CLUSTER_K_KEY,
                            default_clusters,
                        )
                    ),
                    2,
                ),
                max_clusters,
            )
        n_clusters = st.slider(
            "K (clusters)",
            min_value=2,
            max_value=max_clusters,
            key=cluster_key,
        )
        mode_key = widget_key("load", "cluster_mode", scope, store_id(store))
        mode_options = list(_CLUSTER_MODES)
        selected_mode = _seed_selectbox_key(
            key=mode_key,
            remember_key=_LAST_PROJECTION_CLUSTER_MODE_KEY,
            options=mode_options,
            default=mode_options[0],
        )
        mode_label = st.selectbox(
            "Cluster fit",
            options=mode_options,
            index=mode_options.index(selected_mode),
            key=mode_key,
            help=(
                "Mean across layers is the previous behavior. First selected "
                "layer keeps one fixed clustering from the first frame. Per layer "
                "recomputes clustering for each animation frame."
            ),
        )
        st.session_state[_LAST_PROJECTION_CLUSTER_K_KEY] = n_clusters
        st.session_state[_LAST_PROJECTION_CLUSTER_MODE_KEY] = mode_label
        return ProjectionColorConfig(
            color_mode=color_mode,
            n_clusters=n_clusters,
            cluster_mode=_CLUSTER_MODES[mode_label],
        )

    if color_mode == "Persona attribute":
        persona_dataset = _synth_persona_dataset()
        attribute_options = list(synth_persona_attribute_names())
        if not attribute_options:
            st.info("No persona attributes are available for this dataset.")
            return None
        default_attribute = (
            attribute_options.index("sex") if "sex" in attribute_options else 0
        )
        attribute_key = widget_key("load", "attribute", scope, store_id(store))
        selected_attribute = _seed_selectbox_key(
            key=attribute_key,
            remember_key=_LAST_PROJECTION_ATTRIBUTE_KEY,
            options=attribute_options,
            default=attribute_options[default_attribute],
        )
        attribute_name = st.selectbox(
            "Attribute",
            options=attribute_options,
            index=attribute_options.index(selected_attribute),
            format_func=lambda name: attribute_display_label(persona_dataset, name),
            key=attribute_key,
        )
        st.session_state[_LAST_PROJECTION_ATTRIBUTE_KEY] = attribute_name
        info = persona_dataset.attribute_info(attribute_name)
        if info.get("high_cardinality"):
            st.caption(
                "High-cardinality categorical attributes are grouped to the "
                f"top {_MAX_ATTRIBUTE_CATEGORIES} values plus Other."
            )
        return ProjectionColorConfig(
            color_mode=color_mode,
            attribute_name=attribute_name,
        )

    highlight_persona_ids: tuple[str, ...] = ()
    if persona_ids:
        highlight_key = widget_key(
            "load", "persona_highlight", scope, store_id(store), persona_key
        )
        highlighted = st.multiselect(
            "Highlight personas",
            options=persona_ids,
            default=_remember_multiselect(
                key=highlight_key,
                remember_key=_LAST_PROJECTION_HIGHLIGHTS_KEY,
                options=persona_ids,
            ),
            format_func=lambda persona_id: _persona_display_label(
                persona_names, persona_id
            ),
            key=highlight_key,
            help=(
                "Select a few personas to keep their default colors while the rest "
                "are grayed out."
            ),
        )
        highlight_persona_ids = tuple(highlighted)
        st.session_state[_LAST_PROJECTION_HIGHLIGHTS_KEY] = list(highlighted)

    highlight_persona_key = (
        personas_fingerprint(highlight_persona_ids) if highlight_persona_ids else ""
    )

    return ProjectionColorConfig(
        color_mode=color_mode,
        highlight_persona_ids=highlight_persona_ids,
        highlight_persona_key=highlight_persona_key,
    )


def _layered_figure_state_keys(
    store: Store,
    mask_strategy: MaskStrategy,
    *,
    scope: str,
    figure_kind: str,
    n_components: int,
    color_config: ProjectionColorConfig,
    variant: str,
    persona_key: str,
    selected_layers: list[int],
    pair_trajectories: bool,
) -> LayeredFigureStateKeys:
    layer_key = "_".join(map(str, selected_layers))
    figure_key = widget_key(
        "load",
        f"{scope}_fig_state",
        store_id(store),
        store.model_name,
        mask_strategy.value,
        figure_kind,
        str(n_components),
        color_config.color_mode,
        str(color_config.attribute_name),
        str(color_config.n_clusters),
        str(color_config.cluster_mode),
        str(color_config.highlight_persona_key),
        variant,
        "persona_vector",
        persona_key,
        layer_key,
        str(pair_trajectories),
    )
    if figure_kind not in _PROJECTION_KINDS:
        return LayeredFigureStateKeys(figure=figure_key)

    graph_overlay = figure_kind == "isomap"
    projection_key = widget_key(
        "load",
        f"{scope}_projection_state",
        store_id(store),
        store.model_name,
        mask_strategy.value,
        figure_kind,
        str(n_components),
        str(graph_overlay),
        str(_DEFAULT_GRAPH_NEIGHBORS),
        variant,
        "persona_vector",
        persona_key,
        layer_key,
    )
    return LayeredFigureStateKeys(figure=figure_key, projection=projection_key)


def _projection_build_kwargs(
    samples,
    *,
    figure_kind: str,
    selected_layers: list[int],
    n_components: int,
    color_config: ProjectionColorConfig,
    persona_ids: list[str],
    persona_names: dict[str, str],
    projection_key: str | None,
) -> dict:
    if figure_kind not in _PROJECTION_KINDS:
        return {}

    graph_overlay = figure_kind == "isomap"
    build_kwargs = {
        "n_components": n_components,
        "graph_overlay": graph_overlay,
        "graph_n_neighbors": _DEFAULT_GRAPH_NEIGHBORS,
    }
    if color_config.n_clusters is not None:
        build_kwargs["n_clusters"] = color_config.n_clusters
        build_kwargs["cluster_mode"] = color_config.cluster_mode
    if projection_key is not None:
        projection_data = st.session_state.get(projection_key)
        if projection_data is None:
            projection_data = prepare_layered_projection_data(
                samples,
                figure_kind,
                layers=selected_layers,
                n_components=n_components,
                graph_overlay=graph_overlay,
                graph_n_neighbors=_DEFAULT_GRAPH_NEIGHBORS,
            )
            st.session_state[projection_key] = projection_data
        build_kwargs["projection_data"] = projection_data
    if color_config.attribute_name is not None:
        build_kwargs.update(
            attribute_color_kwargs(
                _synth_persona_dataset(),
                color_config.attribute_name,
                persona_ids,
                max_categories=_MAX_ATTRIBUTE_CATEGORIES,
            )
        )
    if color_config.color_mode == "Persona" and color_config.highlight_persona_ids:
        groups = _highlight_persona_groups(
            persona_ids,
            persona_names,
            color_config.highlight_persona_ids,
        )
        if groups is not None:
            build_kwargs["groups"] = groups
    return build_kwargs


def _build_layered_analysis_figures(
    samples,
    *,
    figure_kind: str,
    selected_layers: list[int],
    variant: str,
    title_fn: Callable[[str], str],
    pair_trajectories: bool,
    build_kwargs: dict,
) -> tuple[go.Figure, go.Figure | None]:
    if figure_kind == "similarity" and pair_trajectories:
        return build_similarity_figures(
            samples,
            layers=selected_layers,
            title=title_fn(variant),
            pair_title=(
                "Pair similarity trajectories - "
                f"{prompt_variant_label(variant)} - persona vectors"
            ),
        )

    main_fig = build_layered_figure(
        samples,
        figure_kind,
        layers=selected_layers,
        title=title_fn(variant),
        **build_kwargs,
    )
    if figure_kind in _PROJECTION_KINDS:
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
        if pair_trajectories
        else None
    )
    return main_fig, extra_fig


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
    remember_key: str = _LAST_PROJECTION_PERSONAS_KEY,
    default_count_limit: int = 500,
) -> None:
    """Render a single-variant layered analysis: select → button → figure(s).

    Used for similarity matrix, PCA, and UMAP. Set ``include_pair_trajectories``
    to add the pair-similarity-trajectory figure (similarity matrix only).
    """
    selected = _select_single_variant_samples(
        store,
        mask_strategy,
        scope,
        remember_key=remember_key,
        variant_remember_key=(
            _LAST_PROJECTION_VARIANT_KEY
            if figure_kind in _PROJECTION_KINDS
            else _LAST_SIMILARITY_VARIANT_KEY
        ),
        default_count_limit=default_count_limit,
    )
    if selected is None:
        return
    variant, persona_ids, persona_key, selected_layers = selected

    pair_trajectories = _render_pair_trajectory_control(
        enabled=include_pair_trajectories,
        persona_count=len(persona_ids),
        scope=scope,
        store=store,
    )
    if not _validate_layered_figure_size(
        figure_kind, len(persona_ids), selected_layers
    ):
        return

    color_config = ProjectionColorConfig()
    if figure_kind in _PROJECTION_KINDS:
        color_config = _render_projection_color_config(store, scope, persona_ids)
        if color_config is None:
            return

    state_keys = _layered_figure_state_keys(
        store,
        mask_strategy,
        scope=scope,
        figure_kind=figure_kind,
        n_components=n_components,
        color_config=color_config,
        variant=variant,
        persona_key=persona_key,
        selected_layers=selected_layers,
        pair_trajectories=pair_trajectories,
    )
    if state_keys.projection is not None:
        _clear_old_projection_states(state_keys.projection)
    filename = scope
    _clear_old_figure_states(state_keys.figure)
    persona_names = st.session_state.get(
        _persona_names_state_key(f"{scope}:{store_id(store)}"),
        {},
    )

    if st.button(button_label, type="primary"):
        build_label = {
            "umap": "Computing UMAP projections…",
            "pca": "Computing PCA projections…",
            "isomap": "Computing Isomap projections…",
            "similarity": "Computing similarity matrices…",
        }.get(figure_kind, "Building figure…")
        progress = st.progress(0, text="Loading activation vectors…")
        try:
            progress.progress(15, text="Loading activation vectors…")
            samples = _load_persona_vectors(
                store,
                variant,
                mask_strategy,
                persona_ids,
            )
            progress.progress(55, text=build_label)
            build_kwargs = _projection_build_kwargs(
                samples,
                figure_kind=figure_kind,
                selected_layers=selected_layers,
                n_components=n_components,
                color_config=color_config,
                persona_ids=persona_ids,
                persona_names=persona_names,
                projection_key=state_keys.projection,
            )
            main_fig, extra_fig = _build_layered_analysis_figures(
                samples,
                figure_kind=figure_kind,
                selected_layers=selected_layers,
                variant=variant,
                title_fn=title_fn,
                pair_trajectories=pair_trajectories,
                build_kwargs=build_kwargs,
            )
            if (
                color_config.color_mode == "Persona"
                and color_config.highlight_persona_ids
            ):
                _gray_out_unselected_personas(main_fig)
            progress.progress(90, text="Storing figure state…")
            n_samples = samples.vectors.shape[0]
            del samples
            _store_figure_state(state_keys.figure, (main_fig, extra_fig, n_samples))
            progress.progress(100, text="Done.")
        except Exception as exc:
            st.error(f"Could not build figure: {exc}")
            st.session_state.pop(state_keys.figure, None)
        finally:
            _release_vector_memory(store, [variant])
            progress.empty()

    if state_keys.figure in st.session_state:
        main_fig, extra_fig, n_samples = st.session_state[state_keys.figure]
        _plotly_chart(main_fig)
        figs = [main_fig]
        filenames = [filename]
        if extra_fig is not None:
            st.subheader("Pair trajectories")
            _plotly_chart(extra_fig)
            figs.append(extra_fig)
            filenames.append(f"{filename}__pair_trajectories")
        _render_save_buttons(figs, filenames, scope)
        st.success(f"Loaded {n_samples} samples.")


_LAST_DENDRO_PERSONAS_KEY = "analysis:last_personas:dendro"
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
    persona_ids = _select_artifact_personas(
        store,
        shared_variants,
        mask_strategy,
        widget_scope=f"dendro:{store_id(store)}",
        remember_key=_LAST_DENDRO_PERSONAS_KEY,
        default_count_limit=_DEFAULT_PERSONA_LIMITS["dendro"],
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
            progress.progress(15, text="Loading first variant vectors…")
            samples_a = _load_persona_vectors(
                store,
                variant_a,
                mask_strategy,
                persona_ids,
            )
            progress.progress(40, text="Building first dendrogram…")
            fig_a = plot_persona_dendrogram(
                samples_a,
                layered=layered_mode,
                layers=selected_layers,
                linkage=linkage,
                title=f"Dendrogram — {prompt_variant_label(variant_a)}",
            )
            fig_a.update_layout(height=750)
            del samples_a
            fig_b = None
            if variant_a != variant_b:
                progress.progress(60, text="Loading second variant vectors…")
                samples_b = _load_persona_vectors(
                    store,
                    variant_b,
                    mask_strategy,
                    persona_ids,
                )
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
            progress.progress(90, text="Storing figure state…")
            _store_figure_state(
                fig_key,
                (fig_a, fig_b, len(persona_ids), variant_a, variant_b),
            )
            progress.progress(100, text="Done.")
        except Exception as exc:
            st.error(f"Could not build dendrogram: {exc}")
            st.session_state.pop(fig_key, None)
        finally:
            _release_vector_memory(store, shared_variants)
            progress.empty()

    if fig_key in st.session_state:
        fig_a, fig_b, n_personas, va, vb = st.session_state[fig_key]
        if fig_b is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(prompt_variant_label(va))
                _plotly_chart(fig_a)
            with col_b:
                st.subheader(prompt_variant_label(vb))
                _plotly_chart(fig_b)
        else:
            _plotly_chart(fig_a)

        figs = [fig_a] + ([fig_b] if fig_b else [])
        filenames = [
            _filename("dendro", store.model_name, mask_strategy.value, va),
            *(
                [_filename("dendro", store.model_name, mask_strategy.value, vb)]
                if fig_b
                else []
            ),
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
            key="analysis:hub_model_fallback",
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
    fallback_model = st.session_state.get("analysis:local_model", DEFAULT_COMPARE_MODEL)
    model_options = local_model_options_cached(artifacts_root, mask_strategy.value)
    if not model_options:
        return st.text_input(
            "Local model",
            value=fallback_model,
            key="analysis:local_model",
            help="Compare-only local model id or path.",
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
            help="Compare-only local model id or path.",
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
