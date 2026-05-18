import gc

import plotly.graph_objects as go
import streamlit as st
from persona_data.synth_persona import BASELINE_PERSONA_ID
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import save_plot_html

from tabs.analysis._state import (
    _DEFAULT_LAYER_FRAMES,
    _HIGHLIGHT_OTHER_COLOR,
    _HIGHLIGHT_OTHER_LABEL,
    _LAST_LAYER_FRAMES_KEY,
    _LAST_MASK_STRATEGY_KEY,
    PersonaOptions,
    _is_assistant_persona,
    _persona_names_state_key,
    _personas_empty_message,
    _remembered_selectbox,
    _sequence_to_list,
)
from utils.analysis_sources import (
    Store,
    available_variants,
    load_persona_vectors_cached,
    load_variant_vectors_cached,
    persona_names_cached,
    personas_cached,
    store_cache_parts,
    store_id,
    store_layers_cached,
)
from utils.controls import render_mask_strategy_select
from utils.helpers import personas_fingerprint, prompt_variant_label, widget_key
from utils.theme import active_base, style_plotly_layer_controls


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


def _release_vector_memory() -> None:
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
        include_baseline=True,
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
    max_count_limit: int | None = None,
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
        max_count = (
            min(max_count_limit, len(options.regular_ids))
            if max_count_limit is not None
            else len(options.regular_ids)
        )
        persona_count = st.slider(
            "Personas",
            min_value=0 if options.assistant_id is not None else 1,
            max_value=max_count,
            value=min(default_count, max_count),
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
    max_count_limit: int | None = None,
) -> list[str]:
    empty_message = _personas_empty_message(variants)
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
        max_count_limit=max_count_limit,
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


def _render_persona_select_controls(
    options: PersonaOptions,
    widget_scope: str,
    *,
    max_selections: int | None = None,
) -> list[str]:
    select_key = widget_key("load", "persona_select", widget_scope)
    assistant_key = widget_key("load", "persona_select_assistant", widget_scope)

    label_map = {
        persona_id: f"{options.persona_names.get(persona_id, persona_id)} ({persona_id})"
        for persona_id in options.regular_ids
    }
    sorted_labels = sorted(label_map.values())
    selected_labels = st.multiselect(
        "Select personas",
        options=sorted_labels,
        key=select_key,
        placeholder="Search and select personas...",
        max_selections=max_selections,
    )
    label_to_id = {label: persona_id for persona_id, label in label_map.items()}
    selected_ids = [label_to_id[label] for label in selected_labels]

    if options.assistant_id is not None:
        include_assistant = st.checkbox(
            "Include Assistant persona",
            key=assistant_key,
        )
        if include_assistant:
            selected_ids.append(options.assistant_id)

    st.session_state[_persona_names_state_key(widget_scope)] = dict(
        options.persona_names
    )

    if not selected_ids:
        st.info("Select at least one persona.")

    return selected_ids


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
    base = active_base()
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
        remember_key="source:last_mask_strategy",
        help_text="Which extracted activation set to load.",
    )


def _select_single_variant_samples(
    store: Store,
    mask_strategy: MaskStrategy,
    scope: str,
    *,
    remember_key: str,
    variant_remember_key: str,
    default_count_limit: int,
    max_count_limit: int | None = None,
    allow_specific_personas: bool = False,
) -> tuple[str, list[str], str, list[int]] | None:
    variants = available_variants(store, mask_strategy)
    if not variants:
        st.info("No variants with saved vectors for this model.")
        return None
    variant_key = widget_key("load", "variant", scope, store_id(store))
    default_variant = "biography" if "biography" in variants else variants[0]
    variant = _remembered_selectbox(
        "Variant",
        key=variant_key,
        remember_key=variant_remember_key,
        options=variants,
        default=default_variant,
        format_func=prompt_variant_label,
    )
    widget_scope = f"{scope}:{store_id(store)}"
    select_specific = False
    if allow_specific_personas:
        select_specific = st.toggle(
            "Select specific personas",
            value=False,
            key=widget_key("load", "select_specific_personas", scope, store_id(store)),
            help="Search and select specific personas instead of using the first N.",
        )

    if select_specific:
        options = _load_persona_options(
            store,
            [variant],
            mask_strategy,
            empty_message=_personas_empty_message([variant]),
        )
        if options is None:
            st.session_state.pop(_persona_names_state_key(widget_scope), None)
            return None
        persona_ids = _render_persona_select_controls(
            options,
            widget_scope,
            max_selections=max_count_limit,
        )
    else:
        persona_ids = _select_artifact_personas(
            store,
            [variant],
            mask_strategy,
            widget_scope=widget_scope,
            remember_key=remember_key,
            default_count_limit=default_count_limit,
            max_count_limit=max_count_limit,
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
