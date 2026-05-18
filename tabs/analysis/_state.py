from dataclasses import dataclass

import streamlit as st
from persona_data.synth_persona import BASELINE_PERSONA_ID
from persona_vectors.attributes import DEFAULT_MAX_ATTRIBUTE_CATEGORIES

from utils.helpers import env_int, slugify, widget_key


def _filename(*parts: str) -> str:
    return "__".join(slugify(part) for part in parts if part)


# Keep analysis-tab selection state separate so projection defaults do not
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
    "similarity": 20,
    "pca": 500,
    "umap": 500,
    "isomap": 500,
    "dendro": 20,
}
_MAX_PERSONA_COUNTS = {
    "similarity": 100,
    "dendro": 100,
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
_PROJECTION_COLOR_MODES = ["Persona attribute", "Persona", "K-means clusters"]
_MAX_ATTRIBUTE_CATEGORIES = DEFAULT_MAX_ATTRIBUTE_CATEGORIES


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
    prepared: str | None = None


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


_TRACKED_STATE_KEYS_KEY = "analysis:_tracked_state_keys"
_FIGURE_STATE_ENTRIES = env_int("PERSONA_UI_FIGURE_STATE_ENTRIES", 2)
_PREPARED_STATE_ENTRIES = env_int("PERSONA_UI_PREPARED_STATE_ENTRIES", 4)


def _touch_load_state(current_key: str, suffix: str, *, max_entries: int) -> None:
    # Keep a tiny MRU window of heavy state instead of scanning all of
    # session_state or retaining every figure forever. This makes nearby
    # method-switching feel warm while still giving RAM a hard ceiling.
    tracked: dict[str, list[str]] = st.session_state.setdefault(
        _TRACKED_STATE_KEYS_KEY, {}
    )
    keys = [key for key in tracked.get(suffix, []) if key != current_key]
    keys.append(current_key)
    while len(keys) > max(1, max_entries):
        st.session_state.pop(keys.pop(0), None)
    tracked[suffix] = keys


def _clear_old_figure_states(current_key: str) -> None:
    _touch_load_state(
        current_key,
        "_fig_state",
        max_entries=_FIGURE_STATE_ENTRIES,
    )


def _clear_old_prepared_states(current_key: str) -> None:
    _touch_load_state(
        current_key,
        "_projection_ready",
        max_entries=_PREPARED_STATE_ENTRIES,
    )


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


def _remembered_selectbox(
    label: str,
    *,
    key: str,
    remember_key: str,
    options: list[str],
    default: str,
    **selectbox_kwargs: object,
) -> str:
    selected = _seed_selectbox_key(
        key=key,
        remember_key=remember_key,
        options=options,
        default=default,
    )
    choice = st.selectbox(
        label,
        options=options,
        index=options.index(selected),
        key=key,
        **selectbox_kwargs,
    )
    st.session_state[remember_key] = choice
    return choice


def _personas_empty_message(variants: list[str]) -> str:
    if len(variants) > 1:
        return (
            "No personas have vectors for all selected variants. "
            "Pick a single variant or change the source."
        )
    return "No personas found for this model and variant."


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
