"""Compare-mode steering: build a persona-axis steering vector and render its controls.

The steering vector is the difference-of-means direction of an attribute (age,
education, sex, wealth, politics) in persona-vector space — the same axes the
direction-geometry analysis found to be stable and causal. We build it from the
model's precomputed ``templated`` activations on the Hub and inject it during
generation (see ``persona_vectors.steer_generate``). Only models that have Hub
vectors can be steered; others disable the controls.
"""

from __future__ import annotations

import streamlit as st

from persona_vectors.artifacts import HFPersonaVectorStore
from persona_vectors.extraction import MaskStrategy
from persona_vectors.steer_generate import (
    SteeringSpec,
    build_attribute_direction,
    build_steering_spec,
)

from utils.helpers import widget_key

STEER_REPO = "implicit-personalization/synth-persona-vectors"
# Per-attribute causal-sign flips. The difference-of-means axis is causally correct
# as-is — +coefficient steers toward the labelled direction (verified for age in
# generation + MCQ) — so this is empty by default. Add an entry only if an axis is
# measured to decode an attribute but steer the opposite way. The UI flip toggle
# defaults from this and the user can override per layer.
STEER_SIGN_CAL: dict[str, float] = {}
# Persona attributes that form a single signed steering axis. Binary and
# numeric/ordinal attributes split naturally; categorical attributes use a
# one-vs-rest contrast against their modal class (handled in
# build_attribute_direction). High-cardinality nominals (city, state, ethnicity)
# are omitted — too few personas per class for a reliable direction.
STEER_ATTRIBUTES = [
    "age",
    "sex",
    "highest_degree_received",
    "total_wealth",
    "political_views",
    "party_identification",
    "born_in_us",
    "us_citizenship_status",
    "speak_other_language",
    "race",
    "religion",
    "marital_status",
    "work_status",
]


@st.cache_resource(show_spinner=False)
def _cached_direction(model_name: str, attribute: str, layer: int | None = None) -> dict | None:
    """Build (and cache) the attribute difference-of-means axis for a model.

    ``layer=None`` builds over a mid-network band and keeps the best-separating
    layer (the default). Passing a specific ``layer`` builds the direction *at that
    layer* (the direction is layer-specific) — used by the UI's layer selector.
    Returns ``None`` when the model has no precomputed Hub persona vectors, so
    callers can disable steering gracefully. Cached per (model, attribute, layer).
    """
    from persona_data.synth_persona import SynthPersonaDataset

    try:
        store = HFPersonaVectorStore(
            STEER_REPO, model_name, mask_strategy=MaskStrategy.ANSWER_MEAN
        )
        ids = store.list_personas(["templated"], include_baseline=False)
    except Exception:
        return None
    if not ids:
        return None

    n_layers = int(store.load("templated", ids[0]).shape[0])
    band = list(range(8, n_layers, 4)) or [n_layers // 2]
    candidate_layers = [layer] if layer is not None else band
    info = build_attribute_direction(
        store,
        SynthPersonaDataset(),
        attribute,
        variant="templated",
        candidate_layers=candidate_layers,
        persona_ids=ids,
    )
    return {
        "layer": info["layer"],
        "unit_direction": info["unit_direction"],
        "gap_norm": info["gap_norm"],
        "act_norm": info["act_norm"],
        "auc": info["auc"],
        "n": info["n_personas"],
        "positive": info["positive"],
        "band": band,
    }


def render_steering_controls(scope_key: str, model_name: str) -> SteeringSpec | None:
    """Render axis + strength controls and return a ready SteeringSpec (or None).

    None means "no steering" — disabled, strength 0, or the model has no vectors.
    """
    attribute = st.selectbox(
        "Steer axis",
        STEER_ATTRIBUTES,
        key=widget_key(scope_key, "steer_attr"),
    )
    strength = st.slider(
        "Strength (× class gap)",
        min_value=-3.0,
        max_value=3.0,
        value=1.0,
        step=0.25,
        key=widget_key(scope_key, "steer_strength"),
        help="In gap units: 1 = the opposite-class centroid. ~1–2 = coherent shift; "
        "large values degrade. (Scaling by the residual norm over-steers ~8x.)",
    )

    # Build at the best layer first to learn the candidate band + default layer.
    with st.spinner("Loading steering axis..."):
        base = _cached_direction(model_name, attribute)
    if base is None:
        st.info(
            f"`{model_name}` has no precomputed persona vectors on the Hub — "
            "steering unavailable for this model."
        )
        return None

    band = base["band"]
    layer = st.select_slider(
        "Layer",
        options=band,
        value=base["layer"],
        key=widget_key(scope_key, "steer_layer"),
        help=f"Residual-stream layer to steer. Default {base['layer']} separates "
        "this attribute best; other layers can have a different (even flipped) effect.",
    )
    # Re-fetch the direction at the chosen layer (the axis is layer-specific).
    if layer == base["layer"]:
        info = base
    else:
        with st.spinner(f"Loading axis at layer {layer}..."):
            info = _cached_direction(model_name, attribute, int(layer))
    if info is None:
        return None

    flip = st.checkbox(
        "Flip steering sign",
        value=STEER_SIGN_CAL.get(attribute, 1.0) < 0,
        key=widget_key(scope_key, "steer_flip"),
        help="Some axes decode an attribute but causally steer the opposite way "
        "(e.g. age). Flip so + pushes toward the labelled direction.",
    )
    sign = -1.0 if flip else 1.0

    st.caption(
        f"**+** steers toward _{info['positive']}_{' (flipped)' if flip else ''} · "
        f"axis AUC {info['auc']:.2f} · layer {info['layer']} · n={info['n']} personas"
    )
    if strength == 0:
        return None
    return build_steering_spec(info, strength, sign=sign)
