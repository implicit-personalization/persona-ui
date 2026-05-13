import os

import streamlit as st
from persona_vectors.analysis import LayeredSamples, load_persona_vectors
from persona_vectors.artifacts import (
    ActivationStore,
    HFActivationStore,
    discover_activation_models,
    model_dir_name,
)
from persona_vectors.extraction import MaskStrategy
from persona_vectors.hub import list_hub_vector_models

Store = ActivationStore | HFActivationStore

DEFAULT_HUB_REPO = os.environ.get(
    "PERSONA_VECTORS_HUB_REPO",
    "implicit-personalization/synth-persona-vectors",
)
DEFAULT_COMPARE_MODEL = os.environ.get("DEFAULT_MODEL", "google/gemma-2-2b-it")
SOURCE_HUB = "Hugging Face Hub"
SOURCE_LOCAL = "Local activations"
SOURCES = (SOURCE_HUB, SOURCE_LOCAL)


@st.cache_resource(show_spinner=False, max_entries=1)
def activation_store_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
) -> Store:
    mask_strategy = MaskStrategy(mask_strategy_value)
    if source == SOURCE_HUB:
        return HFActivationStore(location, model_name, mask_strategy=mask_strategy)
    return ActivationStore(model_name, location, mask_strategy=mask_strategy)


@st.cache_data(show_spinner=False)
def available_variants_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
) -> list[str]:
    return activation_store_cached(
        source, location, model_name, mask_strategy_value
    ).available_variants()


@st.cache_data(show_spinner=False)
def personas_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
) -> list[str]:
    return activation_store_cached(
        source, location, model_name, mask_strategy_value
    ).list_personas(list(variants))


@st.cache_data(show_spinner=False)
def persona_names_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
    persona_ids: tuple[str, ...],
) -> dict[str, str]:
    store = activation_store_cached(source, location, model_name, mask_strategy_value)
    names = store.persona_names(list(persona_ids), variants=list(variants))
    # Preserve input order, fall back to the id when the row has no display name.
    return {pid: names.get(pid, pid) for pid in persona_ids}


@st.cache_data(show_spinner=False)
def store_layers_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
    persona_ids: tuple[str, ...],
) -> list[int]:
    return activation_store_cached(
        source, location, model_name, mask_strategy_value
    ).list_layers(list(variants), list(persona_ids))


@st.cache_data(show_spinner=False)
def local_model_options_cached(
    artifacts_root: str, mask_strategy_value: str
) -> list[str]:
    return discover_activation_models(artifacts_root, mask_strategy_value)


@st.cache_data(show_spinner=False)
def hub_models_by_mask_strategy(repo_id: str) -> dict[MaskStrategy, list[str]]:
    valid = {strategy.value for strategy in MaskStrategy}
    return {
        MaskStrategy(strategy_value): models
        for strategy_value, models in list_hub_vector_models(repo_id).items()
        if strategy_value in valid
    }


def store_cache_parts(store: Store) -> tuple[str, str, str]:
    if isinstance(store, HFActivationStore):
        return SOURCE_HUB, store.repo_id, store.model_name
    return SOURCE_LOCAL, str(store.root_dir), store.model_name


def store_id(store: Store) -> str:
    if isinstance(store, HFActivationStore):
        return f"hub:{store.repo_id}"
    return f"local:{store.root_dir}"


def available_variants(store: Store, mask_strategy: MaskStrategy) -> list[str]:
    source, location, model_name = store_cache_parts(store)
    return available_variants_cached(source, location, model_name, mask_strategy.value)


def local_model_matches(left: str, right: str) -> bool:
    return model_dir_name(left) == model_dir_name(right)


def load_persona_vectors_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variant: str,
    persona_ids: tuple[str, ...],
) -> LayeredSamples:
    store = activation_store_cached(source, location, model_name, mask_strategy_value)
    return load_persona_vectors(
        store,
        variant,
        mask_strategy=MaskStrategy(mask_strategy_value),
        persona_ids=list(persona_ids),
    )


def load_variant_vectors_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
    persona_ids: tuple[str, ...],
) -> dict[str, LayeredSamples]:
    return {
        variant: load_persona_vectors_cached(
            source, location, model_name, mask_strategy_value, variant, persona_ids
        )
        for variant in variants
    }


def release_hf_store_cache(
    store: Store,
    variants: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Drop cached HF data for ``variants`` (or all) on Hub stores."""
    release_cache = getattr(store, "release_cache", None)
    if isinstance(store, HFActivationStore) and callable(release_cache):
        release_cache(variants)
