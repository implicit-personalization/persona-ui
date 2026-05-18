import os

import streamlit as st
from persona_vectors.analysis import (
    AnalysisDataset,
    LayeredSamples,
    load_analysis_dataset,
)
from persona_vectors.artifacts import (
    PersonaVectorStore,
    HFPersonaVectorStore,
    discover_activation_models,
    model_dir_name,
)
from persona_vectors.extraction import MaskStrategy
from persona_vectors.hub import list_hub_vector_models
from persona_vectors.plots import (
    LayeredProjectionData,
    prepare_kmeans_groups,
    prepare_layered_projection_data,
)

from utils.helpers import env_int

Store = PersonaVectorStore | HFPersonaVectorStore

DEFAULT_HUB_REPO = os.environ.get(
    "PERSONA_VECTORS_HUB_REPO",
    "implicit-personalization/synth-persona-vectors",
)
DEFAULT_COMPARE_MODEL = os.environ.get("DEFAULT_MODEL", "google/gemma-2-2b-it")
SOURCE_HUB = "Hugging Face Hub"
SOURCE_LOCAL = "Local artifacts"
SOURCES = (SOURCE_HUB, SOURCE_LOCAL)


_STORE_CACHE_ENTRIES = env_int("PERSONA_UI_STORE_CACHE_ENTRIES", 4)
_VECTOR_CACHE_ENTRIES = env_int("PERSONA_UI_VECTOR_CACHE_ENTRIES", 4)
_PREPARED_CACHE_ENTRIES = env_int("PERSONA_UI_PREPARED_CACHE_ENTRIES", 8)


@st.cache_resource(show_spinner=False, max_entries=_STORE_CACHE_ENTRIES)
def activation_store_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
) -> Store:
    mask_strategy = MaskStrategy(mask_strategy_value)
    if source == SOURCE_HUB:
        return HFPersonaVectorStore(location, model_name, mask_strategy=mask_strategy)
    return PersonaVectorStore(model_name, location, mask_strategy=mask_strategy)


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
    *,
    include_baseline: bool = False,
) -> list[str]:
    return activation_store_cached(
        source, location, model_name, mask_strategy_value
    ).list_personas(list(variants), include_baseline=include_baseline)


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
    if isinstance(store, HFPersonaVectorStore):
        return SOURCE_HUB, store.repo_id, store.model_name
    return SOURCE_LOCAL, str(store.root_dir), store.model_name


def store_id(store: Store) -> str:
    if isinstance(store, HFPersonaVectorStore):
        return f"hub:{store.repo_id}"
    return f"local:{store.root_dir}"


def available_variants(store: Store, mask_strategy: MaskStrategy) -> list[str]:
    source, location, model_name = store_cache_parts(store)
    return available_variants_cached(source, location, model_name, mask_strategy.value)


def local_model_matches(left: str, right: str) -> bool:
    return model_dir_name(left) == model_dir_name(right)


@st.cache_resource(show_spinner=False, max_entries=_VECTOR_CACHE_ENTRIES)
def load_analysis_dataset_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
    persona_ids: tuple[str, ...],
) -> AnalysisDataset:
    store = activation_store_cached(source, location, model_name, mask_strategy_value)
    return load_analysis_dataset(
        store,
        variants,
        mask_strategy=MaskStrategy(mask_strategy_value),
        persona_ids=persona_ids,
    )


def load_persona_vectors_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variant: str,
    persona_ids: tuple[str, ...],
) -> LayeredSamples:
    return load_analysis_dataset_cached(
        source,
        location,
        model_name,
        mask_strategy_value,
        (variant,),
        persona_ids,
    ).samples(variant)


def load_variant_vectors_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
    persona_ids: tuple[str, ...],
) -> dict[str, LayeredSamples]:
    return load_analysis_dataset_cached(
        source,
        location,
        model_name,
        mask_strategy_value,
        variants,
        persona_ids,
    ).samples_by_variant


@st.cache_resource(show_spinner=False, max_entries=_PREPARED_CACHE_ENTRIES)
def projection_data_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variant: str,
    persona_ids: tuple[str, ...],
    layers: tuple[int, ...],
    kind: str,
    n_components: int,
    graph_overlay: bool,
    graph_n_neighbors: int,
) -> LayeredProjectionData:
    samples = load_persona_vectors_cached(
        source, location, model_name, mask_strategy_value, variant, persona_ids
    )
    return prepare_layered_projection_data(
        samples,
        kind,
        layers=list(layers),
        n_components=n_components,
        graph_overlay=graph_overlay,
        graph_n_neighbors=graph_n_neighbors,
    )


@st.cache_resource(show_spinner=False, max_entries=_PREPARED_CACHE_ENTRIES)
def kmeans_groups_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variant: str,
    persona_ids: tuple[str, ...],
    layers: tuple[int, ...],
    n_clusters: int,
    cluster_mode: str,
) -> list[str] | dict[int, list[str]]:
    samples = load_persona_vectors_cached(
        source, location, model_name, mask_strategy_value, variant, persona_ids
    )
    return prepare_kmeans_groups(
        samples,
        layers=list(layers),
        n_clusters=n_clusters,
        cluster_mode=cluster_mode,
    )


def prefetch_hub_metadata(
    repo_id: str,
    model_name: str,
    mask_strategy_value: str,
    variant: str | None = None,
) -> None:
    """Warm small Hub metadata caches without loading full activation tensors."""
    if not repo_id or not model_name or not mask_strategy_value:
        return
    hub_models_by_mask_strategy(repo_id)
    available_variants_cached(
        SOURCE_HUB,
        repo_id,
        model_name,
        mask_strategy_value,
    )
    if variant:
        personas_cached(
            SOURCE_HUB,
            repo_id,
            model_name,
            mask_strategy_value,
            (variant,),
        )
