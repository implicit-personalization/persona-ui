import os

import streamlit as st
import torch
from persona_vectors.analysis import LayeredSamples
from persona_vectors.artifacts import (
    ActivationStore,
    HFActivationStore,
    activation_config_name,
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


def _hub_split(repo_id: str, model_name: str, mask_strategy_value: str, variant: str):
    from datasets import load_dataset

    return load_dataset(
        repo_id,
        name=activation_config_name(model_name, mask_strategy_value),
        split=variant,
        keep_in_memory=False,
    )


def _hub_split_columns(
    repo_id: str,
    model_name: str,
    mask_strategy_value: str,
    variant: str,
    columns: list[str],
):
    dataset = _hub_split(repo_id, model_name, mask_strategy_value, variant)
    return dataset.select_columns(columns)


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
    store = activation_store_cached(source, location, model_name, mask_strategy_value)
    return store.available_variants()


@st.cache_data(show_spinner=False)
def personas_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
) -> list[str]:
    if source == SOURCE_HUB:
        variant_ids = [
            list(
                _hub_split_columns(
                    location,
                    model_name,
                    mask_strategy_value,
                    variant,
                    ["persona_id"],
                )["persona_id"]
            )
            for variant in variants
        ]
        if not variant_ids:
            return []
        shared = set(variant_ids[0])
        for ids in variant_ids[1:]:
            shared &= set(ids)
        return [persona_id for persona_id in variant_ids[0] if persona_id in shared]

    store = activation_store_cached(source, location, model_name, mask_strategy_value)
    return store.list_personas(
        list(variants),
        mask_strategy=MaskStrategy(mask_strategy_value),
    )


@st.cache_data(show_spinner=False)
def persona_names_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
    persona_ids: tuple[str, ...],
) -> dict[str, str]:
    if source == SOURCE_HUB:
        requested = set(persona_ids)
        names: dict[str, str] = {}
        for variant in variants:
            metadata = _hub_split_columns(
                location,
                model_name,
                mask_strategy_value,
                variant,
                ["persona_id", "name"],
            )
            for row in metadata:
                persona_id = row["persona_id"]
                if persona_id in requested and persona_id not in names:
                    names[persona_id] = row.get("name") or persona_id
                    if len(names) == len(requested):
                        return {pid: names.get(pid, pid) for pid in persona_ids}
        return {pid: names.get(pid, pid) for pid in persona_ids}

    store = activation_store_cached(source, location, model_name, mask_strategy_value)
    return store.persona_names(
        list(persona_ids),
        variants=list(variants),
        mask_strategy=MaskStrategy(mask_strategy_value),
    )


@st.cache_data(show_spinner=False)
def local_model_options_cached(
    artifacts_root: str, mask_strategy_value: str
) -> list[str]:
    return discover_activation_models(artifacts_root, mask_strategy_value)


@st.cache_data(show_spinner=False)
def hub_models_by_mask_strategy(repo_id: str) -> dict[MaskStrategy, list[str]]:
    raw = list_hub_vector_models(repo_id)
    return {
        MaskStrategy(strategy_value): models
        for strategy_value, models in raw.items()
        if strategy_value in {strategy.value for strategy in MaskStrategy}
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
    return available_variants_cached(
        source,
        location,
        model_name,
        mask_strategy.value,
    )


@st.cache_data(show_spinner=False)
def store_layers_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
    persona_ids: tuple[str, ...],
) -> list[int]:
    if source == SOURCE_HUB:
        shared_layers: set[int] | None = None
        requested = list(persona_ids)
        for variant in variants:
            dataset = _hub_split(location, model_name, mask_strategy_value, variant)
            ids = list(dataset.select_columns(["persona_id"])["persona_id"])
            sample_id = requested[0] if requested else (ids[0] if ids else None)
            if sample_id is None:
                return []
            if requested and any(persona_id not in ids for persona_id in requested):
                return []
            vector = torch.as_tensor(dataset[ids.index(sample_id)]["vector"])
            if vector.ndim != 2:
                raise ValueError(
                    f"tensor for {sample_id!r} must have shape (num_layers, hidden_size)"
                )
            layers = set(range(int(vector.shape[0])))
            shared_layers = layers if shared_layers is None else shared_layers & layers
        return sorted(shared_layers or set())

    store = activation_store_cached(source, location, model_name, mask_strategy_value)
    return store.list_layers(
        list(variants),
        list(persona_ids),
        mask_strategy=MaskStrategy(mask_strategy_value),
    )


def local_model_matches(left: str, right: str) -> bool:
    return model_dir_name(left) == model_dir_name(right)


def load_persona_vectors_lean(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variant: str,
    persona_ids: tuple[str, ...],
) -> LayeredSamples:
    if source != SOURCE_HUB:
        from persona_vectors.analysis import load_persona_vectors

        store = activation_store_cached(
            source,
            location,
            model_name,
            mask_strategy_value,
        )
        return load_persona_vectors(
            store,
            variant,
            mask_strategy=MaskStrategy(mask_strategy_value),
            persona_ids=list(persona_ids),
        )

    dataset = _hub_split(location, model_name, mask_strategy_value, variant)
    metadata = dataset.select_columns(["persona_id", "name"])
    index_by_id: dict[str, int] = {}
    name_by_id: dict[str, str] = {}
    requested = set(persona_ids)
    for index, row in enumerate(metadata):
        persona_id = row["persona_id"]
        if persona_id in requested:
            index_by_id[persona_id] = index
            name_by_id[persona_id] = row.get("name") or persona_id
            if len(index_by_id) == len(requested):
                break

    missing = [
        persona_id for persona_id in persona_ids if persona_id not in index_by_id
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} persona vector(s) in {variant!r}: {missing[:3]}"
        )

    vectors, labels, hover_text = [], [], []
    for persona_id in persona_ids:
        name = name_by_id.get(persona_id, persona_id)
        vector = torch.as_tensor(
            dataset[index_by_id[persona_id]]["vector"],
            dtype=torch.float32,
        )
        if vector.ndim != 2:
            raise ValueError(
                f"tensor for {persona_id!r} must have shape (num_layers, hidden_size)"
            )
        vectors.append(vector)
        labels.append(name)
        hover_text.append(f"Persona: {name}<br>ID: {persona_id}")
    return LayeredSamples(torch.stack(vectors), labels, hover_text)


def load_variant_vectors_lean(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
    persona_ids: tuple[str, ...],
) -> dict[str, LayeredSamples]:
    return {
        variant: load_persona_vectors_lean(
            source,
            location,
            model_name,
            mask_strategy_value,
            variant,
            persona_ids,
        )
        for variant in variants
    }


def release_store_cache(
    store: Store,
    variants: list[str] | tuple[str, ...] | None = None,
) -> None:
    cache = getattr(store, "_cache", None)
    if not isinstance(cache, dict):
        return
    if variants is None:
        cache.clear()
        return
    for variant in variants:
        cache.pop(variant, None)
