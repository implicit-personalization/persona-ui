import os
from pathlib import Path

import streamlit as st
from persona_vectors.artifacts import ActivationStore, HFActivationStore
from persona_vectors.artifacts import list_layers as list_local_layers
from persona_vectors.artifacts import model_dir_name
from persona_vectors.extraction import MaskStrategy

Store = ActivationStore | HFActivationStore

DEFAULT_HUB_REPO = os.environ.get(
    "PERSONA_VECTORS_HUB_REPO",
    "implicit-personalization/synth-persona-vectors",
)
DEFAULT_COMPARE_MODEL = os.environ.get("DEFAULT_MODEL", "google/gemma-2-2b-it")
SOURCE_HUB = "Hugging Face Hub"
SOURCE_LOCAL = "Local activations"
SOURCES = (SOURCE_HUB, SOURCE_LOCAL)

list_layers_cached = st.cache_data(show_spinner=False)(list_local_layers)


@st.cache_resource(show_spinner=False)
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


@st.cache_data(show_spinner=False, ttl=10)
def available_variants_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
) -> list[str]:
    store = activation_store_cached(source, location, model_name, mask_strategy_value)
    return store.available_variants()


@st.cache_data(show_spinner=False, ttl=10)
def personas_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
) -> list[str]:
    store = activation_store_cached(source, location, model_name, mask_strategy_value)
    return store.list_personas(
        list(variants),
        mask_strategy=MaskStrategy(mask_strategy_value),
    )


@st.cache_data(show_spinner=False, ttl=10)
def persona_names_cached(
    source: str,
    location: str,
    model_name: str,
    mask_strategy_value: str,
    variants: tuple[str, ...],
    persona_ids: tuple[str, ...],
) -> dict[str, str]:
    store = activation_store_cached(source, location, model_name, mask_strategy_value)
    return store.persona_names(
        list(persona_ids),
        variants=list(variants),
        mask_strategy=MaskStrategy(mask_strategy_value),
    )


@st.cache_data(show_spinner=False, ttl=10)
def local_model_options_cached(
    artifacts_root: str, mask_strategy_value: str
) -> list[str]:
    root = Path(artifacts_root).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    options = []
    try:
        model_roots = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        return []

    for model_root in model_roots:
        strategy_root = model_root / mask_strategy_value
        if not strategy_root.is_dir():
            continue
        variant_roots = (
            variant_root
            for variant_root in strategy_root.iterdir()
            if variant_root.is_dir()
        )
        if any(
            (variant_root / "manifest.json").exists() for variant_root in variant_roots
        ):
            options.append(model_root.name.replace("__", "/"))
    return options


@st.cache_data(show_spinner=False)
def hub_config_names_cached(repo_id: str) -> list[str]:
    try:
        from huggingface_hub import get_dataset_config_names
    except ImportError:
        from datasets import get_dataset_config_names

    return sorted(get_dataset_config_names(repo_id))


@st.cache_data(show_spinner=False)
def hub_layers_cached(
    repo_id: str,
    model_name: str,
    mask_strategy_value: str,
    variant: str,
    persona_id: str,
) -> list[int]:
    store = HFActivationStore(
        repo_id,
        model_name,
        mask_strategy=MaskStrategy(mask_strategy_value),
    )
    sample = store.load(variant, persona_id)
    return list(range(int(sample.shape[0])))


def parse_hub_config_name(config_name: str) -> tuple[str, MaskStrategy] | None:
    for strategy in MaskStrategy:
        suffix = f"__{strategy.value}"
        if config_name.endswith(suffix):
            model_key = config_name[: -len(suffix)]
            return model_key.replace("__", "/"), strategy
    return None


def hub_models_by_mask_strategy(repo_id: str) -> dict[MaskStrategy, list[str]]:
    models_by_strategy: dict[MaskStrategy, set[str]] = {
        strategy: set() for strategy in MaskStrategy
    }
    for config_name in hub_config_names_cached(repo_id):
        parsed = parse_hub_config_name(config_name)
        if parsed is None:
            continue
        model_name, strategy = parsed
        models_by_strategy[strategy].add(model_name)
    return {
        strategy: sorted(models)
        for strategy, models in models_by_strategy.items()
        if models
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


def local_model_matches(left: str, right: str) -> bool:
    return model_dir_name(left) == model_dir_name(right)
