from __future__ import annotations

from dataclasses import dataclass

import streamlit as st
from persona_vectors.analysis import LayeredSamples
from persona_vectors.probes import (
    AttributeLabels,
    attribute_probe_labels,
    filter_attribute_samples_min_count,
    sweep_attribute,
)

from utils.analysis_metadata import synth_persona_dataset_cached
from utils.analysis_sources import load_persona_vectors_cached
from utils.helpers import env_int

_SWEEP_CACHE_ENTRIES = env_int("PERSONA_UI_PROBE_SWEEP_CACHE_ENTRIES", 4)


@dataclass(frozen=True)
class SweepInputs:
    source: str
    location: str
    model_name: str
    mask_value: str
    variant: str
    persona_ids: tuple[str, ...]
    attributes: tuple[str, ...]
    task: str
    probe_kinds: tuple[str, ...]
    n_pca_components: int | None
    layers: tuple[int, ...]
    min_class_count: int
    seed: int


@st.cache_resource(show_spinner=False, max_entries=_SWEEP_CACHE_ENTRIES)
def cached_sweep(
    inputs: SweepInputs,
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, tuple[AttributeLabels, LayeredSamples]],
]:
    samples = load_persona_vectors_cached(
        inputs.source,
        inputs.location,
        inputs.model_name,
        inputs.mask_value,
        inputs.variant,
        inputs.persona_ids,
    )
    dataset = synth_persona_dataset_cached()
    per_attr: dict[str, tuple[AttributeLabels, LayeredSamples]] = {}

    def labels_and_samples(attribute: str) -> tuple[AttributeLabels, LayeredSamples]:
        if attribute not in per_attr:
            labels = attribute_probe_labels(
                dataset,
                attribute,
                list(inputs.persona_ids),
                task=inputs.task,  # type: ignore[arg-type]
            )
            probe_samples, labels = filter_attribute_samples_min_count(
                samples,
                labels,
                min_count=inputs.min_class_count,
            )
            per_attr[attribute] = (labels, probe_samples)
        return per_attr[attribute]

    def sweep_one(attribute: str, n_pca: int | None) -> list[dict[str, object]]:
        labels, probe_samples = labels_and_samples(attribute)
        return sweep_attribute(
            probe_samples,
            labels,
            layers=list(inputs.layers),
            probe_kinds=list(inputs.probe_kinds),  # type: ignore[arg-type]
            n_pca_components=n_pca,
            seed=inputs.seed,
        )

    def sweep_all(n_pca: int | None) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for attribute in inputs.attributes:
            rows.extend(sweep_one(attribute, n_pca))
        return rows

    rows_by_label = {"full": sweep_all(None)}
    if inputs.n_pca_components is not None:
        rows_by_label[f"pca{inputs.n_pca_components}"] = sweep_all(
            inputs.n_pca_components
        )
    return rows_by_label, per_attr
