from __future__ import annotations

from types import SimpleNamespace

import torch
from persona_vectors.analysis import LayeredSamples
from persona_vectors.probes import AttributeLabels

from tabs import probe_sweep


def test_cached_sweep_keeps_per_attribute_samples_and_full_plus_pca(monkeypatch):
    samples = LayeredSamples(
        vectors=torch.zeros((3, 2, 4)),
        labels=["p0", "p1", "p2"],
        hover_text=["p0", "p1", "p2"],
    )
    sweep_calls: list[tuple[str, int | None]] = []

    monkeypatch.setattr(
        probe_sweep,
        "load_persona_vectors_cached",
        lambda *args: samples,
    )
    monkeypatch.setattr(
        probe_sweep,
        "synth_persona_dataset_cached",
        lambda: SimpleNamespace(),
    )

    def labels_for(_dataset, attribute, _persona_ids, *, task):
        return AttributeLabels(
            attribute_name=attribute,
            task=task,
            y=torch.tensor([0, 1, 0]).numpy(),
            labels=["a", "b", "a"],
            class_names=["a", "b"],
        )

    monkeypatch.setattr(probe_sweep, "attribute_probe_labels", labels_for)

    def filtered(input_samples, labels, *, min_count):
        assert min_count == 2
        return input_samples, labels

    monkeypatch.setattr(
        probe_sweep,
        "filter_attribute_samples_min_count",
        filtered,
    )

    def sweep(input_samples, labels, *, layers, probe_kinds, n_pca_components, seed):
        assert input_samples is samples
        assert layers == [0, 1]
        assert probe_kinds == ["logistic_regression"]
        assert seed == 0
        sweep_calls.append((labels.attribute_name, n_pca_components))
        return [
            {
                "attribute": labels.attribute_name,
                "layer": 0,
                "probe_kind": probe_kinds[0],
                "balanced_accuracy": 0.5,
            }
        ]

    monkeypatch.setattr(probe_sweep, "sweep_attribute", sweep)

    inputs = probe_sweep.SweepInputs(
        source="src",
        location="loc",
        model_name="model",
        mask_value="answer_mean",
        variant="templated",
        persona_ids=("p0", "p1", "p2"),
        attributes=("sex", "gender"),
        task="binary",
        probe_kinds=("logistic_regression",),
        n_pca_components=2,
        layers=(0, 1),
        min_class_count=2,
        seed=0,
    )

    rows_by_label, per_attr = probe_sweep.cached_sweep.__wrapped__(inputs)

    assert list(rows_by_label) == ["full", "pca2"]
    assert [row["attribute"] for row in rows_by_label["full"]] == ["sex", "gender"]
    assert set(per_attr) == {"sex", "gender"}
    assert sweep_calls == [
        ("sex", None),
        ("gender", None),
        ("sex", 2),
        ("gender", 2),
    ]
