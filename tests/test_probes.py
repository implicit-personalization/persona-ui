"""Regression tests for utils.probes.

Covers the probe-artifact filename parser (both naming conventions) and the
correctness fix:

* ``_normalize_batch`` applies PCA independently of the scaler (previously the
  PCA branch was unreachable when no scaler was present).
"""

import pytest
import torch
from persona_vectors.probes import ProbeArtifact

from utils.probe_files import parse_probe_filename
from utils.probes import (
    LoadedProbe,
    _LinearProbe,
    _loaded_probe_from_artifact,
    _normalize_labels,
)

# --------------------------------------------------------------------------- #
# parse_probe_filename
# --------------------------------------------------------------------------- #


def test_parse_cognitive_map_filename():
    meta = parse_probe_filename(
        "cognitive_map_probe_layer12_lr_pre_reasoning_all_general.pt"
    )
    assert meta.layer == 12
    assert meta.model_type == "lr"
    assert meta.location == "pre_reasoning"
    assert meta.scope == "general"


def test_parse_persona_probe_dir_without_pca():
    meta = parse_probe_filename(
        "google__gemma-3-27b-it/answer_mean/biography/sex/"
        "logistic_regression_layer20/probe.json"
    )
    assert meta.layer == 20
    assert meta.model_type == "logistic_regression"
    assert meta.scope is None
    assert meta.attribute_name == "sex"
    assert meta.model_name == "google/gemma-3-27b-it"


def test_parse_persona_probe_dir_with_pca():
    meta = parse_probe_filename(
        "google__gemma-3-27b-it/answer_mean/biography/sex/"
        "logistic_regression_pca10_layer46/weights.safetensors"
    )
    assert meta.layer == 46
    assert meta.model_type == "logistic_regression"
    assert meta.scope == "pca10"
    assert meta.attribute_name == "sex"


def test_parse_unknown_filename_falls_back():
    meta = parse_probe_filename("something_else.bin")
    assert meta.layer is None
    assert meta.model_type == "unknown"


# --------------------------------------------------------------------------- #
# _normalize_labels
# --------------------------------------------------------------------------- #


def test_normalize_labels_list_pads_and_truncates():
    assert _normalize_labels(["a", "b"], 3) == ["a", "b", None]
    assert _normalize_labels(["a", "b", "c"], 2) == ["a", "b"]


def test_normalize_labels_dict_indexes_by_key():
    assert _normalize_labels({"1": "pos", "0": "neg"}, 2) == ["neg", "pos"]


def test_normalize_labels_none():
    assert _normalize_labels(None, 2) == [None, None]


# --------------------------------------------------------------------------- #
# _normalize_batch — scaler and PCA are applied independently
# --------------------------------------------------------------------------- #


def _probe(model_input_dim: int, **kwargs) -> LoadedProbe:
    return LoadedProbe(
        model=_LinearProbe(input_dim=model_input_dim, num_classes=1),
        input_dim=model_input_dim,
        labels=[None],
        model_type="linear",
        layer=0,
        location=None,
        **kwargs,
    )


def test_normalize_batch_noop_without_scaler_or_pca():
    probe = _probe(3)
    batch = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(probe._normalize_batch(batch), batch)


def test_normalize_batch_scaler_only():
    probe = _probe(
        3,
        scaler_mean=torch.ones(3),
        scaler_std=torch.full((3,), 2.0),
    )
    batch = torch.tensor([[3.0, 5.0, 7.0]])
    out = probe._normalize_batch(batch)
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0, 3.0]]))


def test_normalize_batch_pca_only_applies_pca():
    """Regression: PCA must apply even when no scaler is present."""
    probe = _probe(
        2,
        pca_mean=torch.ones(3),
        pca_components=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    batch = torch.tensor([[2.0, 4.0, 9.0]])
    out = probe._normalize_batch(batch)
    # (batch - pca_mean) @ components.T -> rows [1, 3] selected by components
    torch.testing.assert_close(out, torch.tensor([[1.0, 3.0]]))


def test_normalize_batch_scaler_then_pca():
    probe = _probe(
        3,
        scaler_mean=torch.zeros(3),
        scaler_std=torch.ones(3),
        pca_mean=torch.zeros(3),
        pca_components=torch.eye(3),
    )
    batch = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(probe._normalize_batch(batch), batch)


def test_normalize_batch_scaler_shape_mismatch_raises():
    probe = _probe(
        3,
        scaler_mean=torch.ones(5),
        scaler_std=torch.ones(5),
    )
    with pytest.raises(ValueError, match="scaler shape"):
        probe._normalize_batch(torch.zeros(1, 3))


def test_normalize_batch_pca_shape_mismatch_raises():
    probe = _probe(
        2,
        pca_mean=torch.ones(5),
        pca_components=torch.zeros(2, 5),
    )
    with pytest.raises(ValueError, match="PCA mean shape"):
        probe._normalize_batch(torch.zeros(1, 3))


# --------------------------------------------------------------------------- #
# canonical persona-vectors artifacts
# --------------------------------------------------------------------------- #


def test_loaded_probe_from_canonical_artifact():
    artifact = ProbeArtifact(
        metadata={
            "schema_version": 2,
            "input_dim": 2,
            "artifact_feature_dim": 2,
            "class_names": ["neg", "pos"],
            "task": "binary",
            "probe_kind": "logistic_regression",
            "layer": 3,
        },
        tensors={
            "weight": torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
            "bias": torch.zeros(2),
        },
    )
    probe = _loaded_probe_from_artifact(
        filename="m/answer_mean/templated/sex/logistic_regression_layer3/probe.json",
        artifact=artifact,
    )
    assert probe.labels == ["neg", "pos"]
    assert probe.layer == 3
    _, _, predicted = probe.run_batch(torch.tensor([[1.0, 0.0]]))
    assert probe.labels[int(predicted[0])] == "pos"
