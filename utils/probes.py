from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from persona_vectors.probes import ProbeArtifact, load_probe_artifact

from utils.helpers import env_int
from utils.probe_files import (
    download_probe_file,
    download_probe_json_and_weights,
    parse_probe_filename,
)

_PROBE_CACHE_ENTRIES = env_int("PERSONA_UI_PROBE_CACHE_ENTRIES", 8)


@dataclass(frozen=True)
class ProbeRunResult:
    input_dim: int
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_index: int
    predicted_label: str | None


class _LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _MLPProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("MLP probe requires at least one hidden dimension")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class LoadedProbe:
    model: nn.Module
    input_dim: int
    labels: list[str | None]
    model_type: str
    layer: int | None
    location: str | None
    model_name: str | None = None
    attribute_name: str | None = None
    feature_space: str | None = None
    task: str | None = None
    probe_kind: str | None = None
    scaler_mean: torch.Tensor | None = None
    scaler_std: torch.Tensor | None = None
    pca_mean: torch.Tensor | None = None
    pca_components: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.model.eval()

    @property
    def is_regression(self) -> bool:
        """True when the probe outputs a continuous value rather than a class."""
        if self.task is not None:
            return self.task in {"numeric", "ordinal"}
        if self.probe_kind is not None:
            return self.probe_kind == "ridge_regression"
        return False

    def predict_batch(self, activations: torch.Tensor) -> torch.Tensor:
        """Return raw linear-output values for each token — no sigmoid/softmax."""
        if activations.ndim != 2:
            raise ValueError(
                f"predict_batch expects [N, hidden], got {tuple(activations.shape)}"
            )
        if activations.shape[1] != self.input_dim:
            raise ValueError(
                f"Probe expects input dim {self.input_dim}, got {activations.shape[1]}"
            )
        batch = activations.detach().to(dtype=torch.float32, device="cpu")
        normalized = self._normalize_batch(batch)
        with torch.no_grad():
            outputs = self.model(normalized).detach().cpu()
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(-1)
        return outputs

    def run(self, vector: torch.Tensor) -> ProbeRunResult:
        if vector.ndim != 1:
            raise ValueError(
                f"Probe expects a 1D activation vector, got shape {tuple(vector.shape)}"
            )
        if vector.shape[0] != self.input_dim:
            raise ValueError(
                f"Probe expects input dim {self.input_dim}, got {vector.shape[0]}"
            )

        batch = vector.detach().to(dtype=torch.float32, device="cpu").unsqueeze(0)
        logits_batch, probs_batch = self._forward_batch(batch)
        logits = logits_batch.squeeze(0)
        probs = probs_batch.squeeze(0)
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        if probs.ndim == 0:
            probs = probs.unsqueeze(0)

        predicted_index = (
            int(probs.item() >= 0.5)
            if probs.numel() == 1
            else int(torch.argmax(probs).item())
        )
        predicted_label = (
            self.labels[predicted_index]
            if 0 <= predicted_index < len(self.labels)
            else None
        )
        return ProbeRunResult(
            input_dim=int(vector.shape[0]),
            logits=logits,
            probabilities=probs,
            predicted_index=predicted_index,
            predicted_label=predicted_label,
        )

    def run_batch(
        self, activations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the probe over a batch of activations.

        Returns ``(logits[N, C], probs[N, C], predicted_index[N])``. For
        single-output probes ``C == 1`` and ``probs`` holds sigmoid scores.
        """
        if activations.ndim != 2:
            raise ValueError(
                f"run_batch expects [N, hidden], got {tuple(activations.shape)}"
            )
        if activations.shape[1] != self.input_dim:
            raise ValueError(
                f"Probe expects input dim {self.input_dim}, got {activations.shape[1]}"
            )
        batch = activations.detach().to(dtype=torch.float32, device="cpu")
        logits, probs = self._forward_batch(batch)
        if probs.shape[-1] == 1:
            predicted = (probs.squeeze(-1) >= 0.5).long()
        else:
            predicted = torch.argmax(probs, dim=-1)
        return logits, probs, predicted

    def _forward_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self._normalize_batch(batch)
        with torch.no_grad():
            logits = self.model(normalized).detach().cpu()
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=-1)
        return logits, probs

    def _normalize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if self.scaler_mean is not None and self.scaler_std is not None:
            mean = self.scaler_mean.to(dtype=torch.float32)
            std = self.scaler_std.to(dtype=torch.float32)
            if mean.ndim != 1 or std.ndim != 1 or mean.shape[0] != batch.shape[1]:
                raise ValueError(
                    "Probe scaler shape does not match activation hidden size: "
                    f"mean={tuple(mean.shape)} std={tuple(std.shape)} "
                    f"batch={tuple(batch.shape)}"
                )
            safe_std = torch.where(std == 0, torch.ones_like(std), std)
            batch = (batch - mean) / safe_std
        if self.pca_mean is not None and self.pca_components is not None:
            pca_mean = self.pca_mean.to(dtype=torch.float32)
            components = self.pca_components.to(dtype=torch.float32)
            if pca_mean.ndim != 1 or pca_mean.shape[0] != batch.shape[1]:
                raise ValueError(
                    "Probe PCA mean shape does not match activation hidden size: "
                    f"mean={tuple(pca_mean.shape)} batch={tuple(batch.shape)}"
                )
            batch = (batch - pca_mean) @ components.T
        return batch


@st.cache_resource(show_spinner=False, max_entries=_PROBE_CACHE_ENTRIES)
def load_probe(repo_id: str, filename: str) -> LoadedProbe:
    if filename.endswith("probe.json"):
        metadata_path, weights_path = download_probe_json_and_weights(repo_id, filename)
        return _load_persona_probe_artifact(
            filename=filename,
            metadata_path=Path(metadata_path),
            weights_path=Path(weights_path),
        )
    path = download_probe_file(repo_id, filename)
    return _load_probe_payload(
        filename=filename,
        payload=_torch_load(path),
    )


@st.cache_resource(show_spinner=False, max_entries=_PROBE_CACHE_ENTRIES)
def load_local_probe(root_dir: str, filename: str) -> LoadedProbe:
    root = Path(root_dir).expanduser()
    path = (root / filename).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("Probe path must stay inside the selected local directory.")
    if path.name == "probe.json":
        return _load_persona_probe_artifact(
            filename=filename,
            metadata_path=path,
            weights_path=path.with_name("weights.safetensors"),
        )
    if path.name == "weights.safetensors":
        return _load_persona_probe_artifact(
            filename=filename,
            metadata_path=path.with_name("probe.json"),
            weights_path=path,
        )
    return _load_probe_payload(
        filename=filename,
        payload=_torch_load(path),
    )


@st.cache_resource(show_spinner=False, max_entries=_PROBE_CACHE_ENTRIES)
def load_probe_from_bytes(filename: str, data: bytes) -> LoadedProbe:
    return _load_probe_payload(
        filename=filename,
        payload=_torch_load(io.BytesIO(data)),
    )


def _load_probe_payload(
    *,
    filename: str,
    payload: object,
) -> LoadedProbe:
    if not isinstance(payload, dict):
        raise TypeError(f"Probe payload must be a dict, got {type(payload)!r}")

    metadata = parse_probe_filename(filename)
    state_dict = _get_state_dict(payload)
    input_dim = _coerce_probe_dim(payload.get("input_dim"), state_dict, dim="input")
    model_input_dim = _coerce_probe_dim(
        payload.get("artifact_feature_dim") or input_dim,
        state_dict,
        dim="input",
    )
    num_classes = _coerce_probe_dim(
        payload.get("num_classes"), state_dict, dim="classes"
    )
    model = _build_probe_module(
        payload,
        state_dict=state_dict,
        input_dim=model_input_dim,
        num_classes=num_classes,
    )
    labels = _normalize_labels(
        payload.get("idx_to_label") or payload.get("class_names"),
        num_classes,
    )

    raw_layer = payload.get("layer")
    try:
        layer = int(raw_layer) if raw_layer is not None else metadata.layer
    except (TypeError, ValueError):
        layer = metadata.layer
    raw_location = payload.get("location")
    location = (
        raw_location
        if isinstance(raw_location, str) and raw_location
        else metadata.location
    )

    return LoadedProbe(
        model=model,
        input_dim=input_dim,
        labels=labels,
        model_type=str(payload.get("model_type") or metadata.model_type),
        layer=layer,
        location=location,
        model_name=_optional_str(payload.get("model_name")) or metadata.model_name,
        attribute_name=(
            _optional_str(payload.get("attribute_name")) or metadata.attribute_name
        ),
        feature_space=(
            (
                f"pca{payload['n_pca_components']}"
                if payload.get("n_pca_components")
                else None
            )
            or _optional_str(payload.get("feature_space"))
            or metadata.scope
        ),
        task=_optional_str(payload.get("task")),
        probe_kind=_optional_str(payload.get("probe_kind")),
        scaler_mean=_as_cpu_tensor(payload.get("scaler_mean")),
        scaler_std=_as_cpu_tensor(
            _first_present(payload, "scaler_std", "scaler_scale")
        ),
        pca_mean=_as_cpu_tensor(payload.get("pca_mean")),
        pca_components=_as_cpu_tensor(payload.get("pca_components")),
    )


def _torch_load(file_or_buffer: object) -> object:
    return torch.load(file_or_buffer, map_location="cpu", weights_only=True)


def _load_persona_probe_artifact(
    *,
    filename: str,
    metadata_path: Path,
    weights_path: Path,
) -> LoadedProbe:
    if metadata_path.parent != weights_path.parent:
        raise ValueError("Canonical probe files must share one artifact directory.")
    artifact = load_probe_artifact(metadata_path)
    return _loaded_probe_from_artifact(filename=filename, artifact=artifact)


def _loaded_probe_from_artifact(
    *,
    filename: str,
    artifact: ProbeArtifact,
) -> LoadedProbe:
    metadata = artifact.metadata
    tensors = artifact.tensors
    payload = {
        **metadata,
        "model_type": "linear",
        "model_state_dict": {
            "linear.weight": tensors["weight"],
            "linear.bias": tensors["bias"],
        },
        "num_classes": int(tensors["weight"].shape[0]),
        "idx_to_label": metadata.get("class_names"),
        "scaler_mean": tensors.get("scaler_mean"),
        "scaler_std": tensors.get("scaler_scale"),
        "pca_mean": tensors.get("pca_mean"),
        "pca_components": tensors.get("pca_components"),
    }
    return _load_probe_payload(filename=filename, payload=payload)


def _build_probe_module(
    payload: dict[str, Any],
    *,
    state_dict: dict[str, torch.Tensor],
    input_dim: int,
    num_classes: int,
) -> nn.Module:
    model_type = str(payload.get("model_type") or "").lower()
    if model_type in {"lr", "linear", "logreg", "logistic_regression"}:
        module = _LinearProbe(input_dim=input_dim, num_classes=num_classes)
        state_dict = _normalize_linear_state_dict(state_dict)
    elif model_type == "mlp":
        hidden_dims = _coerce_hidden_dims(payload.get("hidden_dims"))
        dropout = float(payload.get("dropout") or 0.0)
        module = _MLPProbe(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )
        state_dict = _strip_known_prefixes(state_dict)
    else:
        if _looks_linear(state_dict):
            module = _LinearProbe(input_dim=input_dim, num_classes=num_classes)
            state_dict = _normalize_linear_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported probe model type: {model_type!r}")

    module.load_state_dict(state_dict, strict=True)
    return module


def _get_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "probe_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return {
                str(k): v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
    raise TypeError("Probe payload is missing model_state_dict")


def _coerce_probe_dim(
    value: object,
    state_dict: dict[str, torch.Tensor],
    *,
    dim: str,
) -> int:
    if value is not None:
        return int(value)

    weights = [
        tensor
        for key, tensor in state_dict.items()
        if key.endswith("weight")
        and isinstance(tensor, torch.Tensor)
        and tensor.ndim == 2
    ]
    if not weights:
        raise ValueError(f"Cannot infer probe {dim} dimension from state dict")

    tensor = weights[0] if dim == "input" else weights[-1]
    return int(tensor.shape[1] if dim == "input" else tensor.shape[0])


def _normalize_linear_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    stripped = _strip_known_prefixes(state_dict)
    if "linear.weight" in stripped:
        return stripped
    if "weight" in stripped:
        out = {"linear.weight": stripped["weight"]}
        if "bias" in stripped:
            out["linear.bias"] = stripped["bias"]
        return out
    return stripped


def _strip_known_prefixes(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        stripped = key
        for prefix in ("module.", "model.", "probe."):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
        out[stripped] = value
    return out


def _looks_linear(state_dict: dict[str, torch.Tensor]) -> bool:
    stripped = _strip_known_prefixes(state_dict)
    return "weight" in stripped or "linear.weight" in stripped


def _coerce_hidden_dims(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    raise TypeError(f"Unsupported hidden_dims value: {type(value)!r}")


def _as_cpu_tensor(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().cpu()


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _normalize_labels(raw_labels: Any, num_classes: int) -> list[str | None]:
    if isinstance(raw_labels, (list, tuple)):
        labels = [str(label) for label in raw_labels[:num_classes]]
        return labels + [None] * (num_classes - len(labels))
    if not isinstance(raw_labels, dict):
        return [None] * num_classes

    labels: list[str | None] = [None] * num_classes
    for raw_idx, raw_label in raw_labels.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < num_classes:
            labels[idx] = str(raw_label)
    return labels
