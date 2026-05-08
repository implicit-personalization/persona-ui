from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

PROBE_FILENAME_RE = re.compile(
    r"^cognitive_map_probe_layer(?P<layer>\d+)_(?P<model_type>[a-z0-9]+)_"
    r"(?P<location>pre_reasoning|post_reasoning)_all_(?P<scope>general|size\d+)\.pt$"
)

DEFAULT_PROBE_REPO = "project-telos/cognitive_map_probes"


@dataclass(frozen=True)
class ProbeFileMetadata:
    filename: str
    layer: int | None
    model_type: str
    location: str | None
    scope: str | None
    label: str


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
    scaler_mean: torch.Tensor | None = None
    scaler_std: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.model.eval()

    def run(self, vector: torch.Tensor) -> ProbeRunResult:
        if vector.ndim != 1:
            raise ValueError(
                f"Probe expects a 1D activation vector, got shape {tuple(vector.shape)}"
            )
        if vector.shape[0] != self.input_dim:
            raise ValueError(
                f"Probe expects input dim {self.input_dim}, got {vector.shape[0]}"
            )

        normalized = self._normalize(
            vector.detach().to(dtype=torch.float32, device="cpu")
        )
        with torch.no_grad():
            logits = self.model(normalized.unsqueeze(0)).squeeze(0).detach().cpu()

        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        if logits.numel() == 1:
            probs = torch.sigmoid(logits).view(1)
            predicted_index = 0
        else:
            probs = F.softmax(logits, dim=-1)
            predicted_index = int(torch.argmax(probs).item())

        predicted_label = (
            self.labels[predicted_index]
            if 0 <= predicted_index < len(self.labels)
            else None
        )
        return ProbeRunResult(
            input_dim=int(normalized.shape[0]),
            logits=logits,
            probabilities=probs,
            predicted_index=predicted_index,
            predicted_label=predicted_label,
        )

    def _normalize(self, vector: torch.Tensor) -> torch.Tensor:
        if self.scaler_mean is None or self.scaler_std is None:
            return vector
        mean = self.scaler_mean.to(dtype=torch.float32)
        std = self.scaler_std.to(dtype=torch.float32)
        if mean.shape != vector.shape or std.shape != vector.shape:
            raise ValueError(
                "Probe scaler shape does not match activation vector shape: "
                f"mean={tuple(mean.shape)} std={tuple(std.shape)} "
                f"vector={tuple(vector.shape)}"
            )
        safe_std = torch.where(std == 0, torch.ones_like(std), std)
        return (vector - mean) / safe_std


def parse_probe_filename(filename: str) -> ProbeFileMetadata:
    match = PROBE_FILENAME_RE.match(Path(filename).name)
    if not match:
        stem = Path(filename).stem.replace("_", " ")
        return ProbeFileMetadata(
            filename=filename,
            layer=None,
            model_type="unknown",
            location=None,
            scope=None,
            label=stem,
        )

    layer = int(match.group("layer"))
    model_type = match.group("model_type")
    location = match.group("location")
    scope = match.group("scope")
    scope_label = scope.replace("size", "size ")
    label = (
        f"Layer {layer} - {model_type.upper()} - "
        f"{location.replace('_', ' ')} - {scope_label}"
    )
    return ProbeFileMetadata(
        filename=filename,
        layer=layer,
        model_type=model_type,
        location=location,
        scope=scope,
        label=label,
    )


@st.cache_data(show_spinner=False, ttl=300)
def list_probe_files(repo_id: str) -> list[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id, repo_type="model")
    return sorted(path for path in files if path.endswith(".pt"))


@st.cache_data(show_spinner=False, ttl=300)
def download_probe_file(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, repo_type="model")


@st.cache_resource(show_spinner=False)
def load_probe(repo_id: str, filename: str) -> LoadedProbe:
    path = download_probe_file(repo_id, filename)
    return _load_probe_payload(
        filename=filename,
        payload=_torch_load(path),
    )


@st.cache_resource(show_spinner=False)
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
    num_classes = _coerce_probe_dim(
        payload.get("num_classes"), state_dict, dim="classes"
    )
    model = _build_probe_module(
        payload,
        state_dict=state_dict,
        input_dim=input_dim,
        num_classes=num_classes,
    )
    labels = _normalize_labels(payload.get("idx_to_label"), num_classes)

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
        scaler_mean=_as_cpu_tensor(payload.get("scaler_mean")),
        scaler_std=_as_cpu_tensor(payload.get("scaler_std")),
    )


def _torch_load(file_or_buffer: object) -> object:
    return torch.load(file_or_buffer, map_location="cpu", weights_only=True)


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
