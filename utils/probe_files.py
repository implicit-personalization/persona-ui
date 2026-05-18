from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

PROBE_FILENAME_RE = re.compile(
    r"^cognitive_map_probe_layer(?P<layer>\d+)_(?P<model_type>[a-z0-9]+)_"
    r"(?P<location>pre_reasoning|post_reasoning)_all_(?P<scope>general|size\d+)\.pt$"
)

PERSONA_PROBE_DIR_RE = re.compile(
    r"^(?P<probe_kind>[a-z_]+?)(?:_pca(?P<pca>\d+))?_layer(?P<layer>\d+)$"
)

DEFAULT_PROBE_REPO = "project-telos/cognitive_map_probes"
DEFAULT_LOCAL_PROBE_DIR = os.environ.get("PERSONA_PROBES_DIR", "artifacts/probes")


@dataclass(frozen=True)
class ProbeFileMetadata:
    filename: str
    layer: int | None
    model_type: str
    location: str | None
    scope: str | None
    label: str
    model_name: str | None = None
    attribute_name: str | None = None


def model_probe_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def parse_probe_filename(filename: str) -> ProbeFileMetadata:
    path = Path(filename)
    match = PROBE_FILENAME_RE.match(path.name)
    if match:
        layer = int(match.group("layer"))
        model_type = match.group("model_type")
        location = match.group("location")
        scope = match.group("scope")
        scope_label = scope.replace("size", "size ")
        return ProbeFileMetadata(
            filename=filename,
            layer=layer,
            model_type=model_type,
            location=location,
            scope=scope,
            label=(
                f"Layer {layer} - {model_type.upper()} - "
                f"{location.replace('_', ' ')} - {scope_label}"
            ),
        )

    parent_match = PERSONA_PROBE_DIR_RE.match(path.parent.name)
    if parent_match and path.name in {"probe.json", "weights.safetensors"}:
        layer = int(parent_match.group("layer"))
        probe_kind = parent_match.group("probe_kind")
        pca = parent_match.group("pca")
        scope = f"pca{pca}" if pca else None
        attribute = path.parent.parent.name or "attribute"
        model_name = path.parts[0].replace("__", "/") if len(path.parts) >= 5 else None
        label = f"{attribute} - layer {layer} - {probe_kind}"
        if pca:
            label += f" (pca{pca})"
        return ProbeFileMetadata(
            filename=filename,
            layer=layer,
            model_type=probe_kind,
            location=None,
            scope=scope,
            label=label,
            model_name=model_name,
            attribute_name=attribute,
        )

    return ProbeFileMetadata(
        filename=filename,
        layer=None,
        model_type="unknown",
        location=None,
        scope=None,
        label=path.stem.replace("_", " "),
    )


@st.cache_data(show_spinner=False, ttl=300)
def list_probe_files(repo_id: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return _dedupe_probe_entries(list_repo_files(repo_id, repo_type="model"))


@st.cache_data(show_spinner=False, ttl=30)
def list_local_probe_files(root_dir: str) -> list[str]:
    root = Path(root_dir).expanduser()
    if not root.is_dir():
        return []
    files = _dedupe_probe_entries(
        [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file()
            and path.name in {"probe.pt", "probe.json", "weights.safetensors"}
        ]
    )
    return sorted(files, key=_probe_sort_key)


@st.cache_data(show_spinner=False, ttl=300)
def download_probe_file(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, repo_type="model")


@st.cache_data(show_spinner=False, ttl=300)
def download_probe_json_and_weights(repo_id: str, filename: str) -> tuple[str, str]:
    from huggingface_hub import hf_hub_download

    metadata_path = hf_hub_download(repo_id, filename, repo_type="model")
    weights_name = str(Path(filename).with_name("weights.safetensors"))
    weights_path = hf_hub_download(repo_id, weights_name, repo_type="model")
    return metadata_path, weights_path


def _probe_sort_key(filename: str) -> tuple[str, str, int, str]:
    metadata = parse_probe_filename(filename)
    return (
        metadata.model_name or "",
        metadata.attribute_name or "",
        metadata.layer if metadata.layer is not None else 10**9,
        filename,
    )


def _dedupe_probe_entries(files: list[str]) -> list[str]:
    by_dir: dict[str, set[str]] = {}
    standalone: list[str] = []
    for filename in files:
        path = Path(filename)
        if path.name in {"probe.pt", "probe.json", "weights.safetensors"}:
            by_dir.setdefault(str(path.parent), set()).add(path.name)
        elif filename.endswith(".pt"):
            standalone.append(filename)

    entries = list(standalone)
    for directory, names in by_dir.items():
        selected = (
            "probe.json"
            if "probe.json" in names
            else "probe.pt"
            if "probe.pt" in names
            else "weights.safetensors"
        )
        entries.append(str(Path(directory) / selected))
    return sorted(entries, key=_probe_sort_key)
