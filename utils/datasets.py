import atexit
import hashlib
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import streamlit as st
from persona_data.nemotron_personas import NemotronPersonasFranceDataset
from persona_data.synth_persona import PersonaDataset as LocalPersonaDataset
from persona_data.synth_persona import SynthPersonaDataset

from .helpers import DATASET_SOURCES


@st.cache_resource(show_spinner=False)
def cached_hf_dataset() -> SynthPersonaDataset:
    """Load the default SynthPersona HuggingFace dataset once."""

    return SynthPersonaDataset()


@st.cache_resource(show_spinner=False)
def cached_nemotron_fr_dataset() -> NemotronPersonasFranceDataset:
    """Load a sampled French persona-only dataset once."""

    return NemotronPersonasFranceDataset(sample_size=200, offset=0)


def _upload_cache_dir() -> Path:
    cache_dir = st.session_state.get("_upload_cache_dir")
    if cache_dir is None:
        cache_dir = mkdtemp(prefix="persona_vectors_uploads_")
        st.session_state["_upload_cache_dir"] = cache_dir
        # Register cleanup so the temp dir is removed when the server process exits.
        atexit.register(shutil.rmtree, cache_dir, ignore_errors=True)
    return Path(cache_dir)


def _uploaded_file_to_temp_path(uploaded_file: Any, stem: str) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jsonl"
    temp_path = _upload_cache_dir() / f"{stem}{suffix}"
    hash_path = temp_path.with_suffix(temp_path.suffix + ".sha256")
    data = uploaded_file.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if temp_path.exists() and hash_path.exists() and hash_path.read_text() == digest:
        return temp_path
    temp_path.write_bytes(data)
    hash_path.write_text(digest)
    return temp_path


def load_dataset(
    dataset_source: str,
    personas_file: Any = None,
    qa_file: Any = None,
) -> tuple[SynthPersonaDataset | LocalPersonaDataset | NemotronPersonasFranceDataset, str]:
    """Load the selected dataset source for the UI."""

    if dataset_source == DATASET_SOURCES[0]:
        return cached_hf_dataset(), "SynthPersona"
    if dataset_source == DATASET_SOURCES[1]:
        dataset = cached_nemotron_fr_dataset()
        return (
            dataset,
            (
                "Nemotron-Personas-France "
                f"(sampled {len(dataset)} personas from {dataset.split}, offset={dataset.offset})"
            ),
        )

    if personas_file is None or qa_file is None:
        raise ValueError("Upload both personas.jsonl and qa.jsonl files")

    personas_path = _uploaded_file_to_temp_path(personas_file, stem="personas")
    qa_path = _uploaded_file_to_temp_path(qa_file, stem="qa")
    return (
        LocalPersonaDataset(personas_path=personas_path, qa_path=qa_path),
        "Local upload",
    )
