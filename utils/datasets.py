import atexit
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import streamlit as st
from persona_data.synth_persona import SynthPersonaDataset

from .helpers import DATASET_SOURCES
from .local_dataset import LocalPersonaDataset


@st.cache_resource(show_spinner=False)
def cached_hf_dataset() -> SynthPersonaDataset:
    """Load the default SynthPersona HuggingFace dataset once."""

    return SynthPersonaDataset()


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
    data = uploaded_file.getvalue()
    if temp_path.exists() and temp_path.stat().st_size == len(data):
        return temp_path
    temp_path.write_bytes(data)
    return temp_path


def load_dataset(
    dataset_source: str,
) -> tuple[SynthPersonaDataset | LocalPersonaDataset, str]:
    """Load the selected dataset source for the UI."""

    if dataset_source == DATASET_SOURCES[0]:
        return cached_hf_dataset(), "SynthPersona"

    personas_file = st.session_state.get("extract__personas_file")
    qa_file = st.session_state.get("extract__qa_file")
    if personas_file is None or qa_file is None:
        raise ValueError("Upload both personas.jsonl and qa.jsonl files")

    personas_path = _uploaded_file_to_temp_path(personas_file, stem="personas")
    qa_path = _uploaded_file_to_temp_path(qa_file, stem="qa")
    return (
        LocalPersonaDataset(personas_path=personas_path, qa_path=qa_path),
        "Local upload",
    )
