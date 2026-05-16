import atexit
import hashlib
import shutil
import threading
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import streamlit as st
from persona_data.nemotron_personas import (
    NemotronPersonasFranceDataset,
    NemotronPersonasUSADataset,
)
from persona_data.synth_persona import PersonaDataset as LocalPersonaDataset
from persona_data.synth_persona import SynthPersonaDataset

from .helpers import DatasetSource


@st.cache_resource(show_spinner=False)
def _cached_dataset(cls: type) -> Any:
    """Instantiate and cache a HuggingFace dataset class once per session."""

    return cls()


_qa_warm_lock = threading.Lock()


def warm_qa_in_background(dataset: Any) -> None:
    """Trigger the dataset's lazy QA parse on a daemon thread, once.

    QA loading is deferred in persona-data (large, unused outside Extract).
    Kicking it off when the Extract tab opens means the parse overlaps with
    the user picking personas/options instead of blocking the first run.
    Idempotent across Streamlit reruns: guarded per cached dataset instance.
    """

    warm = getattr(dataset, "_load_qa", None)
    if warm is None:
        return  # persona-only dataset (e.g. Nemotron) has no QA
    with _qa_warm_lock:
        if getattr(dataset, "_qa_warm_started", False):
            return
        dataset._qa_warm_started = True
    threading.Thread(
        target=warm, name="persona-ui-warm-qa", daemon=True
    ).start()


@st.cache_resource(show_spinner=False)
def _cached_local_dataset(personas_path: str, qa_path: str) -> LocalPersonaDataset:
    """Instantiate and cache a local upload dataset for stable temp paths."""

    return LocalPersonaDataset(personas_path=Path(personas_path), qa_path=Path(qa_path))


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
    data = uploaded_file.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    temp_path = _upload_cache_dir() / f"{stem}_{digest[:16]}{suffix}"
    if temp_path.exists():
        return temp_path
    temp_path.write_bytes(data)
    return temp_path


def load_persona_list(
    dataset_source: str,
    personas_file: Any = None,
    qa_file: Any = None,
) -> tuple[list, str]:
    """Like ``load_dataset`` but returns ``(personas_list, status)``.

    The list is memoized on the cached dataset instance so repeated reruns
    don't pay for re-iteration.
    """

    dataset, status = load_dataset(dataset_source, personas_file, qa_file)
    return load_persona_list_from_dataset(dataset), status


def load_persona_list_from_dataset(dataset: Any) -> list:
    """Materialize and cache personas from an already-loaded dataset."""

    cached = getattr(dataset, "_persona_list_cache", None)
    if cached is None:
        cached = list(dataset)
        try:
            dataset._persona_list_cache = cached
        except (AttributeError, TypeError):
            pass
    return cached


def load_dataset(
    dataset_source: str,
    personas_file: Any = None,
    qa_file: Any = None,
) -> tuple[
    SynthPersonaDataset
    | NemotronPersonasFranceDataset
    | NemotronPersonasUSADataset
    | LocalPersonaDataset,
    str,
]:
    """Load the selected dataset source for the UI."""

    if dataset_source == DatasetSource.SYNTH_PERSONA.value:
        return _cached_dataset(SynthPersonaDataset), "SynthPersona"

    if dataset_source == DatasetSource.NEMOTRON_FRANCE.value:
        return _cached_dataset(NemotronPersonasFranceDataset), "Nemotron France"

    if dataset_source == DatasetSource.NEMOTRON_USA.value:
        return _cached_dataset(NemotronPersonasUSADataset), "Nemotron USA"

    if personas_file is None or qa_file is None:
        raise ValueError("Upload both personas.jsonl and qa.jsonl files")

    personas_path = _uploaded_file_to_temp_path(personas_file, stem="personas")
    qa_path = _uploaded_file_to_temp_path(qa_file, stem="qa")
    return _cached_local_dataset(str(personas_path), str(qa_path)), "Local upload"
