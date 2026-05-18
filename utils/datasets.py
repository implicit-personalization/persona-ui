import atexit
import hashlib
import shutil
import threading
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import streamlit as st
from huggingface_hub import hf_hub_download, list_repo_files, try_to_load_from_cache
from persona_data.nemotron_personas import (
    NemotronPersonasFranceDataset,
    NemotronPersonasUSADataset,
)
from persona_data.synth_persona import PersonaDataset as LocalPersonaDataset
from persona_data.synth_persona import SynthPersonaDataset

from .helpers import DatasetSource

_SYNTH_PERSONA_REPO = "implicit-personalization/synth-persona"
_SYNTH_PERSONA_STARTUP_FILES = (
    "implicit_shared_mc_bank.json",
    "dataset_personas.jsonl",
)
_SYNTH_PERSONA_QA_FILE = "dataset_qa.jsonl"
_NEMOTRON_REPOS = {
    DatasetSource.NEMOTRON_FRANCE.value: "nvidia/Nemotron-Personas-France",
    DatasetSource.NEMOTRON_USA.value: "nvidia/Nemotron-Personas-USA",
}


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

    warm = getattr(dataset, "prefetch_qa", None)
    if warm is None:
        return  # persona-only dataset (e.g. Nemotron) has no QA
    if isinstance(dataset, SynthPersonaDataset):
        # Extract will need QA soon. Make the one-time large transfer explicit,
        # then leave the CPU-heavy parse on the existing background thread.
        _download_missing_startup_files_if_needed(
            _SYNTH_PERSONA_REPO,
            (_SYNTH_PERSONA_QA_FILE,),
            "SynthPersona QA",
        )
    with _qa_warm_lock:
        if getattr(dataset, "_qa_warm_started", False):
            return
        dataset._qa_warm_started = True
    threading.Thread(target=warm, name="persona-ui-warm-qa", daemon=True).start()


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
        _download_missing_startup_files_if_needed(
            _SYNTH_PERSONA_REPO,
            _SYNTH_PERSONA_STARTUP_FILES,
            "SynthPersona",
        )
        return _cached_dataset(SynthPersonaDataset), "SynthPersona"

    if dataset_source == DatasetSource.NEMOTRON_FRANCE.value:
        _prepare_nemotron_startup_download(dataset_source, "Nemotron France")
        return _cached_dataset(NemotronPersonasFranceDataset), "Nemotron France"

    if dataset_source == DatasetSource.NEMOTRON_USA.value:
        _prepare_nemotron_startup_download(dataset_source, "Nemotron USA")
        return _cached_dataset(NemotronPersonasUSADataset), "Nemotron USA"

    if personas_file is None or qa_file is None:
        raise ValueError("Upload both personas.jsonl and qa.jsonl files")

    personas_path = _uploaded_file_to_temp_path(personas_file, stem="personas")
    qa_path = _uploaded_file_to_temp_path(qa_file, stem="qa")
    return _cached_local_dataset(str(personas_path), str(qa_path)), "Local upload"


def _is_cached(repo_id: str, filename: str) -> bool:
    """Return whether a Hub dataset file already exists in the local HF cache."""

    cached = try_to_load_from_cache(repo_id, filename, repo_type="dataset")
    return isinstance(cached, str)


def _download_missing_startup_files_if_needed(
    repo_id: str,
    filenames: tuple[str, ...],
    label: str,
) -> None:
    """Make first-time Hub downloads visible before dataset construction blocks.

    Hugging Face handles byte-level transfer internally. We expose file-level
    progress here, which is the useful unit this UI can know in advance.
    """

    missing = tuple(
        filename for filename in filenames if not _is_cached(repo_id, filename)
    )
    if not missing:
        return

    notice = st.empty()
    notice.warning(
        f"First-time setup for {label}: downloading dataset files from Hugging Face. "
        "Later loads should use the local cache."
    )
    progress = st.progress(0.0, text=f"Preparing {label} download…")
    total = len(missing)
    for index, filename in enumerate(missing, start=1):
        progress.progress(
            (index - 1) / total,
            text=f"Downloading {filename} ({index}/{total})",
        )
        hf_hub_download(repo_id, filename, repo_type="dataset")
        progress.progress(
            index / total,
            text=f"Downloaded {filename} ({index}/{total})",
        )
    notice.empty()


def _prepare_nemotron_startup_download(dataset_source: str, label: str) -> None:
    """Prefetch the first parquet shard used by the default Nemotron sample."""

    repo_id = _NEMOTRON_REPOS[dataset_source]
    parquet_files = tuple(
        sorted(
            filename
            for filename in list_repo_files(repo_id, repo_type="dataset")
            if filename.startswith("data/train-") and filename.endswith(".parquet")
        )
    )
    if parquet_files:
        _download_missing_startup_files_if_needed(repo_id, (parquet_files[0],), label)
