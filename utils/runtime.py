import json
import logging
import os
from collections.abc import Iterable

import streamlit as st

from utils.helpers import env_int, session_key

logger = logging.getLogger(__name__)
_LANGUAGE_MODEL_CLASSES = {"LanguageModel", "StandardizedTransformer"}
_EXPECTED_NDIF_STATES = {"RUNNING", "NOT DEPLOYED", "DEPLOYING", "DELETING"}
_MODEL_CACHE_ENTRIES = env_int("PERSONA_UI_MODEL_CACHE_ENTRIES", 1)
_SESSION_NDIF_API_KEY = session_key("sidebar", "ndif_api_key")


def _iter_deployments(raw: object) -> Iterable[dict]:
    if not isinstance(raw, dict):
        return ()
    deployments = raw.get("deployments", {})
    if not isinstance(deployments, dict):
        return ()
    return (value for value in deployments.values() if isinstance(value, dict))


def _is_visible_deployment(deployment: dict) -> bool:
    return deployment.get("deployment_level") in {"HOT", "WARM"} or (
        "schedule" in deployment
    )


def _repo_id_from_model_key(model_key: str) -> str:
    try:
        repo_id = json.loads(model_key.split(":", 1)[-1]).get("repo_id")
    except Exception:
        return model_key
    return repo_id if isinstance(repo_id, str) else model_key


def _running_language_model(deployment: dict) -> str | None:
    if not _is_visible_deployment(deployment):
        return None

    model_key = deployment.get("model_key", "")
    model_class = model_key.split(":", 1)[0].split(".")[-1]
    if model_class not in _LANGUAGE_MODEL_CLASSES:
        return None
    if deployment.get("application_state", "NOT DEPLOYED") != "RUNNING":
        return None
    return _repo_id_from_model_key(model_key)


def _unexpected_state(deployment: dict) -> tuple[str, str] | None:
    state = deployment.get("application_state", "NOT DEPLOYED")
    if state in _EXPECTED_NDIF_STATES:
        return None
    model_key = deployment.get("model_key", "")
    return _repo_id_from_model_key(model_key), state


@st.cache_data(show_spinner=False, ttl=30)
def list_remote_models() -> list[str]:
    """Return the NDIF language models that are currently running.

    Parses the raw NDIF response directly instead of going through the formatted
    ``nnsight.ndif.status()`` response because formatting crashes whenever NDIF reports
    any deployment with an ``application_state`` that isn't in nnsight's
    ``ModelStatus`` enum (e.g. ``UNHEALTHY``) — one bad deployment poisons
    the whole response. See nnsight 0.6.3 ``ndif.py::status``.
    """

    from nnsight.ndif import status

    try:
        raw = status(raw=True)
    except Exception:
        logger.warning("Failed to fetch NDIF status", exc_info=True)
        return []

    model_names: list[str] = []
    bad_states: list[tuple[str, str]] = []  # (repo_id_or_key, application_state)

    for deployment in _iter_deployments(raw):
        if bad_state := _unexpected_state(deployment):
            bad_states.append(bad_state)
        if model_name := _running_language_model(deployment):
            model_names.append(model_name)

    if bad_states:
        logger.warning(
            "NDIF reported deployments with unexpected application_state values "
            "(nnsight's ModelStatus enum may not know about these): %s",
            bad_states,
        )

    return sorted(set(model_names))


def session_ndif_api_key() -> str | None:
    """Return this visitor's NDIF key without touching process globals."""

    value = st.session_state.get(_SESSION_NDIF_API_KEY)
    return value if isinstance(value, str) and value else None


def configured_ndif_api_key() -> str | None:
    """Return an app-level NDIF key configured through the environment, if any."""

    value = os.environ.get("NDIF_API_KEY")
    return value if value else None


def remote_backend(model: object, api_key: str | None = None, *, on_status=None):
    """Build an NDIF backend with credentials bound to one browser session."""

    from nnsight.intervention.backends.remote import JobStatusDisplay, RemoteBackend

    active_key = api_key or session_ndif_api_key() or configured_ndif_api_key()
    if not active_key:
        raise RuntimeError("Enter your NDIF API key before using remote execution.")

    backend = RemoteBackend(model.to_model_key(), api_key=active_key)
    backend.CONNECT_TIMEOUT = 300.0
    if on_status is None:
        return backend

    class _CallbackJobStatusDisplay(JobStatusDisplay):
        def update(
            self,
            job_id: str = "",
            status_name: str = "",
            description: str = "",
        ):
            super().update(job_id, status_name, description)
            if status_name:
                on_status(job_id, status_name, description)

    backend.status_display = _CallbackJobStatusDisplay(
        enabled=True,
        verbose=backend.verbose,
    )
    return backend


@st.cache_resource(show_spinner=False, max_entries=_MODEL_CACHE_ENTRIES)
def cached_model(model_name: str):
    """Load and cache a standardized nnterp model.

    Streamlit reruns this app on every interaction, so caching keeps one loaded
    model instance instead of reloading weights on every widget change.
    ``remote`` is intentionally not part of the cache key: it matters at
    generation/trace time, but the current ``StandardizedTransformer``
    constructor ignores it, and excluding it avoids loading duplicate local
    model objects when toggling NDIF. The cache defaults to one model to avoid
    keeping multiple large models in RAM.
    """

    import torch
    from nnterp import StandardizedTransformer

    torch.set_grad_enabled(False)

    return StandardizedTransformer(model_name)
