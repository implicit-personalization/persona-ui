import json
import logging
from collections.abc import Iterable

import streamlit as st

logger = logging.getLogger(__name__)
_LANGUAGE_MODEL_CLASSES = {"LanguageModel", "StandardizedTransformer"}
_EXPECTED_NDIF_STATES = {"RUNNING", "NOT DEPLOYED", "DEPLOYING", "DELETING"}


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

    Parses the raw NDIF response directly instead of going through
    ``nnsight.ndif_status()`` because that call crashes whenever NDIF reports
    any deployment with an ``application_state`` that isn't in nnsight's
    ``ModelStatus`` enum (e.g. ``UNHEALTHY``) — one bad deployment poisons
    the whole response. See nnsight 0.6.3 ``ndif.py::status``.
    """

    import nnsight

    try:
        raw = nnsight.ndif_status(raw=True)
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


@st.cache_resource(show_spinner=False, max_entries=1)
def cached_model(model_name: str):
    """Load and cache a standardized nnterp model.

    Streamlit reruns this app on every interaction, so caching keeps one loaded
    model instance per model name instead of reloading weights on every widget
    change. ``remote`` is intentionally not part of the cache key: it matters
    at generation/trace time, but the current ``StandardizedTransformer``
    constructor ignores it, and excluding it avoids loading duplicate local
    model objects when toggling NDIF.
    """

    from nnterp import StandardizedTransformer

    return StandardizedTransformer(model_name)
