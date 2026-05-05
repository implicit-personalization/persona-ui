import logging

import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False, ttl=30)
def list_remote_models() -> list[str]:
    """Return the NDIF language models that are currently running.

    Parses the raw NDIF response directly instead of going through
    ``nnsight.ndif_status()`` because that call crashes whenever NDIF reports
    any deployment with an ``application_state`` that isn't in nnsight's
    ``ModelStatus`` enum (e.g. ``UNHEALTHY``) — one bad deployment poisons
    the whole response. See nnsight 0.6.3 ``ndif.py::status``.
    """

    import json

    import nnsight

    try:
        raw = nnsight.ndif_status(raw=True)
    except Exception:
        logger.warning("Failed to fetch NDIF status", exc_info=True)
        return []

    model_names: list[str] = []
    bad_states: list[tuple[str, str]] = []  # (repo_id_or_key, application_state)

    for value in (raw or {}).get("deployments", {}).values():
        if not isinstance(value, dict):
            continue
        if (
            value.get("deployment_level") not in {"HOT", "WARM"}
            and "schedule" not in value
        ):
            continue

        model_key = value.get("model_key", "")
        model_class = model_key.split(":", 1)[0].split(".")[-1]
        try:
            repo_id = json.loads(model_key.split(":", 1)[-1]).get("repo_id")
        except Exception:
            repo_id = model_key

        state = value.get("application_state", "NOT DEPLOYED")
        if state not in {"RUNNING", "NOT DEPLOYED", "DEPLOYING", "DELETING"}:
            bad_states.append((repo_id or model_key, state))

        if model_class not in {"LanguageModel", "StandardizedTransformer"}:
            continue
        if state != "RUNNING":
            continue
        if isinstance(repo_id, str):
            model_names.append(repo_id)

    if bad_states:
        logger.warning(
            "NDIF reported deployments with unexpected application_state values "
            "(nnsight's ModelStatus enum may not know about these): %s",
            bad_states,
        )

    return sorted(set(model_names))


@st.cache_resource(show_spinner=False, max_entries=1)
def _cached_model_by_name(model_name: str):
    """Load and cache a standardized nnterp model.

    Streamlit reruns this app on every interaction, so caching keeps one loaded
    model instance per model name instead of reloading weights on every widget
    change.
    """

    from nnterp import StandardizedTransformer

    # The remote constructor path is currently unstable for this model wrapper.
    # return StandardizedTransformer(model_name, remote=remote, check_renaming=False)
    return StandardizedTransformer(model_name)


def cached_model(model_name: str, remote: bool):
    """Return the cached model for ``model_name``.

    ``remote`` still matters at generation/trace time, but the current
    ``StandardizedTransformer`` constructor ignores it. Keeping it out of the
    cache key avoids loading duplicate local model objects when toggling NDIF.
    """

    return _cached_model_by_name(model_name)
