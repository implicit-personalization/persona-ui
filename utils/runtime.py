import logging

import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False, ttl=30)
def list_remote_models() -> list[str]:
    """Return the NDIF language models that are currently running."""

    import nnsight

    try:
        status = nnsight.ndif_status()
    except Exception:
        logger.warning("Failed to fetch NDIF status", exc_info=True)
        return []

    model_names: list[str] = []

    for entry in status.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("model_class") not in {"LanguageModel", "StandardizedTransformer"}:
            continue

        state = entry.get("state")
        state_name = getattr(state, "name", None) or getattr(state, "value", None)
        if state_name != "RUNNING":
            continue

        repo_id = entry.get("repo_id")
        if isinstance(repo_id, str):
            model_names.append(repo_id)

    return sorted(set(model_names))


@st.cache_resource(show_spinner=False, max_entries=1)
def cached_model(model_name: str, remote: bool):
    """Load and cache a standardized nnterp model.

    Streamlit reruns this app on every interaction, so caching keeps one loaded
    model instance per ``(model_name, remote)`` instead of reloading weights on
    every widget change.
    """

    from nnterp import StandardizedTransformer

    # HACK: For now do it like this because of the bug.
    # model = StandardizedTransformer(model_name, remote=True)
    return StandardizedTransformer(model_name)
