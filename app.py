import os
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv

from utils.analysis_sources import DEFAULT_COMPARE_MODEL, DEFAULT_HUB_REPO, SOURCE_HUB
from utils.helpers import DATASET_SOURCES, session_key
from utils.preload import preload_once
from utils.runtime import configured_ndif_api_key, list_remote_models
from utils.theme import active_base, install_catppuccin_theme

load_dotenv()
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "google/gemma-2-2b-it")
REMOTE_DEFAULT_MODEL = os.environ.get("REMOTE_DEFAULT_MODEL", "google/gemma-2-9b-it")
_LAST_LOCAL_MODEL_KEY = session_key("sidebar", "last_local_model")
_LAST_REMOTE_MODEL_KEY = session_key("sidebar", "last_remote_model")
_SIDEBAR_ACTIVE_TAB_KEY = session_key("sidebar", "active_tab")
_SIDEBAR_REMOTE_MODEL_CUSTOM_VALUE_KEY = session_key(
    "sidebar", "remote_model_custom_value"
)
_SIDEBAR_REMOTE_MODEL_CUSTOM_ENABLED_KEY = session_key(
    "sidebar", "remote_model_custom_enabled"
)
_SIDEBAR_REMOTE_MODEL_KEY = session_key("sidebar", "remote_model")
_SIDEBAR_LOCAL_MODEL_KEY = session_key("sidebar", "local_model")
_SIDEBAR_REMOTE_KEY = session_key("sidebar", "remote")
_SIDEBAR_DATASET_SOURCE_KEY = session_key("sidebar", "dataset_source")
_SIDEBAR_NDIF_API_KEY = session_key("sidebar", "ndif_api_key")
NDIF_REGISTRATION_URL = "https://login.ndif.us/"


_TABS = ["Chat", "Analysis", "Probing", "Extract"]
_TAB_ICONS = [
    ":material/chat:",
    ":material/search:",
    ":material/biotech:",
    ":material/tune:",
]
_TAB_PRELOAD_MODULES = {
    "Chat": ("tabs.analysis_core", "tabs.extract", "tabs.compare_chat", "tabs.probe"),
    "Analysis": ("tabs.chat", "tabs.extract", "tabs.probe"),
    "Probing": ("tabs.chat", "tabs.analysis_core", "tabs.extract"),
    "Extract": ("tabs.chat", "tabs.analysis_core", "tabs.probe"),
}
_TAB_PRELOAD_FUNCTIONS = {
    "Chat": ("utils.analysis_metadata:synth_persona_attribute_names",),
    "Probing": ("utils.analysis_metadata:synth_persona_attribute_names",),
    "Extract": ("utils.analysis_metadata:synth_persona_attribute_names",),
}


def _hub_metadata_preload_calls() -> tuple[
    tuple[str, tuple[str, str, str, str | None]], ...
]:
    """Warm Hub metadata for the currently-selected source, if any.

    Reads only the shared ``source:*`` keys (written by ``utils.source_controls``)
    so this stays a small, single-purpose preload. Per-tab divergence used to
    matter when each tab picked its own model; the shared keys are now the
    source of truth.
    """

    if st.session_state.get("source:last_source", SOURCE_HUB) != SOURCE_HUB:
        return ()
    repo = st.session_state.get("source:hub_repo", DEFAULT_HUB_REPO)
    model = st.session_state.get("source:hub_model", DEFAULT_COMPARE_MODEL)
    mask_strategy = st.session_state.get("source:last_mask_strategy", "answer_mean")
    variant = st.session_state.get("probe:variant") or st.session_state.get(
        "analysis:last_projection_variant"
    )
    return (
        (
            "utils.analysis_sources:prefetch_hub_metadata",
            (repo, model, mask_strategy, variant),
        ),
    )


@dataclass(frozen=True)
class SidebarState:
    remote: bool
    model_name: str
    dataset_source: str
    active_tab: str


def _remote_model_input(remote_models: list[str]) -> str:
    """Return the active remote model id, picking from running NDIF deployments or a custom value."""

    last_remote = st.session_state.get(_LAST_REMOTE_MODEL_KEY, REMOTE_DEFAULT_MODEL)

    if not remote_models:
        st.warning("No running NDIF models found.")
        model_name = st.text_input(
            "Model",
            value=st.session_state.get(
                _SIDEBAR_REMOTE_MODEL_CUSTOM_VALUE_KEY, last_remote
            ),
            key=_SIDEBAR_REMOTE_MODEL_CUSTOM_VALUE_KEY,
            help="NDIF model id. Use this to cold-load a remote model.",
        )
        st.session_state[_LAST_REMOTE_MODEL_KEY] = model_name
        return model_name

    custom = st.toggle(
        "Custom remote model",
        value=False,
        key=_SIDEBAR_REMOTE_MODEL_CUSTOM_ENABLED_KEY,
        help="Enter any NDIF-loadable model id, even if it is not currently running.",
    )
    if custom:
        model_name = st.text_input(
            "Model",
            value=st.session_state.get(
                _SIDEBAR_REMOTE_MODEL_CUSTOM_VALUE_KEY, last_remote
            ),
            key=_SIDEBAR_REMOTE_MODEL_CUSTOM_VALUE_KEY,
            help="NDIF model id. Example: openai/gpt-oss-20b",
        )
        st.caption(
            f"{len(remote_models)} running NDIF model(s) detected. "
            "Custom model ids can cold-load if your NDIF account allows it."
        )
    else:
        default_model = st.session_state.get(_SIDEBAR_REMOTE_MODEL_KEY, last_remote)
        if default_model not in remote_models:
            default_model = (
                REMOTE_DEFAULT_MODEL
                if REMOTE_DEFAULT_MODEL in remote_models
                else remote_models[0]
            )
        model_name = st.selectbox(
            "Model",
            options=remote_models,
            index=remote_models.index(default_model),
            key=_SIDEBAR_REMOTE_MODEL_KEY,
            help="Running NDIF model.",
        )
    st.session_state[_LAST_REMOTE_MODEL_KEY] = model_name
    return model_name


def _ndif_api_key_input() -> None:
    """Prompt for a per-session NDIF API key."""

    if configured_ndif_api_key():
        st.caption("Using NDIF API key from environment.")
        return

    api_key = st.text_input(
        "NDIF API key",
        type="password",
        key=_SIDEBAR_NDIF_API_KEY,
        help=f"Required for remote (NDIF) execution. Register at {NDIF_REGISTRATION_URL}",
    )
    if not api_key:
        st.caption(f"No NDIF API key found. [Get one]({NDIF_REGISTRATION_URL}).")


def _sidebar_controls() -> SidebarState:
    with st.sidebar:
        st.markdown("## Persona UI")

        if _SIDEBAR_ACTIVE_TAB_KEY not in st.session_state:
            st.session_state[_SIDEBAR_ACTIVE_TAB_KEY] = "Chat"

        active_tab = st.session_state[_SIDEBAR_ACTIVE_TAB_KEY]
        for tab_name, icon in zip(_TABS, _TAB_ICONS, strict=True):
            is_selected = tab_name == active_tab
            if st.button(
                tab_name,
                key=f"sidebar__tab__{tab_name.lower()}",
                width="stretch",
                type="primary" if is_selected else "secondary",
                icon=icon,
            ):
                st.session_state[_SIDEBAR_ACTIVE_TAB_KEY] = tab_name
                st.rerun()

        if active_tab in {"Analysis", "Probing"}:
            # These tabs select their own model in-tab. The global sidebar
            # only carries over the last local model id for breadcrumbs.
            model_name = st.session_state.get(_LAST_LOCAL_MODEL_KEY, DEFAULT_MODEL)
            dataset_source = st.session_state.get(
                _SIDEBAR_DATASET_SOURCE_KEY,
                DATASET_SOURCES[0],
            )
            return SidebarState(
                remote=False,
                model_name=model_name,
                dataset_source=dataset_source,
                active_tab=active_tab,
            )

        st.divider()
        st.caption("Runtime")
        _ndif_api_key_input()
        remote = st.toggle("Remote (NDIF)", value=False, key=_SIDEBAR_REMOTE_KEY)

        if remote:
            model_name = _remote_model_input(list_remote_models())
        else:
            model_name = st.text_input(
                "Model",
                value=st.session_state.get(_LAST_LOCAL_MODEL_KEY, DEFAULT_MODEL),
                key=_SIDEBAR_LOCAL_MODEL_KEY,
                help="Local model id or path.",
            )
            st.session_state[_LAST_LOCAL_MODEL_KEY] = model_name

        st.caption("Data")
        dataset_source = st.selectbox(
            "Source",
            DATASET_SOURCES,
            key=_SIDEBAR_DATASET_SOURCE_KEY,
            help="Dataset for Chat and Extract.",
        )

    return SidebarState(
        remote=remote,
        model_name=model_name,
        dataset_source=dataset_source,
        active_tab=active_tab,
    )


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(page_title="Persona UI", layout="wide")
    install_catppuccin_theme(active_base())

    sidebar = _sidebar_controls()

    if sidebar.active_tab == "Extract":
        from tabs.extract import render_extract_tab

        render_extract_tab(sidebar.remote, sidebar.model_name, sidebar.dataset_source)
    elif sidebar.active_tab == "Analysis":
        from tabs.analysis_core import render_analysis_tab

        render_analysis_tab()
    elif sidebar.active_tab == "Probing":
        from tabs.probe import render_probing_tab

        render_probing_tab()
    else:
        from tabs.chat import render_chat_tab

        render_chat_tab(sidebar.remote, sidebar.model_name, sidebar.dataset_source)

    preload_once(
        f"after-{sidebar.active_tab.lower()}",
        modules=_TAB_PRELOAD_MODULES.get(sidebar.active_tab, ()),
        functions=_TAB_PRELOAD_FUNCTIONS.get(sidebar.active_tab, ()),
        calls=_hub_metadata_preload_calls(),
    )


if __name__ == "__main__":
    main()
