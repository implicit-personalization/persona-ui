import os
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv

from utils.helpers import DATASET_SOURCES

load_dotenv()
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "google/gemma-2-2b-it")
REMOTE_DEFAULT_MODEL = os.environ.get("REMOTE_DEFAULT_MODEL", "google/gemma-2-9b-it")
_LAST_LOCAL_MODEL_KEY = "sidebar:last_local_model"
_LAST_REMOTE_MODEL_KEY = "sidebar:last_remote_model"


_TABS = ["Chat", "Compare", "Extract"]
_TAB_ICONS = [":material/chat:", ":material/search:", ":material/tune:"]


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
                "sidebar__remote_model_custom_value", last_remote
            ),
            key="sidebar__remote_model_custom_value",
            help="NDIF model id. Use this to cold-load a remote model.",
        )
        st.session_state[_LAST_REMOTE_MODEL_KEY] = model_name
        return model_name

    custom = st.toggle(
        "Custom remote model",
        value=False,
        key="sidebar__remote_model_custom_enabled",
        help="Enter any NDIF-loadable model id, even if it is not currently running.",
    )
    if custom:
        model_name = st.text_input(
            "Model",
            value=st.session_state.get(
                "sidebar__remote_model_custom_value", last_remote
            ),
            key="sidebar__remote_model_custom_value",
            help="NDIF model id. Example: openai/gpt-oss-20b",
        )
        st.caption(
            f"{len(remote_models)} running NDIF model(s) detected. "
            "Custom model ids can cold-load if your NDIF account allows it."
        )
    else:
        default_model = st.session_state.get("sidebar__remote_model", last_remote)
        if default_model not in remote_models:
            default_model = (
                REMOTE_DEFAULT_MODEL
                if REMOTE_DEFAULT_MODEL in remote_models
                else remote_models[0]
            )
        if st.session_state.get("sidebar__remote_model") not in remote_models:
            st.session_state["sidebar__remote_model"] = default_model
        model_name = st.selectbox(
            "Model",
            options=remote_models,
            index=remote_models.index(default_model),
            key="sidebar__remote_model",
            help="Running NDIF model.",
        )
    st.session_state[_LAST_REMOTE_MODEL_KEY] = model_name
    return model_name


def _sidebar_controls() -> SidebarState:
    from utils.runtime import list_remote_models

    with st.sidebar:
        st.markdown("## Persona UI")

        if "sidebar__active_tab" not in st.session_state:
            st.session_state["sidebar__active_tab"] = "Chat"

        active_tab = st.session_state["sidebar__active_tab"]
        for tab_name, icon in zip(_TABS, _TAB_ICONS, strict=True):
            is_selected = tab_name == active_tab
            if st.button(
                tab_name,
                key=f"sidebar__tab__{tab_name.lower()}",
                width="stretch",
                type="primary" if is_selected else "secondary",
                icon=icon,
            ):
                st.session_state["sidebar__active_tab"] = tab_name
                st.rerun()

        if active_tab == "Compare":
            model_name = st.session_state.get(_LAST_LOCAL_MODEL_KEY, DEFAULT_MODEL)
            dataset_source = st.session_state.get(
                "sidebar__dataset_source",
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
        remote = st.toggle("Remote (NDIF)", value=False, key="sidebar__remote")

        if remote:
            model_name = _remote_model_input(list_remote_models())
        else:
            model_name = st.text_input(
                "Model",
                value=st.session_state.get(_LAST_LOCAL_MODEL_KEY, DEFAULT_MODEL),
                key="sidebar__local_model",
                help="Local model id or path.",
            )
            st.session_state[_LAST_LOCAL_MODEL_KEY] = model_name

        st.caption("Data")
        dataset_source = st.selectbox(
            "Source",
            DATASET_SOURCES,
            key="sidebar__dataset_source",
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
    from utils.theme import install_catppuccin_theme

    install_catppuccin_theme(st.get_option("theme.base"))

    # Deferred: importing torch is slow; keep it after dotenv load (done at
    # module level above) so the Streamlit page config renders immediately.
    import torch

    torch.set_grad_enabled(False)

    sidebar = _sidebar_controls()

    if sidebar.active_tab == "Extract":
        from tabs.extract import render_extract_tab

        render_extract_tab(sidebar.remote, sidebar.model_name, sidebar.dataset_source)
    elif sidebar.active_tab == "Compare":
        from tabs.compare import render_compare_tab

        render_compare_tab()
    else:
        from tabs.chat import render_chat_tab

        render_chat_tab(sidebar.remote, sidebar.model_name, sidebar.dataset_source)


if __name__ == "__main__":
    main()
