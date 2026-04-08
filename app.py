import os

import streamlit as st
from dotenv import load_dotenv

from utils.helpers import DATASET_SOURCES

load_dotenv()
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "google/gemma-2-2b-it")
REMOTE_DEFAULT_MODEL = os.environ.get("REMOTE_DEFAULT_MODEL", "google/gemma-2-9b-it")
NDIF_API_KEY = os.environ.get("NDIF_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))


_TABS = ["Chat", "Compare", "Extract"]
_TAB_ICONS = [":material/chat:", ":material/search:", ":material/tune:"]


def _sync_sidebar_api_key(env_var: str, value: str) -> None:
    if value:
        os.environ[env_var] = value


def _sidebar_api_keys() -> None:
    with st.sidebar:
        st.divider()
        st.caption("API Keys")

        ndif_api_key = st.text_input(
            "NDIF API key",
            value=NDIF_API_KEY,
            type="password",
            key="sidebar__ndif_api_key",
            help="Overrides NDIF_API_KEY for this session.",
        )
        _sync_sidebar_api_key("NDIF_API_KEY", ndif_api_key)

        hf_token = st.text_input(
            "Hugging Face token",
            value=HF_TOKEN,
            type="password",
            key="sidebar__hf_token",
            help="Overrides HF_TOKEN and HUGGING_FACE_HUB_TOKEN for this session.",
        )
        _sync_sidebar_api_key("HF_TOKEN", hf_token)
        _sync_sidebar_api_key("HUGGING_FACE_HUB_TOKEN", hf_token)


def _sidebar_controls() -> tuple[bool, str, str, str]:
    from utils.runtime import list_remote_models

    with st.sidebar:
        st.markdown("# Persona UI")
        st.caption("Chat, extract, and compare persona runs.")

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

        st.divider()
        st.caption("Runtime")
        remote = st.toggle("Remote (NDIF)", value=False, key="sidebar__remote")

        if remote:
            remote_models = list_remote_models()
            if remote_models:
                default_model = (
                    REMOTE_DEFAULT_MODEL
                    if REMOTE_DEFAULT_MODEL in remote_models
                    else remote_models[0]
                )
                model_name = st.selectbox(
                    "Model",
                    options=remote_models,
                    index=remote_models.index(default_model),
                    key="sidebar__remote_model",
                    help="Running NDIF model.",
                )
            else:
                st.error("No running NDIF models found.")
                model_name = REMOTE_DEFAULT_MODEL
        else:
            model_name = st.text_input(
                "Model",
                value=DEFAULT_MODEL,
                key="sidebar__local_model",
                help="Local model id or path.",
            )

        st.caption("Data")
        dataset_source = st.selectbox(
            "Source",
            DATASET_SOURCES,
            key="sidebar__dataset_source",
            help="Dataset for Chat and Extract.",
        )

    _sidebar_api_keys()

    return remote, model_name, dataset_source, active_tab


def main() -> None:
    """Run the Streamlit app."""

    # Deferred: importing torch is slow; keep it after dotenv load (done at
    # module level above) so the Streamlit page config renders immediately.
    import torch

    torch.set_grad_enabled(False)

    st.set_page_config(page_title="Persona UI", layout="wide")
    remote, model_name, dataset_source, active_tab = _sidebar_controls()

    if active_tab == "Extract":
        from tabs.extract import render_extract_tab

        render_extract_tab(remote, model_name, dataset_source)
    elif active_tab == "Compare":
        from tabs.compare import render_compare_tab

        render_compare_tab(model_name)
    else:
        from tabs.chat import render_chat_tab

        render_chat_tab(remote, model_name, dataset_source)


if __name__ == "__main__":
    main()
