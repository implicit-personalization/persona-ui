import os

import streamlit as st
from dotenv import load_dotenv

from utils.helpers import DATASET_SOURCES

load_dotenv()
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "google/gemma-2-2b-it")
REMOTE_DEFAULT_MODEL = os.environ.get("REMOTE_DEFAULT_MODEL", "google/gemma-2-9b-it")


_TABS = ["Chat", "Compare", "Extract"]
_TAB_ICONS = [":material/chat:", ":material/search:", ":material/tune:"]


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
            custom_remote_key = "sidebar__remote_model_custom_enabled"
            custom_remote_model = st.toggle(
                "Custom remote model",
                value=False,
                key=custom_remote_key,
                help="Enter any NDIF-loadable model id, even if it is not currently running.",
            )
            if remote_models:
                default_model = (
                    REMOTE_DEFAULT_MODEL
                    if REMOTE_DEFAULT_MODEL in remote_models
                    else remote_models[0]
                )
                if custom_remote_model:
                    model_name = st.text_input(
                        "Model",
                        value=st.session_state.get(
                            "sidebar__remote_model_custom_value",
                            REMOTE_DEFAULT_MODEL,
                        ),
                        key="sidebar__remote_model_custom_value",
                        help="NDIF model id. Example: openai/gpt-oss-20b",
                    )
                    st.caption(
                        f"{len(remote_models)} running NDIF model(s) detected. Custom model ids can cold-load if your NDIF account allows it."
                    )
                else:
                    selected_remote_model = st.selectbox(
                        "Model",
                        options=remote_models,
                        index=remote_models.index(default_model),
                        key="sidebar__remote_model",
                        help="Running NDIF model.",
                    )
                    model_name = selected_remote_model
            else:
                st.warning("No running NDIF models found.")
                model_name = st.text_input(
                    "Model",
                    value=st.session_state.get(
                        "sidebar__remote_model_custom_value",
                        REMOTE_DEFAULT_MODEL,
                    ),
                    key="sidebar__remote_model_custom_value",
                    help="NDIF model id. Use this to cold-load a remote model.",
                )
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
