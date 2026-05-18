from __future__ import annotations

from collections.abc import Sequence

import streamlit as st


def remembered_segmented_control(
    label: str,
    *,
    options: Sequence[str],
    key: str,
    remember_key: str | None = None,
    default: str | None = None,
    label_visibility: str = "visible",
) -> str:
    """Render a segmented control with one small, reusable memory pattern."""
    fallback = default or options[0]
    remembered = st.session_state.get(
        remember_key,
        st.session_state.get(key, fallback),
    )
    selected = (
        st.segmented_control(
            label,
            options=options,
            default=remembered if remembered in options else fallback,
            key=key,
            label_visibility=label_visibility,
        )
        or fallback
    )
    if remember_key is not None:
        st.session_state[remember_key] = selected
    return selected
