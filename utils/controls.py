import streamlit as st
from persona_vectors.extraction import MaskStrategy


def render_mask_strategy_select(
    *,
    key: str,
    last_key: str,
    help_text: str,
) -> MaskStrategy:
    last_strategy = st.session_state.get(last_key, MaskStrategy.ANSWER_MEAN.value)
    strategies = list(MaskStrategy)
    selected = st.selectbox(
        "Mask strategy",
        options=strategies,
        index=next(
            (
                idx
                for idx, strategy in enumerate(strategies)
                if strategy.value == last_strategy
            ),
            0,
        ),
        format_func=lambda strategy: strategy.value.replace("_", " ").title(),
        key=key,
        help=help_text,
    )
    st.session_state[last_key] = selected.value
    return selected
