from typing import Literal, cast

import streamlit as st
from persona_vectors.artifacts import SUPPORTED_VARIANTS
from persona_vectors.extraction import run_extraction

from utils.datasets import load_dataset
from utils.helpers import (
    NDIF_STATUS_ICONS,
    persona_label,
    prompt_variant_label,
    widget_key,
)
from utils.runtime import cached_model

# Cross-model / remote-switch persistence — same pattern as compare.py.
# Written on every render so selections survive model or NDIF toggles.
_LAST_VARIANTS_KEY = "extract:last_variants"
_LAST_PERSONA_IDS_KEY = "extract:last_persona_ids"
_LAST_QA_TYPE_KEY = "extract:last_qa_type"
_LAST_DIFFICULTY_KEY = "extract:last_difficulty"
_LAST_MAX_QUESTIONS_KEY = "extract:last_max_questions"

_QA_TYPE_OPTIONS = ["all", "explicit", "implicit"]


def _extract_widget_key(
    model_name: str, remote: bool, dataset_source: str, suffix: str
) -> str:
    return widget_key("extract", str(remote), model_name, dataset_source, suffix)


def _render_local_dataset_uploads() -> None:
    """Render file inputs for local dataset uploads."""

    with st.expander("Local dataset upload", expanded=True):
        st.file_uploader(
            "personas.jsonl",
            type=["jsonl"],
            key="extract__personas_file",
            help="Expected fields: id, persona, templated_view, biography_view",
        )
        st.file_uploader(
            "qa.jsonl",
            type=["jsonl"],
            key="extract__qa_file",
            help="Expected fields: id, qid, type, question, answer, difficulty",
        )


def render_extract_tab(remote: bool, model_name: str, dataset_source: str) -> None:
    """Render the extraction tab."""

    st.title("Extract")

    if dataset_source == "Local JSONL upload":
        _render_local_dataset_uploads()

    last_variants = st.session_state.get(_LAST_VARIANTS_KEY, list(SUPPORTED_VARIANTS))
    default_variants = [v for v in last_variants if v in SUPPORTED_VARIANTS] or list(
        SUPPORTED_VARIANTS
    )
    selected_variants = st.multiselect(
        "Prompt variants",
        options=SUPPORTED_VARIANTS,
        default=default_variants,
        format_func=prompt_variant_label,
        key=_extract_widget_key(model_name, remote, dataset_source, "prompt_variants"),
    )
    st.session_state[_LAST_VARIANTS_KEY] = selected_variants
    if not selected_variants:
        st.info("Select at least one prompt variant.")
        return

    try:
        dataset, dataset_status = load_dataset(
            dataset_source,
            personas_file=st.session_state.get("extract__personas_file"),
            qa_file=st.session_state.get("extract__qa_file"),
        )
        st.caption(dataset_status)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.info(
            "Upload both JSONL files or switch to the built-in SynthPersona source."
        )
        return

    personas = list(dataset)
    if not personas:
        st.warning("No personas found in the selected dataset.")
        st.info(
            "Try another dataset source or check that the personas file is not empty."
        )
        return

    last_persona_ids: set[str] = set(st.session_state.get(_LAST_PERSONA_IDS_KEY, []))
    default_personas = [p for p in personas if p.id in last_persona_ids] or [
        personas[0]
    ]
    selected_personas = st.multiselect(
        "Personas",
        options=personas,
        default=default_personas,
        format_func=persona_label,
        key=_extract_widget_key(model_name, remote, dataset_source, "persona_select"),
    )
    st.session_state[_LAST_PERSONA_IDS_KEY] = [p.id for p in selected_personas]

    if not selected_personas:
        st.info("Select at least one persona.")
        return

    max_questions = 0

    with st.expander("Advanced", expanded=False):
        st.caption("Filters")

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            last_qa_type = st.session_state.get(_LAST_QA_TYPE_KEY, "all")
            qa_type_index = (
                _QA_TYPE_OPTIONS.index(last_qa_type)
                if last_qa_type in _QA_TYPE_OPTIONS
                else 0
            )
            qa_type_select = st.selectbox(
                "QA type",
                options=_QA_TYPE_OPTIONS,
                index=qa_type_index,
                key=_extract_widget_key(
                    model_name, remote, dataset_source, "qa_type_select"
                ),
            )
            st.session_state[_LAST_QA_TYPE_KEY] = qa_type_select
            qa_filter_type: Literal["explicit", "implicit"] | None = (
                cast(Literal["explicit", "implicit"], qa_type_select)
                if qa_type_select in ("explicit", "implicit")
                else None
            )
        with col2:
            last_difficulty = st.session_state.get(_LAST_DIFFICULTY_KEY, [1, 2, 3])
            default_difficulty = [d for d in last_difficulty if d in (1, 2, 3)] or [
                1,
                2,
                3,
            ]
            difficulty_values = st.multiselect(
                "Difficulty",
                options=[1, 2, 3],
                default=default_difficulty,
                key=_extract_widget_key(
                    model_name, remote, dataset_source, "difficulty_select"
                ),
            )
            st.session_state[_LAST_DIFFICULTY_KEY] = difficulty_values
            qa_filter_difficulty = difficulty_values if difficulty_values else None

        runs, skipped = [], []
        for persona in selected_personas:
            qa = list(
                dataset.get_qa(
                    persona.id, type=qa_filter_type, difficulty=qa_filter_difficulty
                )
            )
            if qa:
                runs.append((persona, qa))
            else:
                skipped.append(persona)
        if skipped:
            names = ", ".join(p.name for p in skipped)
            st.warning(f"No QA pairs match filters for: {names}. They will be skipped.")

        if not runs:
            st.info("No personas have matching QA pairs. Widen the filters.")
            return

        max_q = min(len(qa_pairs) for _, qa_pairs in runs)
        last_max = st.session_state.get(_LAST_MAX_QUESTIONS_KEY, max_q)
        default_max = min(max(last_max, 1), max_q)
        max_questions = st.slider(
            "Max questions",
            min_value=1,
            max_value=max_q,
            value=default_max,
            key=_extract_widget_key(
                model_name, remote, dataset_source, "max_questions"
            ),
        )
        st.session_state[_LAST_MAX_QUESTIONS_KEY] = max_questions

    run_clicked = st.button("Run extraction", type="primary")
    if not run_clicked:
        return

    status_box = st.empty()
    status_box.info("Extraction in progress...")
    progress = st.progress(0, text="Preparing extraction...")
    ndif_status_box = st.empty()  # shows live NDIF job status when remote=True

    def _on_ndif_status(job_id: str, status_name: str, description: str) -> None:
        icon = NDIF_STATUS_ICONS.get(status_name, "•")
        ndif_status_box.caption(f"{icon} `{job_id}` **{status_name}** — {description}")

    with st.spinner("Loading model..."):
        model = cached_model(model_name=model_name, remote=remote)

    try:
        total_steps = len(runs) * len(selected_variants)
        step = 0
        results = []

        for persona, qa_pairs in runs:
            for variant in selected_variants:
                progress.progress(
                    step / total_steps if total_steps else 1.0,
                    text=f"{persona.name} · {prompt_variant_label(variant)} ({step + 1}/{total_steps})",
                )
                variant_results = run_extraction(
                    model=model,
                    model_name=model_name,
                    persona=persona,
                    qa_pairs=qa_pairs[:max_questions],
                    variants=(variant,),
                    remote=remote,
                    on_status=_on_ndif_status if remote else None,
                )
                results.extend(variant_results)
                step += 1

        progress.progress(1.0, text="Extraction complete")
    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        return
    finally:
        progress.empty()
        ndif_status_box.empty()

    status_box.success("Extraction complete")
    st.success(f"Saved {len(results)} artifact set(s)")

    for result in results:
        st.markdown(
            f"- **{result.persona_name}** · {prompt_variant_label(result.variant)}: "
            f"{result.n_questions} questions"
        )
