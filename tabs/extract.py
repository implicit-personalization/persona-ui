import streamlit as st

from utils.datasets import load_dataset
from utils.extraction import run_extraction
from utils.helpers import (
    PROMPT_VARIANTS,
    persona_label,
    prompt_variant_label,
    widget_key,
)
from utils.runtime import cached_model


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
            help="Expected fields: id, persona, templated_prompt, biography_md",
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

    selected_variants = st.multiselect(
        "Prompt variants",
        options=PROMPT_VARIANTS,
        default=PROMPT_VARIANTS,
        format_func=prompt_variant_label,
        key=_extract_widget_key(model_name, remote, dataset_source, "prompt_variants"),
    )
    if not selected_variants:
        st.info("Select at least one prompt variant.")
        return

    try:
        dataset, dataset_status = load_dataset(dataset_source)
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

    selected_personas = st.multiselect(
        "Personas",
        options=personas,
        default=[personas[0]] if personas else [],
        format_func=persona_label,
        key=_extract_widget_key(model_name, remote, dataset_source, "persona_select"),
    )

    if not selected_personas:
        st.info("Select at least one persona.")
        return

    qa_filter_type: str | None
    qa_filter_difficulty: list[int] | None

    with st.expander("Advanced", expanded=False):
        st.caption("Filters")

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            qa_type_select = st.selectbox(
                "QA type",
                options=["all", "explicit", "implicit"],
                index=0,
                key=_extract_widget_key(
                    model_name, remote, dataset_source, "qa_type_select"
                ),
            )
            qa_filter_type = (
                qa_type_select if qa_type_select in ("explicit", "implicit") else None
            )
        with col2:
            difficulty_values = st.multiselect(
                "Difficulty",
                options=[1, 2, 3],
                default=[1, 2, 3],
                key=_extract_widget_key(
                    model_name, remote, dataset_source, "difficulty_select"
                ),
            )
            qa_filter_difficulty = difficulty_values if difficulty_values else None

        # Pre-load QA pairs for all selected personas to validate filters and set slider range.
        qa_by_persona = {
            p.id: dataset.get_qa(
                p.id, type=qa_filter_type, difficulty=qa_filter_difficulty
            )
            for p in selected_personas
        }
        personas_without_qa = [p for p in selected_personas if not qa_by_persona[p.id]]
        if personas_without_qa:
            names = ", ".join(p.name for p in personas_without_qa)
            st.warning(f"No QA pairs match filters for: {names}. They will be skipped.")

        personas_to_run = [p for p in selected_personas if qa_by_persona[p.id]]
        if not personas_to_run:
            st.info("No personas have matching QA pairs. Widen the filters.")
            return

        min_qa_count = min(len(qa_by_persona[p.id]) for p in personas_to_run)

        with col3:
            max_questions = st.slider(
                "Max questions",
                min_value=1,
                max_value=min_qa_count,
                value=min_qa_count,
                key=_extract_widget_key(
                    model_name, remote, dataset_source, "max_questions"
                ),
            )

    run_clicked = st.button("Run extraction", type="primary")
    if not run_clicked:
        return

    status_box = st.empty()
    status_box.info("Extraction in progress...")
    progress = st.progress(0, text="Preparing extraction...")

    with st.spinner("Loading model..."):
        model = cached_model(model_name=model_name, remote=remote)

    try:
        total_steps = len(personas_to_run) * len(selected_variants)
        step = 0
        results = []

        for persona in personas_to_run:
            qa_pairs = qa_by_persona[persona.id][:max_questions]
            for variant in selected_variants:
                progress.progress(
                    step / total_steps if total_steps else 1.0,
                    text=f"{persona.name} · {prompt_variant_label(variant)} ({step + 1}/{total_steps})",
                )
                variant_results = run_extraction(
                    model=model,
                    model_name=model_name,
                    persona=persona,
                    qa_pairs=qa_pairs,
                    variants=[variant],
                    remote=remote,
                )
                results.extend(variant_results)
                step += 1

        progress.progress(1.0, text="Extraction complete")
    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        return
    finally:
        progress.empty()

    status_box.success("Extraction complete")
    st.success(f"Saved {len(results)} artifact set(s)")

    for result in results:
        st.markdown(
            f"- **{result.persona_name}** · {prompt_variant_label(result.variant)}: "
            f"{result.n_questions} questions, {result.n_layers} layers, {result.d_model} hidden size"
        )
