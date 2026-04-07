import streamlit as st
from persona_vectors.extraction import run_extraction

from utils.datasets import load_dataset
from utils.helpers import (
    NDIF_STATUS_ICONS,
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

    runs = None
    max_questions = 0

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
        max_questions = st.slider(
            "Max questions",
            min_value=1,
            max_value=max_q,
            value=max_q,
            key=_extract_widget_key(
                model_name, remote, dataset_source, "max_questions"
            ),
        )

    if runs is None:
        return

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
                    variants=[variant],
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
