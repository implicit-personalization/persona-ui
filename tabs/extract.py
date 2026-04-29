import html
from typing import Literal, cast

import streamlit as st
from persona_data.prompts import (
    BASELINE_PERSONA_ID,
    BASELINE_PERSONA_NAME,
    system_prompt_for_variant,
)
from persona_data.synth_persona import PersonaData, QAPair
from persona_vectors.artifacts import SUPPORTED_VARIANTS
from persona_vectors.extraction import (
    MaskStrategy,
    PreparedInput,
    prepare_inputs_for_strategy,
    run_extraction,
)

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
_LAST_MASK_STRATEGY_KEY = "extract:last_mask_strategy"

_QA_TYPE_OPTIONS = ["all", "explicit", "implicit"]


def _baseline_persona() -> PersonaData:
    return PersonaData(
        id=BASELINE_PERSONA_ID,
        persona={"first_name": BASELINE_PERSONA_NAME, "last_name": ""},
        templated_view=BASELINE_PERSONA_NAME,
        biography_view=BASELINE_PERSONA_NAME,
    )


def _build_run_plan(
    selected_variants: list[str],
    runs: list[tuple[PersonaData, list[QAPair]]],
) -> list[tuple[PersonaData, list[QAPair], str]]:
    regular_variants = [v for v in selected_variants if v != "baseline"]
    run_plan = [
        (persona, qa_pairs, variant)
        for persona, qa_pairs in runs
        for variant in regular_variants
    ]
    if "baseline" in selected_variants:
        _, baseline_qa_pairs = runs[0]
        run_plan.append((_baseline_persona(), baseline_qa_pairs, "baseline"))
    return run_plan


def _extract_widget_key(
    model_name: str, remote: bool, dataset_source: str, suffix: str
) -> str:
    return widget_key("extract", str(remote), model_name, dataset_source, suffix)


_TOKEN_LEGEND = (
    '<div style="display:flex;gap:12px;flex-wrap:wrap;font-size:0.8em;margin-bottom:8px">'
    '<span style="background:#86efac;color:black;padding:1px 6px;border-radius:3px">masked</span>'
    '<span style="color:#fde047;padding:1px 6px">question</span>'
    '<span style="color:#22d3ee;padding:1px 6px">response</span>'
    '<span style="color:#d946ef;font-weight:bold;padding:1px 6px">special</span>'
    '<span style="color:#9ca3af;padding:1px 6px">template</span>'
    "</div>"
)

_MAX_PREVIEW_SAMPLES = 3


def _token_style_for_index(
    p: PreparedInput, token_idx: int, special_ids: set[int]
) -> str:
    if p.spans.response.token_start <= token_idx < p.spans.response.token_end:
        style = "color:#22d3ee"
    elif p.spans.question.token_start <= token_idx < p.spans.question.token_end:
        style = "color:#fde047"
    else:
        style = "color:#9ca3af"

    if int(p.input_ids[token_idx]) in special_ids:
        style = "color:#d946ef;font-weight:bold"
    if p.token_mask[token_idx]:
        style = f"{style};background:#86efac;border-radius:2px;padding:0 1px"
    return style


def _render_sample_tokens_html(
    p: PreparedInput, tokenizer, *, max_tokens: int = 200
) -> str:
    """Build an HTML token sequence using the persona-vectors preview layout."""
    special_ids = set(tokenizer.all_special_ids)
    seq_len = int(p.input_ids.shape[0])
    edge = max(8, max_tokens // 4)

    prefix_end = min(p.spans.template.token_start + max_tokens, seq_len)
    tail_start = min(max(prefix_end, p.spans.template.token_end - edge), seq_len)
    answer_end = min(seq_len, p.spans.response.token_end + edge)

    indices: list[int | None] = list(range(0, prefix_end))
    if prefix_end < tail_start:
        indices.append(None)
    indices.extend(range(tail_start, answer_end))
    if answer_end < seq_len:
        indices.append(None)

    spans: list[str] = []
    for idx in indices:
        if idx is None:
            spans.append('<span style="color:#9ca3af"> … </span>')
            continue
        char_start, char_end = p.offset_mapping[idx]
        raw = p.prompt_text[char_start:char_end]
        if not raw:
            raw = tokenizer.convert_ids_to_tokens([int(p.input_ids[idx])])[0]
        escaped = html.escape(raw)
        spans.append(
            f'<span style="{_token_style_for_index(p, idx, special_ids)}">{escaped}</span>'
        )

    return (
        '<pre style="white-space:pre-wrap;font-size:0.82em;line-height:1.5;'
        "background:#0e1117;padding:8px 10px;border-radius:6px;"
        'border:1px solid #333;margin:0">'
        f"{''.join(spans)}</pre>"
    )


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
    if "baseline" in selected_variants:
        st.caption(
            "Baseline uses the persona-less Assistant prompt and is saved once as an Assistant reference. It uses the first selected persona's QA set for extraction."
        )

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

    if not getattr(dataset, "supports_qa", True):
        st.info("This dataset is persona-only for now. Use Chat to browse personas.")
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

    with st.expander("Advanced", expanded=False):
        st.caption("Filters")

        col1, col2, _ = st.columns([2, 2, 1])
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

        st.caption("Extraction settings")
        last_strategy = st.session_state.get(
            _LAST_MASK_STRATEGY_KEY, MaskStrategy.ANSWER_MEAN.value
        )
        strategy_options = list(MaskStrategy)
        strategy_index = next(
            (i for i, s in enumerate(strategy_options) if s.value == last_strategy),
            0,
        )
        mask_strategy = st.selectbox(
            "Mask strategy",
            options=strategy_options,
            index=strategy_index,
            format_func=lambda s: s.value.replace("_", " ").title(),
            key=_extract_widget_key(
                model_name, remote, dataset_source, "mask_strategy"
            ),
            help="Which tokens contribute to the averaged hidden state.",
        )
        st.session_state[_LAST_MASK_STRATEGY_KEY] = mask_strategy.value

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

    run_col, preview_col, _spacer = st.columns([1, 1, 4], gap="small")
    with run_col:
        run_clicked = st.button(
            "Run extraction", type="primary", use_container_width=True
        )
    with preview_col:
        preview_clicked = st.button("Preview tokens", use_container_width=True)

    if preview_clicked:
        with st.spinner("Loading tokenizer..."):
            model = cached_model(model_name=model_name, remote=remote)
        st.markdown(_TOKEN_LEGEND, unsafe_allow_html=True)
        for persona, qa_pairs, variant in _build_run_plan(selected_variants, runs):
            system_prompt = system_prompt_for_variant(persona, variant)
            prepared = prepare_inputs_for_strategy(
                tokenizer=model.tokenizer,
                system_prompt=system_prompt,
                qa_pairs=qa_pairs[:max_questions],
                mask_strategy=mask_strategy,
            )
            display_name = (
                BASELINE_PERSONA_NAME if variant == "baseline" else persona.name
            )
            st.caption(f"{display_name} · {prompt_variant_label(variant)}")
            shown = prepared[:_MAX_PREVIEW_SAMPLES]
            for i, p in enumerate(shown):
                question = (
                    p.question if len(p.question) <= 60 else p.question[:57] + "..."
                )
                seq_len = int(p.input_ids.shape[0])
                masked = int(p.token_mask.sum())
                label = f"sample {i} — {question}  (len={seq_len}, masked={masked})"
                with st.expander(label):
                    st.markdown(
                        _render_sample_tokens_html(p, model.tokenizer),
                        unsafe_allow_html=True,
                    )
            if len(prepared) > _MAX_PREVIEW_SAMPLES:
                remaining = len(prepared) - _MAX_PREVIEW_SAMPLES
                st.caption(f"… and {remaining} more sample(s) not shown.")
        return

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
        run_plan = _build_run_plan(selected_variants, runs)
        total_steps = len(run_plan)
        step = 0
        results = []

        for persona, qa_pairs, variant in run_plan:
            display_name = (
                BASELINE_PERSONA_NAME if variant == "baseline" else persona.name
            )
            progress.progress(
                step / total_steps if total_steps else 1.0,
                text=f"{display_name} · {prompt_variant_label(variant)} ({step + 1}/{total_steps})",
            )
            variant_results = run_extraction(
                model=model,
                model_name=model_name,
                persona=persona,
                qa_pairs=qa_pairs[:max_questions],
                variants=(variant,),
                mask_strategy=mask_strategy,
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
