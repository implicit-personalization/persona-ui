import html
from dataclasses import dataclass

import streamlit as st
from catppuccin import PALETTE
from persona_data.prompts import format_prompt
from persona_data.synth_persona import BASELINE_PERSONA_ID, PersonaData, QAPair
from persona_vectors.artifacts import SUPPORTED_VARIANTS
from persona_vectors.extraction import (
    MaskStrategy,
    prepare_inputs_for_strategy,
    run_extraction,
)
from persona_vectors.preview import TokenSegment, preview_token_segments

from utils.controls import render_mask_strategy_select
from utils.datasets import (
    load_dataset,
    load_persona_list_from_dataset,
    warm_qa_in_background,
)
from utils.helpers import (
    NDIF_STATUS_ICONS,
    persona_label,
    prompt_variant_label,
    session_key,
    widget_key,
)
from utils.runtime import cached_model
from utils.theme import active_base

_LAST_VARIANTS_KEY = "extract:last_variants"
_LAST_BASELINE_KEY = "extract:last_include_baseline"
_LAST_PERSONA_IDS_KEY = "extract:last_persona_ids"
_LAST_MAX_QUESTIONS_KEY = "extract:last_max_questions"
_LAST_MASK_STRATEGY_KEY = "extract:last_mask_strategy"

_PERSONAS_FILE_KEY = session_key("extract", "personas_file")
_QA_FILE_KEY = session_key("extract", "qa_file")

_DEFAULT_MAX_QUESTIONS = 50


@dataclass(frozen=True)
class ExtractSettings:
    mask_strategy: MaskStrategy
    max_questions: int


def _build_run_plan(
    selected_variants: list[str],
    runs: list[tuple[PersonaData, list[QAPair]]],
) -> list[tuple[PersonaData, list[QAPair], str]]:
    """Cartesian product of personas x variants."""
    return [(p, qa, v) for v in selected_variants for p, qa in runs]


def _row_label(persona: PersonaData, variant: str) -> str:
    return f"{persona.name} · {prompt_variant_label(variant)}"


def _extract_widget_key(
    model_name: str, remote: bool, dataset_source: str, suffix: str
) -> str:
    return widget_key("extract", str(remote), model_name, dataset_source, suffix)


def _render_local_dataset_upload(dataset_source: str) -> None:
    if dataset_source != "Local JSONL upload":
        return
    with st.expander("Local dataset upload", expanded=True):
        st.file_uploader(
            "personas.jsonl",
            type=["jsonl"],
            key=_PERSONAS_FILE_KEY,
            help="Expected fields: id, persona, templated_view, biography_view",
        )
        st.file_uploader(
            "qa.jsonl",
            type=["jsonl"],
            key=_QA_FILE_KEY,
            help="Expected fields: id, qid, type, item_type, scope, question, answer",
        )


def _render_variant_controls(
    *,
    model_name: str,
    remote: bool,
    dataset_source: str,
) -> tuple[list[str], bool] | None:
    default_variants = st.session_state.get(
        _LAST_VARIANTS_KEY, list(SUPPORTED_VARIANTS)
    )
    selected_variants = st.multiselect(
        "Persona variants",
        options=SUPPORTED_VARIANTS,
        default=[v for v in default_variants if v in SUPPORTED_VARIANTS]
        or list(SUPPORTED_VARIANTS),
        format_func=prompt_variant_label,
        key=_extract_widget_key(model_name, remote, dataset_source, "persona_variants"),
        help="Extract these variants for each selected persona.",
    )
    include_baseline = st.checkbox(
        "Extract Assistant baseline",
        value=st.session_state.get(_LAST_BASELINE_KEY, False),
        key=_extract_widget_key(model_name, remote, dataset_source, "baseline"),
        help="Also extract the Assistant baseline persona using the first persona's QA set.",
    )
    st.session_state[_LAST_VARIANTS_KEY] = selected_variants
    st.session_state[_LAST_BASELINE_KEY] = include_baseline
    if not selected_variants:
        st.info("Select at least one persona variant.")
        return None
    return selected_variants, include_baseline


def _load_qa_dataset_personas(
    dataset_source: str,
) -> tuple[object, list[PersonaData]] | None:
    try:
        dataset, dataset_status = load_dataset(
            dataset_source,
            personas_file=st.session_state.get(_PERSONAS_FILE_KEY),
            qa_file=st.session_state.get(_QA_FILE_KEY),
        )
        personas = load_persona_list_from_dataset(dataset)
        st.caption(dataset_status)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.info(
            "Upload both JSONL files or switch to the built-in SynthPersona source."
        )
        return None

    if not getattr(dataset, "supports_qa", True):
        st.info("This dataset is persona-only for now. Use Chat to browse personas.")
        return None

    if not personas:
        st.warning("No personas found in the selected dataset.")
        st.info(
            "Try another dataset source or check that the personas file is not empty."
        )
        return None

    # Extract is the only tab that needs QA; warm it now so the parse overlaps
    # with the user configuring the run instead of blocking the first extract.
    warm_qa_in_background(dataset)
    return dataset, personas


def _render_persona_select(
    *,
    personas: list[PersonaData],
    model_name: str,
    remote: bool,
    dataset_source: str,
) -> list[PersonaData] | None:
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
        return None
    return selected_personas


_MAX_PREVIEW_SAMPLES = 3


def _preview_palette():
    flavor = PALETTE.latte if active_base() == "light" else PALETTE.mocha
    return flavor.colors


def _render_token_legend_html() -> str:
    c = _preview_palette()
    return (
        '<div style="display:flex;gap:12px;flex-wrap:wrap;font-size:0.8em;margin-bottom:8px">'
        f'<span style="background:{c.green.hex};color:{c.base.hex};'
        'padding:1px 6px;border-radius:3px">masked</span>'
        f'<span style="color:{c.yellow.hex};padding:1px 6px">question</span>'
        f'<span style="color:{c.sky.hex};padding:1px 6px">response</span>'
        f'<span style="color:{c.mauve.hex};font-weight:bold;padding:1px 6px">special</span>'
        f'<span style="color:{c.subtext1.hex};padding:1px 6px">template</span>'
        "</div>"
    )


def _token_style(segment: TokenSegment) -> str:
    c = _preview_palette()
    style = {
        "response": f"color:{c.sky.hex}",
        "question": f"color:{c.yellow.hex}",
    }.get(segment.role, f"color:{c.subtext1.hex}")

    if segment.is_special:
        style = f"color:{c.mauve.hex};font-weight:bold"
    if segment.is_masked:
        style = (
            f"{style};background:{c.green.hex};color:{c.base.hex};"
            "border-radius:2px;padding:0 1px"
        )
    return style


def _render_sample_tokens_html(p, tokenizer, *, max_tokens: int = 200) -> str:
    spans: list[str] = []
    for segment in preview_token_segments(p, tokenizer, max_tokens=max_tokens):
        spans.append(
            f'<span style="{_token_style(segment)}">{html.escape(segment.text)}</span>'
        )

    return (
        '<pre style="white-space:pre-wrap;font-size:0.82em;line-height:1.5;'
        "background:var(--secondary-background-color,rgba(127,127,127,0.08));"
        "padding:8px 10px;border-radius:6px;"
        'border:1px solid rgba(127,127,127,0.25);margin:0">'
        f"{''.join(spans)}</pre>"
    )


def _render_mask_strategy_select(
    *,
    model_name: str,
    remote: bool,
    dataset_source: str,
) -> MaskStrategy:
    return render_mask_strategy_select(
        key=_extract_widget_key(model_name, remote, dataset_source, "mask_strategy"),
        last_key=_LAST_MASK_STRATEGY_KEY,
        help_text="Which tokens contribute to the averaged hidden state.",
    )


def _collect_runs(
    *,
    dataset,
    selected_personas: list[PersonaData],
) -> list[tuple[PersonaData, list[QAPair]]] | None:
    runs, skipped = [], []
    for persona in selected_personas:
        if persona.id == BASELINE_PERSONA_ID:
            qa = list(
                dataset.get_qa(BASELINE_PERSONA_ID, item_type="mcq", scope="shared")
            )
        elif hasattr(dataset, "train_test_split"):
            qa, _ = dataset.train_test_split(persona.id)
        else:
            qa = list(dataset.get_qa(persona.id))
        if qa:
            runs.append((persona, qa))
        else:
            skipped.append(persona)
    if skipped:
        names = ", ".join(p.name for p in skipped)
        st.warning(f"No train QA pairs found for: {names}. They will be skipped.")
    if not runs:
        st.info("No personas have matching QA pairs.")
        return None
    return runs


def _render_max_questions(
    *,
    model_name: str,
    remote: bool,
    dataset_source: str,
    runs: list[tuple[PersonaData, list[QAPair]]],
) -> int:
    max_q = min(len(qa_pairs) for _, qa_pairs in runs)
    default = min(_DEFAULT_MAX_QUESTIONS, max_q)
    max_questions = st.slider(
        "Max questions (train split)",
        min_value=1,
        max_value=max_q,
        value=min(
            max(st.session_state.get(_LAST_MAX_QUESTIONS_KEY, default), 1), max_q
        ),
        key=_extract_widget_key(model_name, remote, dataset_source, "max_questions"),
    )
    st.session_state[_LAST_MAX_QUESTIONS_KEY] = max_questions
    return max_questions


def _render_extract_actions() -> tuple[bool, bool]:
    run_col, preview_col, _spacer = st.columns([1, 1, 4], gap="small")
    with run_col:
        run_clicked = st.button(
            "Run extraction",
            type="primary",
            width="stretch",
        )
    with preview_col:
        preview_clicked = st.button("Preview tokens", width="stretch")
    return run_clicked, preview_clicked


def _render_token_preview(
    *,
    model_name: str,
    run_plan: list[tuple[PersonaData, list[QAPair], str]],
    settings: ExtractSettings,
) -> None:
    with st.spinner("Loading tokenizer..."):
        model = cached_model(model_name=model_name)
    st.markdown(_render_token_legend_html(), unsafe_allow_html=True)
    for persona, qa_pairs, variant in run_plan:
        system_prompt = format_prompt(persona, variant)  # type: ignore[arg-type]
        prepared = prepare_inputs_for_strategy(
            tokenizer=model.tokenizer,
            system_prompt=system_prompt,
            qa_pairs=qa_pairs[: settings.max_questions],
            mask_strategy=settings.mask_strategy,
        )
        st.caption(_row_label(persona, variant))
        for i, p in enumerate(prepared[:_MAX_PREVIEW_SAMPLES]):
            question = p.question if len(p.question) <= 60 else p.question[:57] + "..."
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


def _run_extraction_plan(
    *,
    remote: bool,
    model_name: str,
    run_plan: list[tuple[PersonaData, list[QAPair], str]],
    settings: ExtractSettings,
) -> None:
    status_box = st.empty()
    status_box.info("Extraction in progress...")
    progress = st.progress(0, text="Preparing extraction...")
    ndif_status_box = st.empty()

    def _on_ndif_status(job_id: str, status_name: str, description: str) -> None:
        icon = NDIF_STATUS_ICONS.get(status_name, "•")
        ndif_status_box.caption(f"{icon} `{job_id}` **{status_name}** — {description}")

    with st.spinner("Loading model..."):
        model = cached_model(model_name=model_name)

    try:
        total_steps = len(run_plan)
        results = []
        for step, (persona, qa_pairs, variant) in enumerate(run_plan):
            progress.progress(
                step / total_steps if total_steps else 1.0,
                text=f"{_row_label(persona, variant)} ({step + 1}/{total_steps})",
            )
            results.extend(
                run_extraction(
                    model=model,
                    model_name=model_name,
                    qa_pairs=qa_pairs[: settings.max_questions],
                    variants=(variant,),
                    persona=persona,
                    mask_strategy=settings.mask_strategy,
                    remote=remote,
                    on_status=_on_ndif_status if remote else None,
                )
            )

        progress.progress(1.0, text="Extraction complete")
    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        return
    finally:
        progress.empty()
        ndif_status_box.empty()

    status_box.empty()
    st.success(f"Saved {len(results)} artifact set(s)")

    for result in results:
        st.markdown(
            f"- **{result.persona_name}** · {prompt_variant_label(result.variant)}: "
            f"{result.n_questions} questions"
        )


def render_extract_tab(remote: bool, model_name: str, dataset_source: str) -> None:
    """Render the extraction tab."""

    st.title("Extract")
    st.caption("Extract per-persona activation vectors from train QA pairs.")

    _render_local_dataset_upload(dataset_source)
    variant_choice = _render_variant_controls(
        model_name=model_name,
        remote=remote,
        dataset_source=dataset_source,
    )
    if variant_choice is None:
        return
    selected_variants, include_baseline = variant_choice

    loaded = _load_qa_dataset_personas(dataset_source)
    if loaded is None:
        return
    dataset, personas = loaded

    selected_personas = _render_persona_select(
        personas=personas,
        model_name=model_name,
        remote=remote,
        dataset_source=dataset_source,
    )
    if selected_personas is None:
        return

    personas_for_runs = list(selected_personas)
    baseline = getattr(dataset, "baseline", None)
    if include_baseline and baseline is not None:
        personas_for_runs.append(baseline)

    runs = _collect_runs(dataset=dataset, selected_personas=personas_for_runs)
    if runs is None:
        return

    max_questions = _render_max_questions(
        model_name=model_name,
        remote=remote,
        dataset_source=dataset_source,
        runs=runs,
    )
    with st.expander("Advanced", expanded=False):
        mask_strategy = _render_mask_strategy_select(
            model_name=model_name,
            remote=remote,
            dataset_source=dataset_source,
        )
    settings = ExtractSettings(
        mask_strategy=mask_strategy,
        max_questions=max_questions,
    )

    run_clicked, preview_clicked = _render_extract_actions()
    run_plan = _build_run_plan(selected_variants, runs)

    if preview_clicked:
        _render_token_preview(
            model_name=model_name,
            run_plan=run_plan,
            settings=settings,
        )
        return

    if not run_clicked:
        return

    _run_extraction_plan(
        remote=remote,
        model_name=model_name,
        run_plan=run_plan,
        settings=settings,
    )
