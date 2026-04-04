import gc
import logging
from dataclasses import dataclass

import torch
from nnterp import StandardizedTransformer

logger = logging.getLogger(__name__)

from persona_data.environment import get_artifacts_dir
from persona_data.synth_persona import PersonaData, QAPair
from persona_vectors.activation_io import save_per_question_vectors
from persona_vectors.activations import extract_activations
from persona_data.prompts import (
    format_biography_prompt,
    format_messages,
    format_templated_prompt,
)


@dataclass
class VariantExtractionResult:
    variant: str
    output_dir: str
    n_questions: int
    n_layers: int
    d_model: int
    persona_name: str = ""


def _prepare_inputs(
    tokenizer: object,
    system_prompt: str,
    qa_pairs: list[QAPair],
) -> tuple[list[str], list[torch.Tensor], list[str]]:
    """Format QA pairs into tokenized prompts with answer-token masks.

    Args:
        tokenizer: HuggingFace-compatible tokenizer from the model.
        system_prompt: System prompt to prepend to each conversation.
        qa_pairs: List of question-answer pairs to format.

    Returns:
        A tuple of (full_texts, token_masks, questions) where full_texts are
        the rendered prompt strings, token_masks are boolean tensors marking
        answer tokens, and questions are the raw question strings.
    """
    full_texts: list[str] = []
    token_masks: list[torch.Tensor] = []
    questions: list[str] = []

    for qa in qa_pairs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": qa.question},
            {"role": "assistant", "content": qa.answer},
        ]
        full_prompt, answer_start = format_messages(messages, tokenizer)
        seq_len = tokenizer(full_prompt, return_tensors="pt").input_ids.shape[1]

        full_texts.append(full_prompt)
        token_masks.append(torch.arange(seq_len) >= answer_start)
        questions.append(qa.question)

    return full_texts, token_masks, questions


def run_extraction(
    model: StandardizedTransformer,
    model_name: str,
    persona: PersonaData,
    qa_pairs: list[QAPair],
    variants: list[str],
    remote: bool,
) -> list[VariantExtractionResult]:
    """Run activation extraction and save outputs for selected variants.

    Args:
        model: Loaded standardized nnterp model.
        model_name: HuggingFace model identifier used for artifact paths.
        persona: The persona whose QA pairs are being extracted.
        qa_pairs: Question-answer pairs to run extraction on.
        variants: Prompt variants to extract (e.g. ``"templated"``, ``"biography"``).
        remote: Whether to execute on NDIF.

    Returns:
        A list of extraction results, one per variant.

    Raises:
        ValueError: If ``qa_pairs`` is empty or an unsupported variant is given.
    """
    if not qa_pairs:
        raise ValueError("No QA pairs selected for extraction")

    tokenizer = model.tokenizer
    activations_dir = get_artifacts_dir() / "activations"

    system_prompt_by_variant = {
        "templated": format_templated_prompt(persona.templated_prompt),
        "biography": format_biography_prompt(persona.biography_md),
    }

    results: list[VariantExtractionResult] = []

    for variant in variants:
        if variant not in system_prompt_by_variant:
            raise ValueError(f"Unsupported variant: {variant}")

        full_texts, token_masks, questions = _prepare_inputs(
            tokenizer=tokenizer,
            system_prompt=system_prompt_by_variant[variant],
            qa_pairs=qa_pairs,
        )

        per_question_vectors = extract_activations(
            model=model,
            full_texts=full_texts,
            token_masks=token_masks,
            remote=remote,
        )

        artifact_dir = save_per_question_vectors(
            root_dir=activations_dir,
            model_name=model_name,
            prompt_variant=variant,
            persona_id=persona.id,
            persona_name=persona.name,
            per_question_vectors=per_question_vectors,
            questions=questions,
        )

        results.append(
            VariantExtractionResult(
                variant=variant,
                output_dir=str(artifact_dir),
                n_questions=per_question_vectors.shape[0],
                n_layers=per_question_vectors.shape[1],
                d_model=per_question_vectors.shape[2],
                persona_name=persona.name,
            )
        )

        # Free activation tensors between variants to keep memory bounded.
        del per_question_vectors, full_texts, token_masks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    return results
