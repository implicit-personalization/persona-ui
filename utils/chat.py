import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
from nnterp import StandardizedTransformer
from persona_data.prompts import format_roleplay_prompt, normalize_messages
from persona_data.synth_persona import PersonaData

logger = logging.getLogger(__name__)
SystemPromptMode = Literal["empty", "templated", "biography", "custom"]


@dataclass
class ChatReply:
    text: str
    prompt_tokens: int
    output_tokens: int
    past_key_values: object | None


def resolve_system_prompt(
    persona: PersonaData | None,
    mode: SystemPromptMode,
) -> str:
    """Resolve the active system prompt for chat.

    Args:
        persona: Selected persona, if any.
        mode: Prompt mode selected in the UI.

    Returns:
        The rendered system prompt string.
    """

    if persona is None:
        return ""

    if mode == "empty":
        return ""
    if mode == "templated":
        return format_roleplay_prompt(persona.templated_view, mode="conversational")
    if mode == "biography":
        return format_roleplay_prompt(persona.biography_view, mode="conversational")
    if mode == "custom":
        return format_roleplay_prompt(mode="conversational")
    raise ValueError(f"Unsupported system prompt mode: {mode}")


def _format_plain_messages(
    messages: list[dict[str, str]], add_generation_prompt: bool
) -> str:
    """Format messages as plain ``Role: content`` text, used as a last-resort fallback."""
    lines: list[str] = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            if content:
                lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.title()}: {content}")

    if add_generation_prompt and (not lines or not lines[-1].startswith("Assistant:")):
        lines.append("Assistant:")

    return "\n\n".join(lines)


def _format_generation_prompt(
    messages: list[dict[str, str]], tokenizer: object
) -> tuple[str, int]:
    """Render messages into a single prompt string and count prompt tokens.

    Tries the tokenizer's chat template first, falls back to normalized messages,
    then to a plain-text format if both template attempts fail.
    """
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        logger.debug(
            "Chat template failed on raw messages, trying normalized", exc_info=True
        )
        messages = normalize_messages(messages)

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            logger.debug(
                "Chat template failed on normalized messages, falling back to plain format",
                exc_info=True,
            )
            prompt = _format_plain_messages(
                messages,
                add_generation_prompt=True,
            )

    prompt_token_count = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    return prompt, prompt_token_count


@contextmanager
def _seeded_rng(seed: int | None):
    """Context manager that forks the RNG state and sets a deterministic seed."""
    if seed is None:
        yield
        return

    cuda_ctx = torch.random.fork_rng(devices=range(torch.cuda.device_count()))
    mps_ctx = (
        torch.random.fork_rng(devices=range(1), device_type="mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else nullcontext()
    )

    with cuda_ctx, mps_ctx:
        torch.manual_seed(seed)
        yield


def generate_chat_reply(
    model: StandardizedTransformer,
    messages: list[dict[str, str]],
    remote: bool,
    past_key_values: object | None = None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
) -> ChatReply:
    """Generate one assistant reply from a full chat history.

    The helper uses ``model.generate`` so it works with both local and NDIF-backed
    nnsight models. The full conversation is re-rendered each turn and the cache from
    the previous turn is reused when available.

    Args:
        model: Loaded standardized nnterp model.
        messages: Full chat history, including any system prompt as the first message.
        remote: Whether to execute the generation on NDIF.
        past_key_values: Cache returned by the previous generation step.
        max_new_tokens: Maximum number of assistant tokens to generate.
        do_sample: Whether to sample from the model distribution.
        temperature: Sampling temperature, used only when sampling is enabled.
        top_p: Nucleus sampling threshold, used only when sampling is enabled.
        top_k: Top-k cutoff, used only when sampling is enabled.
        repetition_penalty: Repetition penalty applied during decoding.
        seed: Optional local RNG seed for sampled generation.

    Returns:
        ChatReply with generated text and the updated cache.
    """

    tokenizer = model.tokenizer
    prompt, prompt_token_count = _format_generation_prompt(messages, tokenizer)

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if not remote:
        # No need for this in remote which also slows down download drastically
        generation_kwargs["return_dict_in_generate"] = True
    if do_sample:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        generation_kwargs["top_k"] = top_k
    if repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = repetition_penalty
    if past_key_values is not None and not remote:
        generation_kwargs["past_key_values"] = past_key_values

    # `remote` is captured by nnsight's RemoteableMixin.trace() and is NOT
    # forwarded to the underlying model's generate
    with _seeded_rng(seed if do_sample and not remote else None):
        with model.generate(prompt, remote=remote, **generation_kwargs) as tracer:
            generated = tracer.result.save()

    if hasattr(generated, "value") and getattr(generated, "value") is not None:
        generated = generated.value

    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    if not isinstance(sequences, torch.Tensor):
        raise TypeError("Generated sequences must be a tensor")

    generated_ids = sequences[0, prompt_token_count:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    output_tokens = int(sequences.shape[1] - prompt_token_count)

    return ChatReply(
        text=text,
        prompt_tokens=prompt_token_count,
        output_tokens=max(0, output_tokens),
        past_key_values=(
            getattr(generated, "past_key_values", None) if not remote else None
        ),
    )
