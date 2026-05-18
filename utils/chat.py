from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from persona_data.prompts import format_messages, format_prompt, normalize_messages

if TYPE_CHECKING:
    import torch
    from nnterp import StandardizedTransformer
    from persona_data.synth_persona import PersonaData

logger = logging.getLogger(__name__)
SystemPromptMode = Literal["empty", "templated", "biography", "custom"]


@dataclass
class ChatReply:
    text: str
    generated_ids: Any | None = None


def build_chat_messages(
    system_prompt: str | None,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Prepend the active system prompt to a chat history when present."""

    return (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    ) + messages


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

    if persona is None or mode == "empty":
        return ""
    if mode == "custom":
        return format_prompt(persona, "templated", mode="conversational")
    if mode in ("templated", "biography"):
        return format_prompt(persona, mode, mode="conversational")
    raise ValueError(f"Unsupported system prompt mode: {mode}")


def _format_plain_messages(
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    """Format messages as plain text when no tokenizer chat template is usable."""

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


def format_generation_prompt(
    messages: list[dict[str, str]],
    tokenizer: object,
    *,
    add_generation_prompt: bool = True,
) -> tuple[str, int]:
    """Render chat messages and count prompt tokens.

    ``persona-data`` owns the standard chat-template path. The fallback below is
    only for tokenizers with broken or missing chat templates.
    """

    try:
        prompt, prompt_token_count = format_messages(
            messages,
            tokenizer,
            add_generation_prompt=add_generation_prompt,
        )
        return prompt, prompt_token_count
    except Exception:
        logger.debug("persona-data format_messages failed", exc_info=True)

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        logger.debug("Chat template failed on raw messages", exc_info=True)
        normalized = normalize_messages(messages)
        try:
            prompt = tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            logger.debug("Chat template fallback failed", exc_info=True)
            prompt = _format_plain_messages(
                normalized,
                add_generation_prompt=add_generation_prompt,
            )

    prompt_token_count = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    return prompt, prompt_token_count


def resolve_saved_tensor(value: object) -> torch.Tensor:
    """Resolve an nnsight ``.save()`` proxy (or raw tensor) to a CPU tensor."""
    import torch

    resolved = value.value if getattr(value, "value", None) is not None else value
    if not isinstance(resolved, torch.Tensor):
        raise TypeError(f"Trace result did not resolve to a tensor: {type(resolved)!r}")
    return resolved.detach().cpu()


def decode_token(tokenizer: object, token_id: int) -> str:
    """Decode a single token id, falling back when ``clean_up_tokenization_spaces`` is unsupported."""
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


@contextmanager
def _seeded_rng(seed: int | None):
    """Context manager that forks the RNG state and sets a deterministic seed."""
    if seed is None:
        yield
        return

    import torch

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
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    on_status: Callable[[str, str, str], None] | None = None,
    ndif_api_key: str | None = None,
) -> ChatReply:
    """Generate one assistant reply from a full chat history.

    The helper uses ``model.generate`` so it works with both local and NDIF-backed
    nnsight models. The full conversation is re-rendered each turn.

    Args:
        model: Loaded standardized nnterp model.
        messages: Full chat history, including any system prompt as the first message.
        remote: Whether to execute the generation on NDIF.
        max_new_tokens: Maximum number of assistant tokens to generate.
        do_sample: Whether to sample from the model distribution.
        temperature: Sampling temperature, used only when sampling is enabled.
        top_p: Nucleus sampling threshold, used only when sampling is enabled.
        top_k: Top-k cutoff, used only when sampling is enabled.
        repetition_penalty: Repetition penalty applied during decoding.
        seed: Optional local RNG seed for sampled generation.

    Returns:
        ChatReply with generated text and token ids.
    """

    import torch

    tokenizer = model.tokenizer
    prompt, prompt_token_count = format_generation_prompt(messages, tokenizer)

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
    # `remote` is captured by nnsight's RemoteableMixin.trace() and is NOT
    # forwarded to the underlying model's generate
    if remote:
        from utils.runtime import remote_backend

        backend = remote_backend(model, ndif_api_key, on_status=on_status)
    else:
        backend = None

    with (
        _seeded_rng(seed if do_sample and not remote else None),
        model.generate(
            prompt,
            remote=remote,
            backend=backend,
            **generation_kwargs,
        ) as tracer,
    ):
        generated = tracer.result.save()

    if getattr(generated, "value", None) is not None:
        generated = generated.value

    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    if not isinstance(sequences, torch.Tensor):
        raise TypeError("Generated sequences must be a tensor")

    generated_ids = sequences[0, prompt_token_count:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return ChatReply(
        text=text,
        generated_ids=generated_ids.detach().cpu(),
    )
