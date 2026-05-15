from __future__ import annotations

import hashlib
from dataclasses import dataclass

import streamlit as st
import torch
from nnterp import StandardizedTransformer
from persona_data.prompts import normalize_messages, supports_system_role

from utils.chat import decode_token, format_generation_prompt, resolve_saved_tensor

_TRACE_CACHE_KEY = "probe:trace_cache"
_MAX_CACHED_TRACES = 3


@dataclass(frozen=True)
class ConversationTrace:
    cache_key: str
    model_name: str
    remote: bool
    prompt_text: str
    prompt_hash: str
    layer: int
    location: str
    input_ids: torch.Tensor
    activations: torch.Tensor
    tokens: list[str]
    # One (start, end_exclusive) per assistant message in order. Empty list if
    # the tokenizer's chat template can't mark assistant tokens.
    assistant_spans: list[tuple[int, int]]
    # Per-position mask; True for tokenizer special ids that we don't want to
    # paint in the overlay (role markers, BOS/EOS, etc.).
    is_special: torch.Tensor

    @property
    def hidden_size(self) -> int:
        return int(self.activations.shape[-1])

    @property
    def n_tokens(self) -> int:
        return int(self.input_ids.shape[0])


def trace_conversation(
    *,
    model: StandardizedTransformer,
    model_name: str,
    messages: list[dict[str, str]],
    layer: int,
    location: str,
    remote: bool,
) -> ConversationTrace:
    prompt_text, _ = format_generation_prompt(
        messages,
        model.tokenizer,
        add_generation_prompt=False,
    )
    assistant_mask_seq = _compute_assistant_mask(model.tokenizer, messages)
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    cache_key = _trace_cache_key(
        model_name=model_name,
        remote=remote,
        prompt_hash=prompt_hash,
        layer=layer,
        location=location,
    )
    cached = _get_cached_trace(cache_key)
    if cached is not None:
        return cached

    accessor = _select_accessor(model, location)
    with torch.no_grad(), model.trace(prompt_text, remote=remote):
        saved_ids = model.input_ids[0].detach().cpu().save()
        saved_acts = accessor[layer][0].detach().float().cpu().save()

    input_ids = resolve_saved_tensor(saved_ids)
    activations = resolve_saved_tensor(saved_acts)
    if input_ids.ndim != 1:
        raise ValueError(
            f"Expected traced input ids to be [seq], got {tuple(input_ids.shape)}"
        )
    if activations.ndim != 2:
        raise ValueError(
            f"Expected traced activations to be [seq, hidden], got {tuple(activations.shape)}"
        )
    if int(input_ids.shape[0]) != int(activations.shape[0]):
        raise ValueError(
            "Trace produced a different number of token ids and activation rows: "
            f"{tuple(input_ids.shape)} vs {tuple(activations.shape)}"
        )

    n_tokens = int(input_ids.shape[0])
    assistant_spans = _clip_spans(
        _assistant_spans_from_offsets(
            model.tokenizer, prompt_text, messages, n_tokens
        ),
        n_tokens,
    )
    if not assistant_spans and assistant_mask_seq is not None:
        assistant_spans = _assistant_spans(assistant_mask_seq, n_tokens)
    if not assistant_spans:
        prefix_spans = _assistant_spans_from_prefixes(model.tokenizer, messages)
        assistant_spans = _clip_spans(prefix_spans or [], n_tokens)
    is_special = _special_token_mask(model.tokenizer, input_ids)

    trace = ConversationTrace(
        cache_key=cache_key,
        model_name=model_name,
        remote=remote,
        prompt_text=prompt_text,
        prompt_hash=prompt_hash,
        layer=layer,
        location=location,
        input_ids=input_ids,
        activations=activations,
        tokens=[
            decode_token(model.tokenizer, int(token_id))
            for token_id in input_ids.tolist()
        ],
        assistant_spans=assistant_spans,
        is_special=is_special,
    )
    _store_cached_trace(cache_key, trace)
    return trace


def _select_accessor(model: StandardizedTransformer, location: str):
    normalized = location.lower()
    if normalized in {"pre_reasoning", "pre", "input", "layers_input"}:
        return model.layers_input
    if normalized in {"post_reasoning", "post", "output", "layers_output"}:
        return model.layers_output
    raise ValueError(f"Unsupported trace location: {location!r}")


def _trace_cache_key(
    *,
    model_name: str,
    remote: bool,
    prompt_hash: str,
    layer: int,
    location: str,
) -> str:
    return "::".join(
        (
            "probe-trace",
            model_name,
            str(remote),
            prompt_hash,
            str(layer),
            location,
        )
    )


def _get_cached_trace(cache_key: str) -> ConversationTrace | None:
    cache = st.session_state.get(_TRACE_CACHE_KEY)
    if not isinstance(cache, dict):
        return None
    trace = cache.get(cache_key)
    if not isinstance(trace, ConversationTrace):
        return None
    cache.pop(cache_key, None)
    cache[cache_key] = trace
    return trace


def _trace_cache() -> dict[str, ConversationTrace]:
    cache = st.session_state.get(_TRACE_CACHE_KEY)
    if isinstance(cache, dict):
        return cache
    cache = {}
    st.session_state[_TRACE_CACHE_KEY] = cache
    return cache


def _store_cached_trace(cache_key: str, trace: ConversationTrace) -> None:
    cache = _trace_cache()
    cache.pop(cache_key, None)
    cache[cache_key] = trace
    while len(cache) > _MAX_CACHED_TRACES:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)


def _compute_assistant_mask(
    tokenizer: object, messages: list[dict[str, str]]
) -> list[int] | None:
    """Return a per-token 0/1 mask marking assistant content, or None if unknown.

    Uses ``apply_chat_template(return_assistant_tokens_mask=True)`` when the
    tokenizer supports it (modern chat templates with ``{% generation %}``
    blocks). Returns ``None`` when the template doesn't mark assistant spans.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None or not messages:
        return None
    try:
        encoded = apply(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
    except Exception:
        return None
    mask = encoded.get("assistant_masks") if isinstance(encoded, dict) else None
    if not mask:
        return None
    if isinstance(mask, list) and mask and isinstance(mask[0], list):
        mask = mask[0]
    values = [int(value) for value in mask]
    if not any(values):
        return None
    return values


def _assistant_spans_from_offsets(
    tokenizer: object,
    prompt_text: str,
    messages: list[dict[str, str]],
    n_tokens: int,
) -> list[tuple[int, int]]:
    """Locate assistant bodies by char-offset, aligned to the traced sequence.

    The chat-template token arithmetic in ``_assistant_spans_from_prefixes``
    drifts whenever the template tokenizes differently than how ``model.trace``
    tokenizes the rendered prompt string (extra/missing BOS, trailing
    whitespace, etc.), which leaves the overlay unalignable. This instead finds
    each assistant message's text inside ``prompt_text`` and maps those char
    ranges to token indices via the fast tokenizer's offset mapping, retokenizing
    the exact string the trace ran on so the indices line up by construction.
    """
    if not getattr(tokenizer, "is_fast", False):
        return []
    contents = [
        message["content"]
        for message in messages
        if message.get("role") == "assistant" and message.get("content")
    ]
    if not contents:
        return []

    offsets = None
    for add_special_tokens in (True, False):
        try:
            encoded = tokenizer(
                prompt_text,
                return_offsets_mapping=True,
                add_special_tokens=add_special_tokens,
            )
        except Exception:
            return []
        mapping = encoded.get("offset_mapping")
        if mapping is not None and len(mapping) == n_tokens:
            offsets = mapping
            break
    if offsets is None:
        return []

    spans: list[tuple[int, int]] = []
    search_from = 0
    for content in contents:
        char_start = prompt_text.find(content, search_from)
        if char_start < 0:
            return []
        char_end = char_start + len(content)
        search_from = char_end
        tok_start: int | None = None
        tok_end: int | None = None
        for i, (start, end) in enumerate(offsets):
            if start == end:  # special tokens map to an empty (0, 0) range
                continue
            if tok_start is None and end > char_start:
                tok_start = i
            if start < char_end:
                tok_end = i + 1
        if tok_start is not None and tok_end is not None and tok_start < tok_end:
            spans.append((tok_start, tok_end))
    return spans


def _assistant_spans_from_prefixes(
    tokenizer: object, messages: list[dict[str, str]]
) -> list[tuple[int, int]] | None:
    """Fallback span detection when the chat template doesn't mark assistant tokens.

    For each assistant message at index ``i``, tokenize ``messages[:i]`` with
    ``add_generation_prompt=True`` to find where the body starts, and
    ``messages[:i+1]`` with ``add_generation_prompt=False`` to find where it
    ends. Mirrors the prefix arithmetic used by ``utils.contrast``.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None or not messages:
        return None
    if not supports_system_role(tokenizer):
        messages = normalize_messages(messages)
    spans: list[tuple[int, int]] = []
    try:
        for i, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prefix_ids = apply(
                messages[:i], tokenize=True, add_generation_prompt=True
            )
            through_ids = apply(
                messages[: i + 1], tokenize=True, add_generation_prompt=False
            )
            prefix_ids = _flatten_ids(prefix_ids)
            through_ids = _flatten_ids(through_ids)
            if prefix_ids is None or through_ids is None:
                return None
            start = len(prefix_ids)
            end = len(through_ids)
            if 0 <= start < end:
                spans.append((start, end))
    except Exception:
        return None
    return spans


def _flatten_ids(value: object) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if value and isinstance(value[0], list):
        value = value[0]
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError):
        return None


def _clip_spans(
    spans: list[tuple[int, int]], n_tokens: int
) -> list[tuple[int, int]]:
    clipped: list[tuple[int, int]] = []
    for start, end in spans:
        s = max(0, min(start, n_tokens))
        e = max(0, min(end, n_tokens))
        if s < e:
            clipped.append((s, e))
    return clipped


def _assistant_spans(
    assistant_mask_seq: list[int] | None, n_tokens: int
) -> list[tuple[int, int]]:
    """Convert a per-token mask into ``[(start, end_exclusive), ...]`` runs.

    Returns an empty list when the mask is missing or doesn't line up with the
    traced sequence, so the caller can show a clear "no overlay" state instead
    of painting the entire conversation.
    """
    if assistant_mask_seq is None or len(assistant_mask_seq) != n_tokens:
        return []
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(assistant_mask_seq):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, n_tokens))
    return spans


def _special_token_mask(tokenizer: object, input_ids: torch.Tensor) -> torch.Tensor:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if not special_ids:
        return torch.zeros(int(input_ids.shape[0]), dtype=torch.bool)
    return torch.tensor(
        [int(token_id) in special_ids for token_id in input_ids.tolist()],
        dtype=torch.bool,
    )
