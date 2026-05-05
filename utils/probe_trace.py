from __future__ import annotations

import hashlib
from dataclasses import dataclass

import streamlit as st
import torch
from nnterp import StandardizedTransformer

from utils.chat import format_generation_prompt

_TRACE_CACHE_KEY = "probe:trace_cache"
_MAX_CACHED_TRACES = 3


@dataclass(frozen=True)
class TokenVector:
    vector: torch.Tensor
    mode: str
    token_index: int
    hidden_size: int


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

    input_ids = _resolve_saved_tensor(saved_ids)
    activations = _resolve_saved_tensor(saved_acts)
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
    )
    _store_cached_trace(cache_key, trace)
    return trace


def vectorize_token(
    trace: ConversationTrace,
    *,
    token_index: int,
) -> TokenVector:
    if token_index < 0 or token_index >= trace.n_tokens:
        raise ValueError(f"Invalid token {token_index} for {trace.n_tokens} tokens")

    return TokenVector(
        vector=trace.activations[token_index].detach().cpu(),
        mode="single_token",
        token_index=token_index,
        hidden_size=trace.hidden_size,
    )


def decode_token(tokenizer: object, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def _select_accessor(model: StandardizedTransformer, location: str):
    normalized = location.lower()
    if normalized in {"pre_reasoning", "pre", "input", "layers_input"}:
        return model.layers_input
    if normalized in {"post_reasoning", "post", "output", "layers_output"}:
        return model.layers_output
    raise ValueError(f"Unsupported trace location: {location!r}")


def _resolve_saved_tensor(value) -> torch.Tensor:
    resolved = value.value if getattr(value, "value", None) is not None else value
    if not isinstance(resolved, torch.Tensor):
        raise TypeError(f"Trace result did not resolve to a tensor: {type(resolved)!r}")
    return resolved.detach().cpu()


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
