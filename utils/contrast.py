"""
Contrastive token-level log-probability comparison for compare mode.

For a pair of responses generated under different persona contexts, each token
gets a weight:

    w(token) = log P(token | context_A) − log P(token | context_B)

Positive (red)  → token is more characteristic of persona A.
Negative (blue) → token is more characteristic of persona B.
Near-zero (gray) → both personas would emit this token with similar likelihood.
"""

import logging
from dataclasses import dataclass
from html import escape

import torch
from nnterp import StandardizedTransformer

from utils.chat import format_generation_prompt

logger = logging.getLogger(__name__)


@dataclass
class TokenContrast:
    tokens: list[str]
    weights: list[float]  # normalised to [-1, 1], used for coloring
    raw_diffs: list[float]  # unclipped log P(A) - log P(B) per token
    label_a: str
    label_b: str


# ── Weight computation ────────────────────────────────────────────────────────


def _normalise_diffs(diffs: torch.Tensor) -> list[float]:
    """
    Clip at the 95th percentile of |diff| and scale to [-1, 1] so a few
    high-magnitude tokens don't wash out everything else.
    """
    if len(diffs) < 2:
        return diffs.tolist()
    clip_val = max(torch.quantile(diffs.abs(), 0.95).item(), 0.3)
    return (diffs.float().clamp(-clip_val, clip_val) / clip_val).tolist()


def _decode_ids(tokenizer: object, ids: list[int]) -> str:
    """Decode token IDs, falling back when clean_up_tokenization_spaces is unsupported."""
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(ids, skip_special_tokens=False)


def _strip_special_ids(
    ids: torch.Tensor,
    tokenizer: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return display ids and a mask that excludes special tokens."""
    ids = ids.cpu()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if not special_ids or ids.numel() == 0:
        return ids, torch.ones(ids.shape[0], dtype=torch.bool)
    keep = torch.tensor(
        [tid.item() not in special_ids for tid in ids], dtype=torch.bool
    )
    return ids[keep], keep


def _prepare_trace_text(
    tokenizer: object,
    context_messages: list[dict[str, str]],
    response_ids: torch.Tensor,
) -> tuple[str, int, int]:
    """Build the trace text and return ``(full_text, n_ctx, n_resp)``."""
    context_prompt, _ = format_generation_prompt(context_messages, tokenizer)
    context_ids = tokenizer(context_prompt, return_tensors="pt").input_ids[0]
    response_text = _decode_ids(tokenizer, response_ids.tolist())
    full_text = context_prompt + response_text
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    expected_ids = torch.cat([context_ids, response_ids.cpu()])
    if full_ids.tolist() != expected_ids.tolist():
        logger.warning(
            "contrast trace text did not round-trip to the expected token ids "
            "(expected %d tokens, got %d); contrast scores may be slightly misaligned",
            len(expected_ids),
            len(full_ids),
        )
    n_ctx = len(context_ids)
    n_resp = len(response_ids)
    return full_text, n_ctx, n_resp


def _build_contrast(
    tokenizer: object,
    response_ids: torch.Tensor,
    lp_a: torch.Tensor,
    lp_b: torch.Tensor,
    label_a: str,
    label_b: str,
) -> TokenContrast:
    diffs = (lp_a - lp_b).cpu()
    display_ids, keep_mask = _strip_special_ids(response_ids, tokenizer)
    display_diffs = diffs[keep_mask]
    return TokenContrast(
        tokens=[_token_display(tokenizer, tid.item()) for tid in display_ids],
        weights=_normalise_diffs(display_diffs),
        raw_diffs=display_diffs.float().tolist(),
        label_a=label_a,
        label_b=label_b,
    )


def _token_display(tokenizer: object, token_id: int) -> str:
    """Render a single token id as normal decoded text."""
    return _decode_ids(tokenizer, [token_id])


# Each spec: (key, full_text, n_ctx, n_resp, target_ids).
PassSpec = tuple[str, str, int, int, torch.Tensor]


def _score_passes(
    model: StandardizedTransformer,
    specs: list[PassSpec],
    remote: bool,
) -> dict[str, torch.Tensor]:
    """
    Run one forward pass per spec and return reduced per-token logprobs.

    The log-softmax and target-pick happen *inside* the trace, so only the
    reduced ``[n_resp]`` logprob vector per pass is shipped back — not the full
    ``[1, seq, vocab]`` logits (which would be hundreds of MB per pass on NDIF).
    """

    def _score_pass(
        full_text: str,
        n_ctx: int,
        n_resp: int,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad(), model.trace(full_text, remote=remote):
            # logit at position i predicts token i+1, so response token j
            # (at full-text position n_ctx+j) uses logit at n_ctx+j-1.
            resp_logits = model.logits[0, n_ctx - 1 : n_ctx - 1 + n_resp].float()
            log_probs = torch.log_softmax(resp_logits, dim=-1)
            targets = target_ids.to(log_probs.device).view(-1, 1)
            picked = log_probs.gather(1, targets).view(-1)
            out = picked.detach().cpu().save()

        if getattr(out, "value", None) is not None:
            out = out.value
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"contrast score did not resolve to a tensor: {type(out)!r}"
            )
        return out.detach().cpu()

    return {
        key: _score_pass(full_text, n_ctx, n_resp, target_ids)
        for key, full_text, n_ctx, n_resp, target_ids in specs
    }


def _specs_for_response(
    tokenizer: object,
    response_ids: torch.Tensor,
    context_a: list[dict[str, str]],
    context_b: list[dict[str, str]],
    prefix: str,
) -> list[PassSpec]:
    """Build the (under_a, under_b) pass specs for a single response."""
    text_a, n_ctx_a, n_resp = _prepare_trace_text(tokenizer, context_a, response_ids)
    text_b, n_ctx_b, _ = _prepare_trace_text(tokenizer, context_b, response_ids)
    return [
        (f"{prefix}_under_a", text_a, n_ctx_a, n_resp, response_ids),
        (f"{prefix}_under_b", text_b, n_ctx_b, n_resp, response_ids),
    ]


def compute_contrast(
    model: StandardizedTransformer,
    context_a: list[dict[str, str]],
    context_b: list[dict[str, str]],
    response_ids: torch.Tensor,
    label_a: str,
    label_b: str,
    remote: bool = False,
) -> "TokenContrast | None":
    """Compute per-token contrast weights for a single response (2 forward passes)."""
    tokenizer = model.tokenizer
    if response_ids.numel() == 0:
        return None

    specs = _specs_for_response(tokenizer, response_ids, context_a, context_b, "r")
    out = _score_passes(model, specs, remote)
    return _build_contrast(
        tokenizer, response_ids, out["r_under_a"], out["r_under_b"], label_a, label_b
    )


def compute_contrast_pair(
    model: StandardizedTransformer,
    context_a: list[dict[str, str]],
    context_b: list[dict[str, str]],
    response_ids_a: torch.Tensor,
    response_ids_b: torch.Tensor,
    label_a: str,
    label_b: str,
    remote: bool = False,
) -> tuple["TokenContrast | None", "TokenContrast | None"]:
    """
    Compute contrast weights for both panel responses (up to 4 remote passes).
    """
    tokenizer = model.tokenizer
    if response_ids_a.numel() == 0 and response_ids_b.numel() == 0:
        return None, None

    specs: list[PassSpec] = []
    if response_ids_a.numel() > 0:
        specs += _specs_for_response(
            tokenizer, response_ids_a, context_a, context_b, "a"
        )
    if response_ids_b.numel() > 0:
        specs += _specs_for_response(
            tokenizer, response_ids_b, context_a, context_b, "b"
        )

    out = _score_passes(model, specs, remote)

    def _build(resp_ids: torch.Tensor, prefix: str) -> "TokenContrast | None":
        k_a, k_b = f"{prefix}_under_a", f"{prefix}_under_b"
        if resp_ids.numel() == 0 or k_a not in out or k_b not in out:
            return None
        return _build_contrast(
            tokenizer, resp_ids, out[k_a], out[k_b], label_a, label_b
        )

    return _build(response_ids_a, "a"), _build(response_ids_b, "b")


# ── HTML rendering ────────────────────────────────────────────────────────────


def _weight_to_bg(w: float) -> str:
    """Map a normalised weight in [-1, 1] to a CSS rgba background color."""
    w = max(-1.0, min(1.0, w))
    alpha = abs(w) * 0.5  # cap at 0.5 opacity so text stays readable
    if w > 0.05:
        return f"rgba(210,60,60,{alpha:.3f})"
    if w < -0.05:
        return f"rgba(50,110,210,{alpha:.3f})"
    return "rgba(0,0,0,0)"


_CONTRAST_CSS = (
    "<style>"
    ".contrast-tok{position:relative;border-radius:2px;padding:0 1px;"
    "cursor:default;white-space:pre;}"
    ".contrast-tok>.contrast-tip{display:none;position:absolute;bottom:100%;"
    "left:50%;transform:translateX(-50%);margin-bottom:4px;padding:2px 6px;"
    "border-radius:3px;background:#222;color:#eee;font-size:0.72em;"
    "font-family:ui-monospace,monospace;white-space:nowrap;pointer-events:none;"
    "z-index:10;box-shadow:0 2px 6px rgba(0,0,0,0.3);}"
    ".contrast-tok:hover>.contrast-tip{display:block;}"
    "</style>"
)


def render_contrast_html(result: TokenContrast) -> str:
    """
    Render each token with a colored background reflecting how A- or B-specific
    it is, with a hover tooltip showing the raw Δlog P, plus a legend.
    """
    spans: list[str] = []
    for token, weight, raw in zip(result.tokens, result.weights, result.raw_diffs):
        bg = _weight_to_bg(weight)
        tip = escape(f"Δlog P(A−B): {raw:+.3f}")
        text = escape(token)
        spans.append(
            f'<span class="contrast-tok" style="background:{bg};">'
            f'{text}<span class="contrast-tip">{tip}</span></span>'
        )

    la = escape(result.label_a)
    lb = escape(result.label_b)

    return (
        _CONTRAST_CSS + '<div style="font-family:inherit;line-height:1.75;'
        'white-space:pre-wrap;word-break:break-word;padding:2px 0 6px 0;">'
        + "".join(spans)
        + '<div style="margin-top:10px;font-size:0.72em;color:#888;'
        + 'display:flex;gap:12px;flex-wrap:wrap;">'
        + '<span><span style="background:rgba(210,60,60,0.45);'
        + f'padding:1px 6px;border-radius:2px;">&thinsp;</span>&nbsp;{la}</span>'
        + '<span><span style="background:rgba(50,110,210,0.45);'
        + f'padding:1px 6px;border-radius:2px;">&thinsp;</span>&nbsp;{lb}</span>'
        + '<span style="color:#aaa;">gray = shared by both</span>'
        + "</div>"
        + "</div>"
    )
