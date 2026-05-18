"""Per-message probe overlays that paint assistant text with class colors.

Mirrors the integration shape of ``utils.contrast``: one overlay per assistant
message, attached as ``message["_probe_overlay"]`` and rendered inline by
``render_chat_message``. Overlays cover only the message body — special tokens
(role markers, BOS/EOS) are filtered out at build time.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import torch

from utils.probe_trace import ConversationTrace

_CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (210, 60, 60),
    (50, 110, 210),
    (60, 170, 90),
    (210, 150, 50),
    (170, 80, 200),
    (200, 80, 130),
    (90, 180, 200),
    (170, 170, 70),
)
_MAX_ALPHA = 0.55

_PROBE_CSS = (
    "<style>"
    ".probe-tok{position:relative;border-radius:2px;padding:0 1px;"
    "cursor:default;white-space:pre;}"
    ".probe-tok>.probe-tip{display:none;position:absolute;bottom:100%;"
    "left:50%;transform:translateX(-50%);margin-bottom:4px;padding:2px 6px;"
    "border-radius:3px;background:#222;color:#eee;font-size:0.72em;"
    "font-family:ui-monospace,monospace;white-space:nowrap;pointer-events:none;"
    "z-index:10;box-shadow:0 2px 6px rgba(0,0,0,0.3);}"
    ".probe-tok:hover>.probe-tip{display:block;}"
    ".probe-wrap{line-height:1.75;white-space:pre-wrap;word-break:break-word;}"
    "</style>"
)


@dataclass(frozen=True)
class ProbeOverlay:
    tokens: list[str]
    labels: list[str | None]
    is_regression: bool
    attribute_name: str | None
    # Classification fields (empty when is_regression).
    probs: list[list[float]]
    predicted: list[int]
    binary: bool
    # Regression field (empty when not is_regression).
    values: list[float]


# ---------------------------------------------------------------------------
# Building overlays from a trace
# ---------------------------------------------------------------------------


def _body_indices(trace: ConversationTrace, start: int, end: int) -> list[int]:
    """Indices inside an assistant span, with special tokens dropped."""
    return [i for i in range(start, end) if not bool(trace.is_special[i].item())]


def build_classification_overlays(
    *,
    trace: ConversationTrace,
    probs: torch.Tensor,
    predicted: torch.Tensor,
    labels: list[str | None],
    binary: bool,
    attribute_name: str | None = None,
) -> list[ProbeOverlay]:
    overlays: list[ProbeOverlay] = []
    for start, end in trace.assistant_spans:
        idx = _body_indices(trace, start, end)
        if not idx:
            continue
        overlays.append(
            ProbeOverlay(
                tokens=[trace.tokens[i] for i in idx],
                labels=list(labels),
                is_regression=False,
                attribute_name=attribute_name,
                probs=[probs[i].tolist() for i in idx],
                predicted=[int(predicted[i].item()) for i in idx],
                binary=binary,
                values=[],
            )
        )
    return overlays


def build_regression_overlays(
    *,
    trace: ConversationTrace,
    values: torch.Tensor,
    labels: list[str | None],
    attribute_name: str | None = None,
) -> list[ProbeOverlay]:
    if values.ndim == 2 and values.shape[1] >= 1:
        values = values[:, 0]
    overlays: list[ProbeOverlay] = []
    for start, end in trace.assistant_spans:
        idx = _body_indices(trace, start, end)
        if not idx:
            continue
        overlays.append(
            ProbeOverlay(
                tokens=[trace.tokens[i] for i in idx],
                labels=list(labels),
                is_regression=True,
                attribute_name=attribute_name,
                probs=[],
                predicted=[],
                binary=False,
                values=[float(values[i].item()) for i in idx],
            )
        )
    return overlays


def attach_overlays(messages: list[dict], overlays: list[ProbeOverlay]) -> None:
    """Attach one overlay to each assistant message, in order.

    Requires a 1:1 match. If the counts don't line up (e.g. the chat template
    doesn't mark assistant tokens), clear overlays so the caller can show a
    clear status instead of painting the wrong message.
    """
    assistant_idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    clear_overlays(messages)
    if not assistant_idxs or len(overlays) != len(assistant_idxs):
        return
    for msg_idx, overlay in zip(assistant_idxs, overlays, strict=True):
        messages[msg_idx]["_probe_overlay"] = overlay


def clear_overlays(messages: list[dict]) -> None:
    for message in messages:
        message.pop("_probe_overlay", None)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _label_for(labels: list[str | None], idx: int) -> str:
    if 0 <= idx < len(labels) and labels[idx]:
        return labels[idx]
    return str(idx)


def _display_token(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ")


def _background(
    probs_row: list[float], pred_idx: int, *, binary: bool, num_classes: int
) -> str:
    if binary:
        score = probs_row[0] if len(probs_row) == 1 else probs_row[-1]
        signed = score - 0.5
        alpha = min(1.0, abs(signed) * 2) * _MAX_ALPHA
        r, g, b = (210, 60, 60) if signed > 0 else (50, 110, 210)
    else:
        baseline = 1.0 / max(num_classes, 2)
        confidence = probs_row[pred_idx] if 0 <= pred_idx < len(probs_row) else 0.0
        normalized = max(0.0, (confidence - baseline) / max(1e-6, 1.0 - baseline))
        alpha = normalized * _MAX_ALPHA
        r, g, b = _CLASS_COLORS[pred_idx % len(_CLASS_COLORS)]
    if alpha < 0.02:
        return "transparent"
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _tooltip(probs_row: list[float], labels: list[str | None]) -> str:
    if len(probs_row) == 1:
        positive = probs_row[0]
        positive_label = _label_for(labels, 0)
        # Single-output sigmoid: synthesize the complementary class so the
        # hover shows both label probabilities, not just one.
        return escape(
            f"{positive_label} {positive:.2f} · not {positive_label} {1 - positive:.2f}"
        )
    ranked = sorted(enumerate(probs_row), key=lambda item: item[1], reverse=True)
    parts = [f"{_label_for(labels, idx)} {prob:.2f}" for idx, prob in ranked]
    return escape(" · ".join(parts))


def _regression_background(value: float, normalizer: float) -> str:
    """Red for positive, blue for negative, alpha by |value| relative to span max."""
    if normalizer <= 1e-9:
        return "transparent"
    intensity = min(1.0, abs(value) / normalizer) * _MAX_ALPHA
    if intensity < 0.02:
        return "transparent"
    r, g, b = (210, 60, 60) if value >= 0 else (50, 110, 210)
    return f"rgba({r},{g},{b},{intensity:.3f})"


def render_probe_html(overlay: ProbeOverlay) -> str:
    """Render the assistant message as colored token spans with hover tips."""
    spans: list[str] = []
    if overlay.is_regression:
        normalizer = max((abs(v) for v in overlay.values), default=0.0)
        attribute = overlay.attribute_name or (
            overlay.labels[0] if overlay.labels and overlay.labels[0] else "prediction"
        )
        for token, value in zip(overlay.tokens, overlay.values, strict=True):
            bg = _regression_background(value, normalizer)
            tip = escape(f"{attribute}: {value:.3f}")
            text = escape(_display_token(token))
            spans.append(
                f'<span class="probe-tok" style="background:{bg};">'
                f'{text}<span class="probe-tip">{tip}</span></span>'
            )
    else:
        num_classes = max(1, len(overlay.probs[0]) if overlay.probs else 1)
        for token, probs_row, pred_idx in zip(
            overlay.tokens, overlay.probs, overlay.predicted, strict=True
        ):
            bg = _background(
                probs_row, pred_idx, binary=overlay.binary, num_classes=num_classes
            )
            tip = _tooltip(probs_row, overlay.labels)
            text = escape(_display_token(token))
            spans.append(
                f'<span class="probe-tok" style="background:{bg};">'
                f'{text}<span class="probe-tip">{tip}</span></span>'
            )
    return _PROBE_CSS + '<div class="probe-wrap">' + "".join(spans) + "</div>"
