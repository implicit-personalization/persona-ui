from __future__ import annotations

import torch

from tabs import probe_ui
from utils import probe_trace


def test_store_derived_cache_evicts_oldest(monkeypatch):
    session_state: dict[str, object] = {}
    monkeypatch.setattr(probe_ui.st, "session_state", session_state)
    monkeypatch.setattr(probe_ui, "_DERIVED_CACHE_ENTRIES", 2)

    probe_ui._store_derived_cache("k1", 1)
    probe_ui._store_derived_cache("k2", 2)
    probe_ui._store_derived_cache("k3", 3)

    assert "k1" not in session_state
    assert session_state["k2"] == 2
    assert session_state["k3"] == 3
    assert session_state[probe_ui._DERIVED_CACHE_TRACKER_KEY] == ["k2", "k3"]


def test_get_derived_cache_refreshes_recently_used_entry(monkeypatch):
    session_state: dict[str, object] = {}
    monkeypatch.setattr(probe_ui.st, "session_state", session_state)
    monkeypatch.setattr(probe_ui, "_DERIVED_CACHE_ENTRIES", 2)

    probe_ui._store_derived_cache("k1", 1)
    probe_ui._store_derived_cache("k2", 2)

    assert probe_ui._get_derived_cache("k1") == 1
    probe_ui._store_derived_cache("k3", 3)

    assert "k1" in session_state
    assert "k2" not in session_state
    assert session_state[probe_ui._DERIVED_CACHE_TRACKER_KEY] == ["k1", "k3"]


def test_trace_eviction_drops_derived_results(monkeypatch):
    session_state: dict[str, object] = {}
    monkeypatch.setattr(probe_trace.st, "session_state", session_state)
    monkeypatch.setattr(probe_trace, "_MAX_CACHED_TRACES", 1)

    trace = probe_trace.ConversationTrace(
        cache_key="old",
        model_name="m",
        remote=False,
        prompt_text="p",
        prompt_hash="h",
        layer=0,
        location="post_reasoning",
        input_ids=torch.tensor([1]),
        activations=torch.zeros((1, 1)),
        tokens=["x"],
        assistant_spans=[],
        is_special=torch.tensor([False]),
    )
    old_prediction_key = "probe_predictions::old::probe"
    kept_prediction_key = "probe_predictions::new::probe"
    session_state[probe_trace._DERIVED_CACHE_TRACKER_KEY] = [
        old_prediction_key,
        kept_prediction_key,
    ]
    session_state[old_prediction_key] = object()
    session_state[kept_prediction_key] = object()

    probe_trace._store_cached_trace("old", trace)
    probe_trace._store_cached_trace(
        "new",
        probe_trace.ConversationTrace(
            **{**trace.__dict__, "cache_key": "new"},
        ),
    )

    assert old_prediction_key not in session_state
    assert kept_prediction_key in session_state
    assert session_state[probe_trace._DERIVED_CACHE_TRACKER_KEY] == [
        kept_prediction_key
    ]
