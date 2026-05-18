from __future__ import annotations

from utils import analysis_sources


class _Notice:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.empty_calls = 0

    def warning(self, message: str) -> None:
        self.messages.append(message)

    def empty(self) -> None:
        self.empty_calls += 1


def test_hub_vector_notice_is_transient_for_unopened_variants(monkeypatch):
    notice = _Notice()

    class DummyHubStore:
        _datasets = {"templated": object()}

    monkeypatch.setattr(
        analysis_sources,
        "HFPersonaVectorStore",
        DummyHubStore,
    )
    monkeypatch.setattr(analysis_sources.st, "empty", lambda: notice)

    with analysis_sources._hub_vector_notice(
        DummyHubStore(), ("templated", "biography")
    ):
        pass

    assert notice.messages
    assert "persona vectors from Hugging Face" in notice.messages[0]
    assert notice.empty_calls == 1


def test_hub_vector_notice_stays_quiet_when_variants_are_open(monkeypatch):
    class DummyHubStore:
        _datasets = {"templated": object()}

    monkeypatch.setattr(
        analysis_sources,
        "HFPersonaVectorStore",
        DummyHubStore,
    )

    called = []
    monkeypatch.setattr(analysis_sources.st, "empty", lambda: called.append(True))

    with analysis_sources._hub_vector_notice(DummyHubStore(), ("templated",)):
        pass

    assert called == []
