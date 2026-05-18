from __future__ import annotations

from utils import runtime


def test_session_ndif_api_key_is_read_from_current_session(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime.st,
        "session_state",
        {"sidebar:ndif_api_key": "user-a-key"},
    )
    assert runtime.session_ndif_api_key() == "user-a-key"

    monkeypatch.setattr(
        runtime.st,
        "session_state",
        {"sidebar:ndif_api_key": "user-b-key"},
    )
    assert runtime.session_ndif_api_key() == "user-b-key"


def test_configured_ndif_api_key_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("NDIF_API_KEY", "env-key")
    assert runtime.configured_ndif_api_key() == "env-key"


def test_remote_backend_binds_explicit_session_key(monkeypatch) -> None:
    from nnsight.intervention.backends import remote

    seen: list[str | None] = []

    class FakeBackend:
        def __init__(self, model_key: str, api_key: str | None = None) -> None:
            self.model_key = model_key
            self.api_key = api_key
            self.verbose = False
            seen.append(api_key)

    class FakeModel:
        def to_model_key(self) -> str:
            return "model-key"

    monkeypatch.setattr(remote, "RemoteBackend", FakeBackend)
    monkeypatch.setattr(
        runtime.st,
        "session_state",
        {"sidebar:ndif_api_key": "ambient-session-key"},
    )

    backend = runtime.remote_backend(FakeModel(), "explicit-user-key")

    assert backend.api_key == "explicit-user-key"
    assert seen == ["explicit-user-key"]


def test_remote_backend_falls_back_to_environment_key(monkeypatch) -> None:
    from nnsight.intervention.backends import remote

    class FakeBackend:
        def __init__(self, model_key: str, api_key: str | None = None) -> None:
            self.model_key = model_key
            self.api_key = api_key
            self.verbose = False

    class FakeModel:
        def to_model_key(self) -> str:
            return "model-key"

    monkeypatch.setattr(remote, "RemoteBackend", FakeBackend)
    monkeypatch.setattr(runtime.st, "session_state", {})
    monkeypatch.setenv("NDIF_API_KEY", "env-key")

    backend = runtime.remote_backend(FakeModel())

    assert backend.api_key == "env-key"
