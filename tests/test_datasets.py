from __future__ import annotations

from utils import datasets


class _Progress:
    def __init__(self) -> None:
        self.updates: list[tuple[float, str | None]] = []

    def progress(self, value: float, *, text: str | None = None) -> None:
        self.updates.append((value, text))


def test_download_missing_startup_files_only_fetches_uncached_files(monkeypatch):
    warnings: list[str] = []
    progress = _Progress()
    downloads: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        datasets,
        "_is_cached",
        lambda _repo, filename: filename == "already.jsonl",
    )
    monkeypatch.setattr(datasets.st, "warning", warnings.append)
    monkeypatch.setattr(
        datasets.st,
        "progress",
        lambda value, *, text=None: progress,
    )
    monkeypatch.setattr(
        datasets,
        "hf_hub_download",
        lambda repo, filename, *, repo_type: downloads.append(
            (repo, filename, repo_type)
        ),
    )

    datasets._download_missing_startup_files_if_needed(
        "org/repo",
        ("already.jsonl", "missing.jsonl"),
        "Example",
    )

    assert warnings and "First-time setup for Example" in warnings[0]
    assert downloads == [("org/repo", "missing.jsonl", "dataset")]
    assert progress.updates[-1] == (1.0, "Downloaded missing.jsonl (1/1)")


def test_download_missing_startup_files_stays_quiet_when_cached(monkeypatch):
    monkeypatch.setattr(datasets, "_is_cached", lambda *_args: True)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("cold-download UI should not render for warm cache")

    monkeypatch.setattr(datasets.st, "warning", unexpected)
    monkeypatch.setattr(datasets.st, "progress", unexpected)
    monkeypatch.setattr(datasets, "hf_hub_download", unexpected)

    datasets._download_missing_startup_files_if_needed(
        "org/repo",
        ("cached.jsonl",),
        "Example",
    )


def test_prepare_nemotron_prefetches_first_parquet_shard(monkeypatch):
    calls: list[tuple[str, tuple[str, ...], str]] = []
    monkeypatch.setattr(
        datasets,
        "list_repo_files",
        lambda *_args, **_kwargs: (
            "README.md",
            "data/train-00001-of-00002.parquet",
            "data/train-00000-of-00002.parquet",
        ),
    )
    monkeypatch.setattr(
        datasets,
        "_download_missing_startup_files_if_needed",
        lambda repo, filenames, label: calls.append((repo, filenames, label)),
    )

    datasets._prepare_nemotron_startup_download(
        datasets.DatasetSource.NEMOTRON_USA.value,
        "Nemotron USA",
    )

    assert calls == [
        (
            "nvidia/Nemotron-Personas-USA",
            ("data/train-00000-of-00002.parquet",),
            "Nemotron USA",
        )
    ]


def test_warm_qa_makes_synth_qa_download_visible_before_thread(monkeypatch):
    calls: list[tuple[str, tuple[str, ...], str]] = []
    started: list[bool] = []

    class DummySynth:
        def prefetch_qa(self) -> None:
            pass

    class DummyThread:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            started.append(True)

    monkeypatch.setattr(datasets, "SynthPersonaDataset", DummySynth)
    monkeypatch.setattr(
        datasets,
        "_download_missing_startup_files_if_needed",
        lambda repo, filenames, label: calls.append((repo, filenames, label)),
    )
    monkeypatch.setattr(datasets.threading, "Thread", DummyThread)

    datasets.warm_qa_in_background(DummySynth())

    assert calls == [
        (
            "implicit-personalization/synth-persona",
            ("dataset_qa.jsonl",),
            "SynthPersona QA",
        )
    ]
    assert started == [True]
