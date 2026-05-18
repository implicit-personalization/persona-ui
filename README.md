---
title: persona-ui
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---
# Persona UI

[![Deploy to Hugging Face Spaces](https://huggingface.co/spaces/implicit-personalization/persona-ui/badge.svg)](https://huggingface.co/spaces/implicit-personalization/persona-ui)

Streamlit interface for persona vector extraction, analysis, and chat.

> [!WARNING]
> This is a proof-of-concept UI, mostly vibe-coded. It will likely be replaced by a proper frontend/backend in the future.

## Overview

A web app built on top of [persona-vectors](../persona-vectors) that provides these tabs:

- **Chat** — interactive conversations with a model using persona-based system prompts (templated or biography)
- **Analysis** — load local or Hub persona vectors and explore cosine similarity, PCA, UMAP, attribute-colored projections, and dendrograms
- **Probing** — sweep and inspect linear probes trained over saved persona vectors
- **Extract** — run persona-vector extraction from HuggingFace persona datasets or a local JSONL dataset directly from the browser

## Repository Layout

```
persona-ui/
├── app.py                   # Main entry point (Streamlit)
├── state.py                 # Session state management (chat history, KV cache)
├── tabs/
│   ├── chat.py / chat_ui.py / chat_shared.py  # Chat tab
│   ├── analysis_core.py                       # Analysis tab (cosine sim, PCA, UMAP, Isomap, dendrogram)
│   ├── compare_chat.py      # Side-by-side chat comparison mode
│   ├── extract.py           # Extraction tab
│   ├── probe.py             # Probe sweep + diagnostics tab
│   └── probe_ui.py          # Probe upload and tracing controls
└── utils/
    ├── analysis_sources.py  # Local + Hub persona-vector store wiring
    ├── chat.py              # Chat generation logic
    ├── chat_export.py       # Export chat logs to JSON
    ├── contrast.py          # Contrastive token log-prob coloring
    ├── datasets.py          # Dataset loader wrapper
    ├── helpers.py           # UI labels and slug helpers
    ├── probe_trace.py       # Chat-token activation tracing
    ├── probe_overlay.py     # Per-token probe-score overlay
    ├── probes.py            # Probe loading and scoring
    └── runtime.py           # Model caching and NDIF queries
```

Dataset loading and environment helpers are provided by the sibling
[persona-data](../persona-data) package. Core extraction, analysis, and
steering logic comes from [persona-vectors](../persona-vectors).

## Installation

```bash
uv sync
cp .env.example .env
```

## Local Development

The checked-in dependency config uses published packages. For local package
work, uncomment the `tool.uv.sources` block in `pyproject.toml` and keep sibling
checkouts next to this repo.

Example:

```bash
git clone <persona-data-url> ../persona-data
git clone <persona-vectors-url> ../persona-vectors
```

Expected layout:

```text
parent/
├── persona-ui
├── persona-data
└── persona-vectors
```

## Quickstart

```bash
streamlit run app.py
```

## Hugging Face Spaces Deployment

This app can be deployed to Hugging Face Spaces using Docker.

### Prerequisites

Dependencies are published on PyPI, so deployment does not require sibling
checkouts. Remote NDIF execution still needs an API key, either configured as an
environment variable or entered by each user in the sidebar.

### Build Locally (Optional)

```bash
docker build -t persona-ui .
# Pass your local .env if you want the container to use the same configuration
docker run --env-file .env --rm -p 8501:8501 persona-ui
```

## Configuration

Copy `.env.example` to `.env` and fill in:

```bash
NDIF_API_KEY=...       # Optional shared NDIF key; users can also enter one per session
HF_HOME=...            # Optional: HuggingFace cache directory
ARTIFACTS_DIR=...      # Optional: where persona vectors are read from (default: ./artifacts)
PERSONA_VECTORS_HUB_REPO=...  # Optional: default Analysis/Probing Hub dataset repo
PERSONA_UI_VECTOR_CACHE_ENTRIES=4     # Optional: loaded analysis datasets kept warm
PERSONA_UI_PREPARED_CACHE_ENTRIES=8   # Optional: prepared projections / k-means groups kept warm
PERSONA_UI_FIGURE_STATE_ENTRIES=2     # Optional: recent rendered Analysis figures kept in-session
PERSONA_UI_PREPARED_STATE_ENTRIES=4   # Optional: recent projection-ready markers kept in-session
```

The app picks up this file automatically via `load_dotenv()` on startup. If
`NDIF_API_KEY` is unset, Chat and Extract users are prompted for a per-session
key when they need remote execution.

## Persona Vectors

The Analysis and Probing tabs read persona vectors from either a Hugging Face
dataset (pushed by `persona-vectors/main.py push` or the
`extraction_*.sh` scripts) or from local artifacts. The Extract tab writes
local artifacts to:

```
artifacts/
├── activations/<model_dir>/<mask_strategy>/<prompt_variant>/   # also: persona-vectors/...
│   ├── manifest.json
│   └── <persona_id>.safetensors
└── chats/<model_dir>/<persona_id>/
    └── <export>.json
```

`<model_dir>` is the model name with `/` replaced by `__` (e.g.
`google__gemma-2-9b-it`). The manifest stores persona names, tensor shape
metadata, and sample ids. Chat exports still store `dataset_source` in the
JSON payload.

The all-questions extraction script (`persona-vectors/scripts/extraction_all_questions.sh`)
writes to `artifacts/persona-vectors/` instead of `artifacts/activations/`,
so all-questions and train-split runs can coexist; point `ARTIFACTS_DIR` (or
the Analysis/Probing tab's Local source path) at the tree you want to load.

The store classes are `PersonaVectorStore` (local) and `HFPersonaVectorStore`
(Hub) — same API, both imported by `utils/analysis_sources.py`.

## Analysis responsiveness

The Analysis tab keeps small bounded caches of loaded vector datasets, prepared projection data, and a tiny MRU window of rendered figures. Once a projection has been computed, recoloring it by persona, attribute, or k-means group reuses the same coordinates; nearby method switches can reuse the last couple of figures instead of rebuilding immediately, while the caps keep RAM bounded. Tune `PERSONA_UI_VECTOR_CACHE_ENTRIES` if RAM is tight or you regularly switch among many selections, `PERSONA_UI_PREPARED_CACHE_ENTRIES` if you revisit several projection configurations in one session, and `PERSONA_UI_FIGURE_STATE_ENTRIES` if you want more or less method-switch warmth. Probe loading, probe sweeps, and per-trace probe outputs are bounded separately via `PERSONA_UI_PROBE_CACHE_ENTRIES`, `PERSONA_UI_PROBE_SWEEP_CACHE_ENTRIES`, and `PERSONA_UI_PROBE_DERIVED_CACHE_ENTRIES`; the derived-output cache defaults to a wider MRU window because those tensors are small compared with traced activations and are cheap wins to keep warm.
