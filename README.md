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
│   ├── compare_chat.py      # Side-by-side chat comparison mode
│   ├── analysis_core.py     # Analysis tab entry point
│   ├── analysis/            # Analysis tab internals
│   │   ├── _shared.py / _state.py            # Shared loading + session state
│   │   ├── cosine.py        # Cosine similarity view
│   │   ├── dendrogram.py    # Persona dendrograms
│   │   └── layered.py       # PCA/UMAP/Isomap projections
│   ├── extract.py           # Extraction tab
│   ├── probe.py / probe_ui.py  # Probe diagnostics + upload/tracing controls
│   └── probe_sweep.py       # Probe sweep tab
└── utils/
    ├── analysis_sources.py  # Local + Hub persona-vector store wiring
    ├── chat.py              # Chat generation logic
    ├── chat_export.py       # Export chat logs to JSON
    ├── contrast.py          # Contrastive token log-prob coloring
    ├── datasets.py          # Dataset loader wrapper
    ├── helpers.py           # UI labels and slug helpers
    ├── probe_trace.py       # Chat-token activation tracing
    ├── probe_overlay.py     # Per-token probe-score overlay
    ├── probes.py / probe_files.py  # Probe loading, scoring, artifact paths
    ├── preload.py           # Background startup warmup
    └── runtime.py           # Model caching and NDIF queries
```

Dataset loading and environment helpers are provided by the sibling [persona-data](https://github.com/implicit-personalization/persona-data) package. 
Core extraction, analysis, and steering logic comes from [persona-vectors](https://github.com/implicit-personalization/persona-vectors).

## Installation

```bash
uv sync
cp .env.example .env
```

## Local Development

The checked-in dependency config uses published packages. For local package
work, uncomment the `tool.uv.sources` block in `pyproject.toml` and keep sibling checkouts next to this repo.

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

### Build Locally

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
HF_TOKEN=...           # Optional: higher Hugging Face Hub rate limits; public datasets do not require it
ARTIFACTS_DIR=...      # Optional: where persona vectors are read from (default: ./artifacts)
PERSONA_VECTORS_HUB_REPO=...  # Optional: default Analysis/Probing Hub dataset repo
PERSONA_UI_STORE_CACHE_ENTRIES=4      # Optional: open local/Hub vector stores kept warm
PERSONA_UI_VECTOR_CACHE_ENTRIES=4     # Optional: loaded analysis datasets kept warm
PERSONA_UI_PREPARED_CACHE_ENTRIES=8   # Optional: prepared projections / k-means groups kept warm
PERSONA_UI_FIGURE_STATE_ENTRIES=2     # Optional: recent rendered Analysis figures kept in-session
PERSONA_UI_PREPARED_STATE_ENTRIES=4   # Optional: recent projection-ready markers kept in-session
```

The app picks up `.env` automatically via `load_dotenv()` on startup, and hosted
environments such as Hugging Face Spaces can provide the same values as
environment variables. If `NDIF_API_KEY` is unset, Chat and Extract users are prompted for a per-session key when they need remote execution.

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

`<model_dir>` is the model name with `/` replaced by `__` (e.g. `google__gemma-2-9b-it`). 
The manifest stores persona names, tensor shape metadata, and sample ids. 
Chat exports still store `dataset_source` in the JSON payload.
