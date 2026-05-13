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

A web app built on top of [persona-vectors](../persona-vectors) that provides three tabs:

- **Chat** — interactive conversations with a model using persona-based system prompts (templated or biography)
- **Compare** — load local or Hub persona vectors and explore cosine similarity, PCA, UMAP, attribute-colored projections, and dendrograms
- **Extract** — run activation extraction from HuggingFace persona datasets or a local JSONL dataset directly from the browser

## Repository Layout

```
persona-ui/
├── app.py                   # Main entry point (Streamlit)
├── state.py                 # Session state management (chat history, KV cache)
├── tabs/
│   ├── chat.py              # Chat tab
│   ├── analysis.py          # Analysis tab (cosine similarity, PCA, UMAP, Isomap, dendrogram)
│   ├── compare_chat.py      # Side-by-side chat comparison mode
│   ├── extract.py           # Extraction tab
│   └── probe_ui.py          # Probe upload and tracing controls
└── utils/
    ├── chat.py              # Chat generation logic
    ├── chat_export.py       # Export chat logs to JSON
    ├── contrast.py          # Contrastive token log-prob coloring
    ├── datasets.py          # Dataset loader wrapper
    ├── helpers.py           # UI labels and slug helpers
    ├── probe_trace.py       # Chat-token activation tracing
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

This checkout is configured to use the sibling `../persona-vectors` package as
an editable dependency. For deployment, switch `persona-vectors` back to the
published package or another installable source.

`persona-data` can also be checked out next to this repo for local package work.

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

No secrets needed! The dependencies are published on PyPI.

### Build Locally (Optional)

```bash
docker build -t persona-ui .
# Specify your local .env to have things working as expectd
docker run --env-file .env --rm -p 8501:8501 persona-ui
```

## Configuration

Copy `.env.example` to `.env` and fill in:

```bash
NDIF_API_KEY=...       # Required for remote (NDIF) model execution
HF_HOME=...            # Optional: HuggingFace cache directory
ARTIFACTS_DIR=...      # Optional: where activations are read from (default: ./artifacts)
PERSONA_VECTORS_HUB_REPO=...  # Optional: default Compare-tab Hub dataset repo
```

The app picks up this file automatically via `load_dotenv()` on startup.

## Persona Vectors

The Compare tab reads persona vectors from either a Hugging Face dataset created
by `persona-vectors/scripts/push_to_hf.py` or from local artifacts. The Extract
tab writes local artifacts to:

```
artifacts/
├── activations/<model_dir>/<mask_strategy>/<prompt_variant>/
│   ├── manifest.json
│   └── <persona_id>.safetensors
└── chats/<model_dir>/<persona_id>/
    └── <export>.json
```

`<model_dir>` is the model name with `/` replaced by `__` (e.g. `google__gemma-2-9b-it`). The manifest stores persona names, tensor shape metadata, and sample ids. Chat exports still store `dataset_source` in the JSON payload.
