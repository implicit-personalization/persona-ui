# Persona UI

Streamlit interface for persona vector extraction, analysis, and chat.

> [!WARNING]
> This is a proof-of-concept UI, mostly vibe-coded. It will likely be replaced by a proper frontend/backend in the future.

## Overview

A web app built on top of [persona-vectors](../persona-vectors) that provides three tabs:

- **Chat** — interactive conversations with a model using persona-based system prompts (templated or biography)
- **Compare** — load saved activations and explore layer-wise cosine similarity, PCA, and UMAP projections
- **Extract** — run activation extraction from HuggingFace or a local JSONL dataset directly from the browser

## Repository Layout

```
persona-ui/
├── app.py                   # Main entry point (Streamlit)
├── state.py                 # Session state management (chat history, KV cache)
├── tabs/
│   ├── chat.py              # Chat tab
│   ├── compare.py           # Activation comparison tab
│   └── extract.py           # Extraction tab
└── utils/
    ├── artifacts.py         # Load saved activations metadata
    ├── chat.py              # Chat generation logic
    ├── chat_export.py       # Export chat logs to JSON
    ├── datasets.py          # Dataset loader wrapper
    ├── extraction.py        # Extraction orchestration
    ├── helpers.py           # UI labels and slug helpers
    ├── local_dataset.py     # Local JSONL dataset parsing
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

## Quickstart

```bash
streamlit run app.py
```

## Configuration

Copy `.env.example` to `.env` and fill in:

```bash
NDIF_API_KEY=...       # Required for remote (NDIF) model execution
HF_HOME=...            # Optional: HuggingFace cache directory
ARTIFACTS_DIR=...      # Optional: where activations are read from (default: ./artifacts)
```

The app picks up this file automatically via `load_env()` on startup.

## Saved Artifacts

The Compare and Extract tabs read from / write to:

```
artifacts/
├── activations/<model_dir>/<prompt_variant>/<persona_id>/
│   ├── activations.safetensors
│   └── metadata.json
└── chats/<model_dir>/<prompt_variant>/
    └── <export>.json
```

`<model_dir>` is the model name with `/` replaced by `__` (e.g. `google__gemma-2-9b-it`).
