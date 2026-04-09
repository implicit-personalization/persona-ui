---
title: persona-ui
emoji: "🤖"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---
# Persona UI

[![Deploy to Hugging Face Spaces](https://huggingface.co/landing/badge.svg)](https://huggingface.co/spaces/implicit-personalization/persona-ui)

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
    ├── chat.py              # Chat generation logic
    ├── chat_export.py       # Export chat logs to JSON
    ├── datasets.py          # Dataset loader wrapper
    ├── helpers.py           # UI labels and slug helpers
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

The committed dependency graph uses git sources so `persona-ui` can install cleanly in a Hugging Face Space or any isolated environment.

For local sibling checkouts, uncomment the `path` sources in `persona-ui/pyproject.toml` and `persona-vectors/pyproject.toml`, then comment out the git sources.

## Local Setup Note

For local development, `persona-data` and `persona-vectors` can still be checked out in the parent directory of `persona-ui`.

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
docker run -p 8501:8501 persona-ui
```

## Configuration

Copy `.env.example` to `.env` and fill in:

```bash
NDIF_API_KEY=...       # Required for remote (NDIF) model execution
HF_HOME=...            # Optional: HuggingFace cache directory
ARTIFACTS_DIR=...      # Optional: where activations are read from (default: ./artifacts)
```

The app picks up this file automatically via `load_dotenv()` on startup.

You can also override the active NDIF or Hugging Face token from the sidebar
`API Keys` section. Those inputs only apply for the current session.

## Saved Artifacts

The Compare and Extract tabs read from / write to:

```
artifacts/
├── activations/<model_dir>/<prompt_variant>/<persona_id>/
│   ├── activations.safetensors
│   └── metadata.json   # used for persona names and layer counts
└── chats/<model_dir>/<prompt_variant>/
    └── <export>.json
```

`<model_dir>` is the model name with `/` replaced by `__` (e.g. `google__gemma-2-9b-it`).
