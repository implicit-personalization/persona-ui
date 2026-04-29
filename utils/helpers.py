import re

from persona_data.synth_persona import PersonaData

# Variant key -> human-readable label mapping
VARIANT_LABELS = {
    "empty": "None",
    "baseline": "Baseline",
    "templated": "Template",
    "biography": "Biography",
    "custom": "Custom",
}

# For selectbox options: list of labels in definition order
MODE_LABELS = list(VARIANT_LABELS.values())

# Reverse lookup: label -> key
MODE_LABEL_TO_KEY = {v: k for k, v in VARIANT_LABELS.items()}

DATASET_SOURCES = [
    "HuggingFace: synth-persona",
    "HuggingFace: nemotron-france",
    "HuggingFace: nemotron-usa",
    "Local JSONL upload",
]
ANALYSIS_MODES = ["Cosine similarity", "Similarity matrix", "PCA", "UMAP"]

ANALYSIS_HELP_TEXT = {
    "Cosine similarity": "Compare layer-wise alignment between variants.",
    "Similarity matrix": "Compare centered pairwise similarity between persona means by layer, with pair trajectories across layers.",
    "PCA": "Project per-persona mean activations into a 2D global view.",
    "UMAP": "Project per-persona mean activations into a 2D local-neighborhood view.",
}

NDIF_STATUS_ICONS = {
    "RECEIVED": "◉",
    "QUEUED": "◎",
    "DISPATCHED": "◈",
    "RUNNING": "●",
    "COMPLETED": "✓",
    "ERROR": "✗",
}


def slugify(value: str) -> str:
    """Convert a string to a filesystem-safe slug."""

    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def widget_key(*parts: str) -> str:
    """Generate a namespaced Streamlit widget key from parts."""

    return "::".join(parts)


def prompt_variant_label(variant: str) -> str:
    """Return a human-friendly prompt-variant label."""

    return VARIANT_LABELS.get(variant, variant.title())


def persona_label(persona: PersonaData) -> str:
    """Format a persona for selection widgets."""

    return f"{persona.name} ({persona.id})"


def persona_display_label(persona_id: str, persona_name: str | None) -> str:
    """Format a persona id with an optional display name."""

    if persona_name:
        return f"{persona_name} ({persona_id})"
    return persona_id
