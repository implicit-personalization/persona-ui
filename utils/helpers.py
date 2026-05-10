import hashlib
import re
from collections.abc import Iterable

from persona_data.synth_persona import PersonaData

# Variant key -> human-readable label mapping
VARIANT_LABELS = {
    "empty": "None",
    "baseline": "Baseline",
    "templated": "Template",
    "biography": "Biography",
    "custom": "Custom",
}

CHAT_PROMPT_MODES = ("empty", "templated", "biography", "custom")
CHAT_PROMPT_MODE_LABELS = [VARIANT_LABELS[key] for key in CHAT_PROMPT_MODES]
CHAT_PROMPT_MODE_LABEL_TO_KEY = {VARIANT_LABELS[key]: key for key in CHAT_PROMPT_MODES}


DATASET_SOURCES = [
    "HuggingFace: synth-persona",
    "HuggingFace: nemotron-france",
    "HuggingFace: nemotron-usa",
    "Local JSONL upload",
]
ANALYSIS_MODES = ["Cosine similarity", "Similarity matrix", "PCA", "UMAP", "Dendrogram"]

ANALYSIS_HELP_TEXT = {
    "Cosine similarity": "Compare layer-wise alignment between variants.",
    "Similarity matrix": "Compare centered pairwise similarity between persona vectors by layer, with pair trajectories across layers.",
    "PCA": "Project per-persona vectors into a 2D or 3D global view.",
    "UMAP": "Project per-persona vectors into a 2D or 3D local-neighborhood view.",
    "Dendrogram": "Hierarchical clustering of persona vectors — shows biography and templated side by side for direct comparison.",
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


def personas_fingerprint(persona_ids: Iterable[str]) -> str:
    """Stable short fingerprint for a set of persona ids.

    Used as a discriminator in widget keys and session-state keys. At ~1k
    personas, joining ids would produce ~20 KB strings; the sha1 prefix is
    fixed-length and keeps tracebacks readable.
    """

    joined = "|".join(sorted(persona_ids))
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


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
