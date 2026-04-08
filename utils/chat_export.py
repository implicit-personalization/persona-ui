import json
from datetime import datetime, timezone
from pathlib import Path

from persona_data.environment import get_artifacts_dir

from utils.helpers import slugify


def save_chat_export(
    *,
    model_name: str,
    dataset_source: str,
    persona_id: str,
    persona_name: str | None,
    prompt_mode: str,
    system_prompt: str | None,
    messages: list[dict[str, str]],
    generation: dict[str, object],
    panel_label: str | None = None,
) -> Path:
    """Save the current chat session to ``artifacts/chats`` as JSON.

    Args:
        model_name: Model identifier used for the chat.
        dataset_source: Human-readable dataset source label.
        persona_id: Selected persona id.
        persona_name: Selected persona display name, if available.
        prompt_mode: Active system prompt mode.
        system_prompt: Current system prompt text, if any.
        messages: Conversation messages without the system prompt.
        generation: Generation settings used for the chat.

    Returns:
        The path the export was written to.
    """

    payload = {
        "model_name": model_name,
        "dataset_source": dataset_source,
        "persona": {
            "id": persona_id,
            "name": persona_name,
        },
        "panel_label": panel_label,
        "prompt_mode": prompt_mode,
        "generation": generation,
        "messages": (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        + messages,
    }

    export_dir = (
        get_artifacts_dir()
        / "chats"
        / "__".join(slugify(part) for part in model_name.split("/"))
        / slugify(dataset_source)
        / slugify(persona_id)
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename_parts = [
        timestamp,
        slugify(persona_name or persona_id),
        slugify(prompt_mode),
    ]
    if panel_label:
        filename_parts.append(slugify(panel_label))
    export_path = export_dir / f"{'__'.join(filename_parts)}.json"
    export_path.write_text(
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n",
        encoding="utf-8",
    )

    return export_path
