from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def synth_persona_dataset_cached() -> Any:
    from persona_data.synth_persona import SynthPersonaDataset

    return SynthPersonaDataset()


@lru_cache(maxsize=1)
def synth_persona_attribute_names() -> tuple[str, ...]:
    return tuple(synth_persona_dataset_cached().attribute_names)
