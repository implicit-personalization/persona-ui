import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from persona_data.synth_persona import PersonaData, QAPair


@dataclass
class LocalPersonaDataset:
    """Dataset loaded from local JSONL files."""

    personas_path: Path
    qa_path: Path

    def __post_init__(self) -> None:
        with self.personas_path.open() as f:
            self._personas: list[PersonaData] = []
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self._personas.append(
                    PersonaData(
                        id=data["id"],
                        persona=data["persona"],
                        templated_prompt=data["templated_prompt"],
                        biography_md=data["biography_md"],
                    )
                )

        self._qa: dict[str, list[QAPair]] = defaultdict(list)
        with self.qa_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self._qa[data["id"]].append(
                    QAPair(
                        qid=data["qid"],
                        type=data["type"],
                        question=data["question"],
                        answer=data["answer"],
                        difficulty=data["difficulty"],
                    )
                )

    def __len__(self) -> int:
        return len(self._personas)

    def __iter__(self) -> Iterator[PersonaData]:
        return iter(self._personas)

    def __getitem__(self, idx: int) -> PersonaData:
        return self._personas[idx]

    def get_qa(
        self,
        persona_id: str,
        type: Literal["explicit", "implicit"] | None = None,
        difficulty: int | list[int] | None = None,
    ) -> list[QAPair]:
        pairs = self._qa.get(persona_id, [])
        if type is not None:
            pairs = [pair for pair in pairs if pair.type == type]

        if difficulty is not None:
            levels = {difficulty} if isinstance(difficulty, int) else set(difficulty)
            pairs = [pair for pair in pairs if pair.difficulty in levels]

        return pairs
