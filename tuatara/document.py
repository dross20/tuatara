from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class Document:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.pages = []

    @property
    def filetype(self) -> str:
        return self.path.suffix

    def __len__(self):
        return len(self.pages)


@dataclass
class Page:
    number: int
    text: str
    chunks: list[str] = field(default_factory=list)
