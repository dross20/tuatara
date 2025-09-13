from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

class Document:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._pages = []

    @property
    def pages(self) -> list[Page]:
        return self._pages
    
    @property
    def filetype(self) -> str:
        return self.path.suffix

@dataclass
class Page:
    number: int
    text: str