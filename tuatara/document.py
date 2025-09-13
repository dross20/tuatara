from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

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