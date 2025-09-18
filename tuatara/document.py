"""Contains all classes related to documents, pages, and chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class Document:
    """Encapsulates a document's content and metadata."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.pages = []

    @property
    def filetype(self) -> str:
        """The file extension of the document."""
        return self.path.suffix

    @property
    def n_pages(self) -> int:
        """The number of pages contained in the document."""
        return len(self)

    def __len__(self):
        return len(self.pages)

    def __repr__(self):
        return (
            f"Document(path='{self.path}', filetype='{self.filetype}',"
            f"n_pages='{self.n_pages}')"
        )


@dataclass
class Page:
    """Encapsulates a document page's content and and metadata."""

    number: int
    text: str
    chunks: list[str] = field(default_factory=list)
