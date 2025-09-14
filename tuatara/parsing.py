from pipeline import PipelineStep
from pathlib import Path
from dataclasses import dataclass
from document import Document, Page
import warnings
from abc import ABC, abstractmethod

@dataclass
class ParsedResult():
    text: str
    src_doc: str
    src_page: int


class FileParser(PipelineStep):
    _registry = {}
    
    @classmethod
    def register_parser(cls, extension: str):
        def decorator(target):
            cls._registry[extension] = target
            return target
        return decorator
    
    def forward(self, docs: str | Path | Document | list[Document]):
        if not isinstance(docs, list):
            docs = [docs]
        docs = [Document(doc) if not isinstance(doc, Document) else doc for doc in docs]

        out_docs = []

        for doc in docs:
            parser = self._registry.get(doc.filetype)

            if parser is None:
                supported_filetypes = ", ".join(self._registry.keys())
                warnings.warn(
                    f"Parsing is not supported for file type '{doc.filetype}'. "
                    f"Supported file types: {supported_filetypes}"
                )
            else:
                out_docs.extend(parser().forward(doc))
        return out_docs

class Parser(PipelineStep):
    def forward(self, data: str | Path | Document | list[Document]) -> list[Document]:
        if not isinstance(data, list):
            data = [data]
        data = [Document(doc) if not isinstance(doc, Document) else doc for doc in data]
        parsed_docs = [self.parse(doc.path) for doc in data]
        for doc, parsed_doc in zip(data, parsed_docs):
            for idx, page in enumerate(parsed_doc):
                doc.pages.append(Page(number=idx, text=page))
        return data

    @abstractmethod
    def parse(self, path: Path) -> list[str]:
        ...

@FileParser.register_parser(".txt")
class TXTParser(Parser):
    def parse(self, path: Path) -> list[str]:
        with open(path, "r") as file:
            return [file.read()] 

@FileParser.register_parser(".pdf")
class PDFParser(Parser):
    def __init__(self):
        import pdfplumber
        self.pdfplumber = pdfplumber
    
    def parse(self, path: Path) -> list[str]:
        pages = []
        with self.pdfplumber.open(path) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        return pages