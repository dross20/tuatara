from pipeline import PipelineStep
from pathlib import Path
from dataclasses import dataclass
from document import Document
import warnings

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
    
    def forward(self, docs: Document | list[Document]):
        if not isinstance(docs, list):
            docs = [docs]

        parsed_results = []

        for doc in docs:
            parser = self._registry.get(doc.filetype)

            if parser is None:
                supported_filetypes = ", ".join(self._registry.keys())
                warnings.warn(
                    f"Parsing is not supported for file type '{doc.filetype}'. "
                    f"Supported file types: {supported_filetypes}"
                )
            else:
                parsed_results.extend(parser()(doc.path))
        return parsed_results


@FileParser.register_parser(".txt")
class TXTParser(PipelineStep):
    def forward(self, docs: Document | list[Document]):
        if not isinstance(docs, list):
            docs = [docs]

        texts = []
        for doc in docs:
            with open(doc.path, "r") as file:
                texts.append(ParsedResult(file.read(), doc, 1))
        return texts
        

@FileParser.register_parser(".pdf")
class PDFParser(PipelineStep):
    def __init__(self):
        import pdfplumber
        self.ocr = pdfplumber
    
    def forward(self, docs: Document | list[Document]):
        if not isinstance(docs, list):
            docs = list[docs]

        texts = []
        for doc in docs:
            with self.ocr.open(doc.path) as pdf:
                texts.extend([ParsedResult(page.extract_text(), doc, i) for i, page in enumerate(pdf.pages)])
        return texts