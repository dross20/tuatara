"""Contains all classes related to text, table, and image parsing."""

import re
import warnings
from abc import abstractmethod
from pathlib import Path

from loguru import logger

from tuatara.document import Document, Page
from tuatara.pipeline import PipelineStep


class AutoParser(PipelineStep):
    """Filetype-agnostic document parser."""

    _registry = {}
    _registry_re = {}

    @classmethod
    def register_parser(cls, extension: str):
        """
        Register a new parser in `AutoParser`'s registry. `Parser` child classes
        registered with this method will be used when `AutoParser` processes files
        with the given extension.

        Args:
            extension: The file extension that this parser handles.
        """

        def decorator(target):
            cls._registry[extension] = target
            return target

        return decorator

    @classmethod
    def register_parser_re(cls, pattern: str):
        """
        Register a new parser in `AutoParser`'s registry. `Parser` child classes
        registered with this method will be used when `AutoParser` processes files
        whose paths match the given regular expression.

        Args:
            pattern: The regular expression against which paths will be matched.
        """

        def decorator(target):
            cls._registry_re[re.compile(pattern)] = target
            return target

        return decorator

    def forward(self, docs: str | Path | Document | list[Document]) -> list[Document]:
        """
        Parse document(s) according to their file extension.

        Args:
            docs: The document(s) from which to parse text.
        Returns:
            A list containing a `Document` object for each input file.
        """
        if not isinstance(docs, list):
            docs = [docs]
        docs = [Document(doc) if not isinstance(doc, Document) else doc for doc in docs]

        out_docs = []

        for doc in docs:
            matched = False
            for pattern, parser in self._registry_re.items():
                if pattern.match(str(doc.path)):
                    out_docs.extend(parser().forward(doc))
                    matched = True
                    break

            if not matched:
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
    """Abstract class that defines the interface for parsers."""

    def forward(self, data: str | Path | Document | list[Document]) -> list[Document]:
        """
        Parses text from the file or `Document` provided as input.

        Args:
            data: The path to the file or `Document` from which to parse text.
        Returns:
            A `Document` object whose `pages` attribute is populated with parsed text.
        """
        if not isinstance(data, list):
            data = [data]

        logger.debug(f"Parsing {len(data)} documents")

        data = [Document(doc) if not isinstance(doc, Document) else doc for doc in data]
        parsed_docs = [self.parse(doc.path) for doc in data]
        for doc, parsed_doc in zip(data, parsed_docs):
            for idx, page in enumerate(parsed_doc):
                doc.pages.append(Page(number=idx, text=page))
        return data

    @abstractmethod
    def parse(self, path: Path) -> list[str]:
        """
        Parses text from each page of the file located at `path`.

        Args:
            path: The file path of the document from which to parse text.
        Returns:
            A list of strings, each of which contains the text from one page in the
            input document.
        """
        ...


@AutoParser.register_parser(".txt")
class TXTParser(Parser):
    """Parser for `.txt` files."""

    def parse(self, path: Path) -> list[str]:
        return [path.read_text(encoding="utf-8")]


@AutoParser.register_parser(".pdf")
class PDFParser(Parser):
    """
    Parser for `.pdf` files. Uses the `pdfplumber` library for text, table, and
    image extraction.
    """

    def __init__(self):
        import pdfplumber

        self.pdfplumber = pdfplumber

    def parse(self, path: Path) -> list[str]:
        pages = []
        with self.pdfplumber.open(path) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        return pages


@AutoParser.register_parser_re(r"https?://\S+")
class WebScrapingParser(Parser):
    """
    Parser for web URLs. Use the `requests` and `BeautifulSoup` libraries to scrape
    webpages. By default, webpages are split on "h1" tags.
    """

    def __init__(self):
        import requests
        from bs4 import BeautifulSoup

        self.requests = requests
        self.BeautifulSoup = BeautifulSoup

    def parse(self, path: Path, delimiter: str = "h1") -> list[str]:
        response = self.requests.get(str(path))
        response.raise_for_status()
        soup = self.BeautifulSoup(response.text, "html.parser")

        pages = []
        for section in soup.find_all(delimiter):
            text = []
            for sibling in section.find_next_siblings():
                if sibling.name == delimiter:
                    break
                text.append(sibling.get_text())
            pages.append(section.get_text() + "\n" + "\n".join(text))

        return pages if pages else [soup.get_text()]
