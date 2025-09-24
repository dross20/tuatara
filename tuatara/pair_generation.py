"""Contains all classes related to fine-tuning pair generation.."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from tuatara.document import Document
from tuatara.inference import Inference
from tuatara.pipeline import PipelineStep
from tuatara.utils import load_prompt_template, parse_json_pairs

if TYPE_CHECKING:
    from tuatara.document import Page


@dataclass(slots=True)
class FineTuningPair:
    """
    Model class for representing fine-tuning pairs.

    Attributes:
        prompt: The prompt to pass to the LLM during the training instance.
        response: The ground truth response to the prompt.
        source_doc: The `Document` from which the pair was created.
        source_chunks: The list of chunks from which the pair was created.
    """

    prompt: str
    response: str
    source_doc: Document
    source_chunks: list[str]


class PairGenerator(PipelineStep):
    """Abstract class that defines the interface for fine-tuning pair generators."""

    def forward(self, data: list[Document]) -> list[FineTuningPair]:
        """
        Generates a list of fine-tuning pairs grounded by a list of documents.

        Args:
            data: The documents from which to generate the pairs. Information is
                  extracted directly from these documents.
        Returns:
            A list of `FineTuningPair` objects.
        """
        logger.info(f"Creating fine-tuning pairs from {len(data)} documents")

        all_pairs = []
        for doc in data:
            doc_metadata = self._prepare_document_metadata(doc)
            source_chunk_groups = self._select_source_chunk_groups(doc, doc_metadata)
            for chunk_group in source_chunk_groups:
                pair_tuples = self._generate_pairs(chunk_group, doc_metadata)
                pairs = [
                    FineTuningPair(
                        prompt=prompt,
                        response=response,
                        source_doc=doc,
                        source_chunks=chunk_group,
                    )
                    for prompt, response in pair_tuples
                ]
                all_pairs.extend(pairs)

        logger.info(f"Created {len(all_pairs)} fine-tuning pairs")

        return all_pairs

    @abstractmethod
    def _prepare_document_metadata(self, doc: Document) -> dict[str, Any]:
        """
        Prepares a document's metadata for downstream use in pair generation.

        Args:
            doc: The document for which to prepare metadata.
        Returns:
            A metadata dictionary.
        """
        ...

    @abstractmethod
    def _select_source_chunk_groups(
        self, doc: Document, metadata: dict[str, Any]
    ) -> list[list[str]]:
        """
        Selects groups of chunks from a `Document`.

        Args:
            doc: The `Document` from which to extract chunks.
        Returns:
            The list of selected chunk groups.
        """
        ...

    @abstractmethod
    def _generate_pairs(
        self, chunks: list[str], metadata: dict[str, Any]
    ) -> list[tuple[str, str]]:
        """
        Generates a list of fine-tuning tuples.

        Args:
            chunks: The list of text chunks from which to generate the pairs.
        Returns:
            A list of fine-tuning pairs stored in tuples. Each tuple contains a prompt
            and a ground truth response.
        """
        ...


class StandardPairGenerator(PairGenerator):
    """Pair generator that creates pairs by prompting an LLM."""

    def __init__(self, inference: Inference, model: str):
        super().__init__()
        self.inference = inference
        self.model = model

    def _summarize_document(self, document: Document, page_summaries: list[str]) -> str:
        template = load_prompt_template("summarize_document_standard")
        prompt = template.format(
            title=getattr(document, "title", "Unknown"),
            document_length=len(document),
            summaries="\n\n".join(page_summaries),
        )
        return self.inference.generate(self.model, prompt)

    def _summarize_page(self, page: Page) -> str:
        text = page.text
        template = load_prompt_template("summarize_page_standard")
        prompt = template.format(
            page_text=f"{text[:2000]}{'...' if len(text) > 2000 else ''}"
        )
        return self.inference.generate(self.model, prompt)

    def _prepare_document_metadata(self, doc: Document) -> dict[str, Any]:
        page_summaries = [self._summarize_page(page) for page in doc.pages]
        global_summary = self._summarize_document(doc, "\n\n".join(page_summaries))
        return {"global_summary": global_summary, "page_summaries": page_summaries}

    def _select_source_chunk_groups(
        self, doc: Document, metadata: dict[str, Any]
    ) -> list[list[str]]:
        """
        Selects groups of chunks from a `Document`.

        Args:
            doc: The `Document` from which to extract chunks.
        Returns:
            The list of selected chunk groups.
        """
        all_chunks = [chunk for page in doc.pages for chunk in page.chunks]
        return [all_chunks[i : i + 5] for i in range(0, len(all_chunks), 5)]

    def _generate_pairs(
        self, chunks: list[str], metadata: dict[str, Any]
    ) -> list[tuple[str, str]]:
        """
        Generates a list of fine-tuning tuples.

        Args:
            chunks: The list of text chunks from which to generate the pairs.
        Returns:
            A list of fine-tuning pairs stored in tuples. Each tuple contains a prompt
            and a ground truth response.
        """
        chunk_text = "\n\n".join(chunks)

        template = load_prompt_template("generate_pairs_standard")
        prompt = template.format(
            summary=metadata.get("global_summary", ""),
            chunk_text=f"{chunk_text[:3000]}{'...' if len(chunk_text) > 3000 else ''}",
        )

        response = self.inference.generate(self.model, prompt)
        pairs = parse_json_pairs(response)

        return pairs
