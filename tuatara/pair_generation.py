"""Contains all classes related to fine-tuning pair generation.."""

from abc import abstractmethod
from dataclasses import dataclass

from tuatara.document import Document
from tuatara.pipeline import PipelineStep


@dataclass
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
        all_pairs = []
        for doc in data:
            source_chunk_groups = self._select_source_chunk_groups(doc)
            for chunk_group in source_chunk_groups:
                pair_tuples = self._generate_pairs(chunk_group)
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
        return all_pairs

    @abstractmethod
    def _select_source_chunk_groups(self, doc: Document) -> list[list[str]]:
        """
        Selects groups of chunks from a `Document`.

        Args:
            doc: The `Document` from which to extract chunks.
        Returns:
            The list of selected chunk groups.
        """
        ...

    @abstractmethod
    def _generate_pairs(self, chunks: list[str]) -> list[tuple[str, str]]:
        """
        Generates a list of fine-tuning tuples.

        Args:
            chunks: The list of text chunks from which to generate the pairs.
        Returns:
            A list of fine-tuning pairs stored in tuples. Each tuple contains a prompt
            and a ground truth response.
        """
        ...
