from pipeline import PipelineStep
from document import Document
from abc import abstractmethod
from dataclasses import dataclass

@dataclass
class FineTuningPair():
    prompt: str
    response: str
    source: Document

class PairGenerator(PipelineStep):
    def forward(self, data: list[Document]) -> list[FineTuningPair]:
        all_pairs = []
        for doc in data:
            source_chunks = self._select_source_chunk_groups(doc)
            for chunk_group in source_chunks:
                pair_tuples = self._generate_pairs(chunk_group)
                pairs = [
                    FineTuningPair(prompt=prompt, response=response, source=doc)
                    for prompt, response in pair_tuples
                ]
                all_pairs.extend(pairs)
        return all_pairs

    @abstractmethod
    def _select_source_chunk_groups(self, doc: Document) -> list[list[str]]:
        ...

    @abstractmethod
    def _generate_pairs(self, chunks: list[str]) -> list[tuple[str, str]]:
        ...