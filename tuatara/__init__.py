from __future__ import annotations

from typing import TYPE_CHECKING

from tuatara.chunking import TokenChunker
from tuatara.filtering import SemanticSimilarityFilter
from tuatara.inference import OpenAIInference
from tuatara.pair_generation import StandardPairGenerator
from tuatara.parsing import AutoParser

if TYPE_CHECKING:
    from tuatara.inference import Inference
    from tuatara.pipeline import Pipeline


def default_pipeline(
    inference: Inference | None = None, model: str = "gpt-4o"
) -> Pipeline:
    """
    Creates a default pipeline for converting documents to fine-tuning pairs.

    Args:
        inference: The `Inference` instance to use for generating text completions.
        model: The ID of the model to use for inference.
    """
    if inference is None:
        inference = OpenAIInference()

    return (
        AutoParser()
        | TokenChunker(100, overlap=50)
        | StandardPairGenerator(inference, model)
        | SemanticSimilarityFilter(representative_strategy="centroid")
    )
