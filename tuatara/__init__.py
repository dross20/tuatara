"""Library for generating high-quality fine-tuning pairs."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from tuatara.inference import Inference
    from tuatara.pipeline import Pipeline


def _format_log(record):
    """
    Returns a formatted log string. Includes the `step` metadata in the log.

    Args:
        record: The log record.
    Returns:
        The log string, containing the date time, log level, step (if applicable), call
        location, and message.
    """
    missing_step_str = "None"
    extra = record.get("extra", {})
    step = extra.get("step", missing_step_str)
    step = f"{step: <15}"
    if step.strip() == missing_step_str:
        step = f"<dim>{step}</dim>"

    format_str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        f"<light-yellow>step</light-yellow> = <bold>{step}</bold> | "
        "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>\n"
    )
    return format_str


logger.remove()
logger.add(sys.stderr, colorize=True, format=_format_log, level="WARNING")


def default_pipeline(
    inference: Inference | None = None, model: str = "gpt-4o"
) -> Pipeline:
    """
    Creates a default pipeline for converting documents to fine-tuning pairs.

    Args:
        inference: The `Inference` instance to use for generating text completions.
        model: The ID of the model to use for inference.
    """
    from tuatara.chunking import TokenChunker
    from tuatara.filtering import SemanticSimilarityFilter
    from tuatara.inference import OpenAIInference
    from tuatara.pair_generation import StandardPairGenerator
    from tuatara.parsing import AutoParser

    inference = inference or OpenAIInference()

    return (
        AutoParser()
        | TokenChunker(100, overlap=50)
        | StandardPairGenerator(inference, model)
        | SemanticSimilarityFilter(representative_strategy="centroid")
    )


__all__ = ["default_pipeline"]
