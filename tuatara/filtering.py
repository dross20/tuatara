from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from tuatara.pipeline import PipelineStep

if TYPE_CHECKING:
    from tuatara.pair_generation import FineTuningPair


class Filter(PipelineStep):
    """Abstract class that defines the interface for filters."""

    def forward(self, data: list[FineTuningPair]) -> list[FineTuningPair]:
        return self._filter(data)

    @abstractmethod
    def _filter(self, pairs: list[FineTuningPair]) -> list[FineTuningPair]:
        """
        Applies a filter to fine-tuning pairs, returning only those pairs that pass
        some criteria.

        Args:
            pairs: The list of fine-tuning pairs to filter.
        Returns:
            The list of fine-tuning pairs that make it through the filter.
        """
        ...
