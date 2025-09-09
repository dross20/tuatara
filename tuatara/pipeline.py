from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

@dataclass
class PipelineResult:
    name: str
    result: Any

class Pipeline():
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps
    
    def __call__(self, data) -> tuple[Any, List[PipelineResult]]:
        results = []
        for step in self.steps:
            data = step.forward(data)
            results.append(
                PipelineResult(
                    type(step).__name__, data
                )
            )
        return data, results
    
    def __or__(self, value):
        if isinstance(value, PipelineStep):
            return Pipeline([*self.steps, value])
        if isinstance(value, Pipeline):
            return Pipeline([*self.steps, *value.steps])

class PipelineStep(ABC):
    @abstractmethod
    def forward(self, data):
        ...

    def __or__(self, value):
        if isinstance(value, PipelineStep):
            return Pipeline([self, value])
        elif isinstance(value, Pipeline):
            return Pipeline([self, *value.steps])
        else:
            raise TypeError("Instances of `PipelineStep` cannot be piped with non-`PipelineStep` instances")