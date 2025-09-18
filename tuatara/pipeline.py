from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List


@dataclass
class PipelineResult:
    name: str
    result: Any


class Pipeable(ABC):
    @abstractmethod
    def call(self, data) -> tuple[Any, PipelineResult]: ...

    def __call__(self, *args, **kwds):
        return self.call(*args, **kwds)

    def __or__(self, value: Pipeable | Pipeline):
        if isinstance(value, Pipeable):
            return Pipeline([self]) | value
        else:
            raise TypeError(
                "Instances of `Pipeable` cannot be piped with objects that don't "
                "implement `Pipeable`"
            )


class Pipeline(Pipeable):
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    def call(self, data) -> tuple[Any, List[PipelineResult]]:
        history = []
        for step in self.steps:
            data, result_obj = step(data)
            history.append(result_obj)
        return data, history

    def __or__(self, value):
        if isinstance(value, Pipeable):
            return Pipeline([*self.steps, value])
        if isinstance(value, Pipeline):
            return Pipeline([*self.steps, *value.steps])


class PipelineStep(Pipeable):
    @abstractmethod
    def forward(self, data): ...

    def create_result(self, result_value):
        return PipelineResult(type(self).__name__, result_value)

    def call(self, data) -> tuple[Any, PipelineResult]:
        result_value = self.forward(data)
        result_obj = self.create_result(result_value)
        return result_value, result_obj


class ParallelPipelineStep(Pipeable):
    def __init__(self, steps: list[PipelineStep] | Pipeline):
        if isinstance(steps, Pipeline):
            steps = steps.steps
        self.steps = steps

    def call(self, data):
        values, results = zip(*[step(data) for step in self.steps])
        result_obj = PipelineResult(type(self).__name__, results)
        return list(values), result_obj
