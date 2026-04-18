"""Contains definitions for objects that can be include in a pipeline."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, List

import yaml
from loguru import logger

from tuatara.inference import AutoInferenceMeta
from tuatara.serializing import Serializable


@dataclass
class PipelineResult:
    name: str
    result: Any


class Pipeable(ABC, metaclass=AutoInferenceMeta):
    @abstractmethod
    def call(self, data) -> tuple[Any, PipelineResult]: ...

    def __call__(self, *args, **kwds):
        return self.call(*args, **kwds)

    def __or__(self, value: Pipeable | Pipeline):
        if isinstance(value, Pipeable):
            return Pipeline([self]) | value
        else:
            raise TypeError(
                "Instances of `Pipeable` can only be piped with other instances of "
                "`Pipeable`."
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

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.serialize() for step in self.steps]}

    def to_json(self, path: Path | str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, path: Path | str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Pipeline:
        return cls(steps=[Serializable.deserialize(step) for step in dct["steps"]])

    @classmethod
    def from_json(cls, path: Path | str) -> Pipeline:
        with open(path, "r") as f:
            dct = json.load(f)
        return cls.from_dict(dct)

    @classmethod
    def from_yaml(cls, path: Path | str) -> Pipeline:
        with open(path, "r") as f:
            dct = yaml.safe_load(f)
        return cls.from_dict(dct)


class PipelineStep(Pipeable, Serializable):
    @abstractmethod
    def forward(self, data): ...

    def create_result(self, result_value):
        return PipelineResult(type(self).__name__, result_value)

    def call(self, data) -> tuple[Any, PipelineResult]:
        logger.debug(f"Running step {type(self).__name__}")
        start_time = time()

        with logger.contextualize(step=type(self).__name__):
            result_value = self.forward(data)
            result_obj = self.create_result(result_value)

        elapsed_time = time() - start_time
        logger.debug(
            f"Finished running step {type(self).__name__} in {elapsed_time:.2f}s"
        )

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
