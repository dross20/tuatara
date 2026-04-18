from __future__ import annotations

import importlib
import inspect
from typing import Any


class Serializable:
    def to_config(self) -> dict:
        cfg = {}
        for param in inspect.signature(self.__init__).parameters:
            if param != "self" and hasattr(self, param):
                value = getattr(self, param)
                if isinstance(value, Serializable):
                    value = value.serialize()
                cfg[param] = value
        return cfg

    @classmethod
    def from_config(cls, cfg):
        cfg = {
            k: (
                cls.deserialize(v)
                if isinstance(v, dict) and "class" in v and "config" in v
                else v
            )
            for k, v in cfg.items()
        }
        return cls(**cfg)

    def serialize(self):
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "config": self.to_config(),
        }

    @staticmethod
    def deserialize(dct: dict[str, Any]) -> Serializable:
        def _resolve_class(name: str) -> type:
            module_name, qualified_name = name.rsplit(".", maxsplit=1)
            module = importlib.import_module(module_name)
            return getattr(module, qualified_name)

        return _resolve_class(dct["class"]).from_config(dct["config"])
