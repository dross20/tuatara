from __future__ import annotations

import importlib
import inspect
from typing import Any


class Serializable:
    """
    Class that can be serialized.
    """

    def _to_config(self) -> dict[str, Any]:
        """
        Serializes an object's configuration settings.

        Returns:
            A `dict`, where each key is the name of the setting and each value is the
            setting itself.
        """
        cfg = {}
        for param in inspect.signature(self.__init__).parameters:
            if param != "self" and hasattr(self, param):
                value = getattr(self, param)
                if isinstance(value, Serializable):
                    value = value.serialize()
                cfg[param] = value
        return cfg

    @classmethod
    def _from_config(cls, cfg: dict[str, Any]) -> Serializable:
        """
        Instantiates an object from its configuration settings.

        Args:
            cfg: The configuration settings from which to create the instance.
        Returns:
            The newly instantiated object.
        """
        cfg = {
            k: (
                cls.deserialize(v)
                if isinstance(v, dict) and "class" in v and "config" in v
                else v
            )
            for k, v in cfg.items()
        }
        return cls(**cfg)

    def serialize(self) -> dict[str, Any]:
        """
        Serializes an instance to a `dict`.

        Returns:
            The `dict`, with fields for the instance's class and configuration
            settings.
        """
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "config": self._to_config(),
        }

    @staticmethod
    def deserialize(dct: dict[str, Any]) -> Serializable:
        """
        Deserializes an instance from a `dict`.

        Args:
            dct: The `dict` from which to deserialize the object.
        Returns:
            The deserialized instance.
        """

        def _resolve_class(name: str) -> type:
            module_name, qualified_name = name.rsplit(".", maxsplit=1)
            module = importlib.import_module(module_name)
            return getattr(module, qualified_name)

        return _resolve_class(dct["class"])._from_config(dct["config"])
