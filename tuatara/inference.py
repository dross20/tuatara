"""Contains all classes related to LLM inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, Callable
import inspect

if TYPE_CHECKING:
    from openai import OpenAI


class AutoInferenceMeta(type):
    """
    Metaclass for automatically converting inference clients to `Inference` instances.
    """
    _factory_registry = {}

    def __call__(cls, *args, **kwargs):
        converted_args, converted_kwargs = cls._convert_args(args, kwargs)
        return super().__call__(*converted_args, **converted_kwargs)

    @classmethod
    def _convert_args(
        cls, args: list[Any], kwargs: dict[Any]
    ) -> tuple[tuple[Any], dict[str, Any]]:
        """
        Converts any argument whose type annotation is `Inference` into an `Inference`
        instance.

        Args:
            args: The positional arguments to convert.
            kwargs: The keyword arguments to convert.
        Returns:
            A tuple containing the converted positional arguments and keyword arguments.
        """
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        
        converted_args, converted_kwargs = [], {}

        positional_params = [
            param for param in params.values()
            if param.kind in [
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]
            and param.name != "self"
        ]

        for i, arg in enumerate(args):
            if i < len(positional_params):
                param = positional_params[i]
                if param.annotation == Inference and not isinstance(arg, Inference):
                    try:
                        inference = cls._from_object(arg)
                        converted_args.append(inference)
                    except Exception:
                        raise TypeError(
                            f"Object of type {type(arg).__name__} could not be cast to `Inference`"
                        )
                else:
                    converted_args.append(arg)
            else:
                converted_args.append(arg)

        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_name in params:
                if params[kwarg_name].annotation == Inference and not isinstance(kwarg_value, Inference):
                    try:
                        inference = cls._from_object(kwarg_value)
                        converted_kwargs[kwarg_name] = inference
                    except Exception:
                        raise TypeError(
                            f"Object of type {type(kwarg_value).__name__} could not be cast to `Inference`"
                        )
                else:
                    converted_kwargs[kwarg_name] = kwarg_value
            else:
                converted_kwargs[kwarg_name] = kwarg_value

        return tuple(converted_args), converted_kwargs

    @classmethod
    def _register_factory_method(cls, target_cls: type):
        """Register a factory method for a given inference client."""
        def decorator(function: Callable):
            cls._factory_registry[target_cls] = function
            return function
        return decorator

    @classmethod
    def _from_object(cls, obj: Any) -> Inference:
        """
        Create an `Inference` instance from a registered inference client.

        Args:
            obj: The client object from which to create the `Inference` instance.
        Returns:
            The newly created `Inference` instance.
        """
        obj_type = type(obj)
        factory_fn = cls._factory_registry[obj_type]
        inference = factory_fn(obj)
        return inference


@dataclass(slots=True)
class InferenceRequest:
    """
    Model class for inference requests.

    Attributes:
        model: The ID of the model from which to obtain the text completion.
        prompt: The prompt to pass to the inference API.
        attachments: The multimodal attachments to attach to the API request.
    """

    model: str
    prompt: str
    attachments: list[Any] | None


class Inference(ABC):
    """Abstract class that defines the interface for LLM inference backends."""

    def generate(
        self,
        model: str | None,
        prompt: str | None,
        attachments: list[Any] | None = None,
        request: InferenceRequest | None = None,
    ) -> str:
        """
        Generates a textual response using an LLM.

        Args:
            model: The ID of the model from which to obtain the text completion.
            prompt: The prompt to pass to the inference API.
            attachments: The multimodal attachments to attach to the API request.
            request: An `InferenceRequest` object containing the request details. If
                     this argument is passed in, all other arguments will be ignored.
        """
        return (
            self._get_completion(**request)
            if request is not None
            else self._get_completion(model, prompt, attachments)
        )

    @abstractmethod
    def _get_completion(model: str, prompt: str, attachments: list[Any] | None):
        """
        Makes a call to the inference backend.

        Args:
            model: The ID of the model from which to obtain the text completion.
            prompt: The prompt to pass to the inference API.
            attachments: The multimodal attachments to attach to the API request.
        Returns:
            The text content of the response body.
        """
        ...


class OpenAIInference(Inference):
    """
    Inference backend for the OpenAI API. Used for obtaining text completions from
    GPT-series models.
    """

    def __init__(self):
        try:
            from openai import OpenAI

            self.client = OpenAI()
        except ImportError:
            raise ImportError(
                "The `openai` library must be installed to use `OpenAIInference`. "
                "To install it, run the following command: `pip install openai`"
            )
        
    @classmethod
    @AutoInferenceMeta._register_factory_method(OpenAI)
    def _from_client(cls, client: OpenAI) -> Inference:
        inference = cls()
        inference.client = client
        return inference

    def _get_completion(
        self, model: str, prompt: str, attachments: list[Any] | None
    ) -> str:
        completion = (
            self.client.responses.create(model=model, input=prompt)
            .output[0]
            .content[0]
            .text
        )
        return completion
