"""Contains all classes related to LLM inference."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
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
        attachments: list[Any] | None,
        request: InferenceRequest | None,
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

    def _get_completion(
        self, model: str, prompt: str, attachments: list[Any] | None
    ) -> str:
        completion = self.client.responses.create(model=model, input=prompt)
        return completion
