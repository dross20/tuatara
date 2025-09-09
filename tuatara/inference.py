from abc import ABC, abstractmethod

class Inference(ABC):
    @abstractmethod
    def generate(self, model: str, prompt: str) > str:
        ...

class OpenAIInference(Inference):
    def __init__(self):
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except:
            raise ImportError(
                "The `openai` library must be installed to use `OpenAIInference`. "
                "To install it, run the following command: `pip install openai`"
            )

    def generate(self, model: str, prompt: str) -> str:
        completion = self.client.responses.create(model=model, input=prompt)
        return completion