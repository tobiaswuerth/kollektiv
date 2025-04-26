import ollama
import pydantic
import random


class LLMClient:
    def __init__(self, model_name: str = "phi4:latest") -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        format: pydantic.BaseModel = None,
        num_predict=1024,
    ) -> str:
        response = ollama.generate(
            self.model_name,
            prompt=prompt,
            format=format.model_json_schema() if format else None,
            stream=False,
            options={
                "num_predict": num_predict,
                "temperature": 0.5,
                "top_p": 0.9,
                "num_ctx": 10000,
                "seed": random.randint(0, 2**30 - 1),
            },
        )
        response = response["response"]

        if format:
            return format.model_validate_json(response)
        return response
