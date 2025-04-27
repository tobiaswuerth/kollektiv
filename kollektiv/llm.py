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
        retry_count=3,
    ) -> str:
        for attempt in range(retry_count):
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
                try:
                    return format.model_validate_json(response)
                except pydantic.ValidationError as e:
                    print(f'[WARNING] Validation error: {e}')
                    prompt += f"""
---------------------------------
!! SYSTEM ERROR, INVALID RESPONSE !!
!! Exception: {e}
!! Please try again (attempt {attempt+1} of {retry_count}).
"""
                    continue

            return response

        raise RuntimeError(
            f"Failed to generate response after {retry_count} attempts. {e=}, {prompt=}, {response=}, {format=}"
        )
