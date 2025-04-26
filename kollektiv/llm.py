import ollama


class LLMClient:
    def __init__(self, model_name: str = "gemma3:12b") -> None:
        self.model_name = model_name

    def generate(self, prompt: str, format=None, num_predict=1024) -> str:
        response = ollama.generate(
            self.model_name,
            prompt=prompt,
            format=format,
            stream=False,
            options={
                "num_predict": num_predict,
                "temperature": 0.5,
                "top_p": 0.9,
            },
        )
        return response["response"]
