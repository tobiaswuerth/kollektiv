from pydantic import BaseModel
import os

from .llm import Storage


def save_pydantic_json(obj: BaseModel, file_name: str) -> None:
    path = os.path.join(Storage.directory, file_name)
    os.makedirs(Storage.directory, exist_ok=True)
    content = obj.model_dump_json(indent=2)
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def load_pydantic_json(file_name: str, model: type) -> BaseModel:
    path = os.path.join(Storage.directory, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File '{file_name}' not found in '{Storage.directory}'."
        )
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    return model.model_validate_json(content, strict=True)
