from typing import NamedTuple
import yaml
import os


CONFIG_FILE = "config.yaml"


class Config(NamedTuple):
    output_dir: str
    valid_tool_output_extensions: list[str]
    brave_search_api_key: str


def _load_global_config() -> Config:
    assert os.path.exists(CONFIG_FILE), f"Config file '{CONFIG_FILE}' not found."
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    return Config(**config_data)


config: Config = _load_global_config()
