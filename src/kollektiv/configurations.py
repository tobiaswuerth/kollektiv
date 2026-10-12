"""Validated, atomic, portable team presets independent of execution checkpoints."""

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import Settings
from .workspace import atomic_write


class NodePosition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float = Field(ge=-100_000, le=100_000, allow_inf_nan=False)
    y: float = Field(ge=-100_000, le=100_000, allow_inf_nan=False)


def validate_positions(positions: dict | None, settings: Settings) -> dict[str, dict]:
    positions = {} if positions is None else positions
    names = {agent.name for agent in settings.team}
    if not isinstance(positions, dict) or set(positions) - names:
        raise ValueError("Graph positions must name members of the team")
    return {
        name: NodePosition.model_validate(point).model_dump() for name, point in positions.items()
    }


class TeamConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    id: str | None = None
    name: str = Field(min_length=1, max_length=120)
    goal: str = ""
    settings: Settings
    positions: dict[str, NodePosition] = Field(default_factory=dict)
    updated_at: str | None = None

    @model_validator(mode="after")
    def validate_configuration(self) -> "TeamConfiguration":
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("Configuration name cannot be empty")
        validate_positions(
            {key: value.model_dump() for key, value in self.positions.items()}, self.settings
        )
        if self.id is not None and not re.fullmatch(r"[a-f0-9]{32}", self.id):
            raise ValueError("Invalid configuration ID")
        return self


class ConfigurationStore:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self._lock = threading.RLock()

    def path(self, configuration_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", configuration_id):
            raise ValueError("Invalid configuration ID")
        path = (self.directory / f"{configuration_id}.json").resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Configuration leaves its directory")
        return path

    def list(self) -> list[dict]:
        with self._lock:
            results = []
            for path in self.directory.glob("*.json"):
                try:
                    if path.resolve().is_relative_to(self.directory):
                        config = TeamConfiguration.model_validate_json(
                            path.read_text(encoding="utf-8")
                        )
                        if config.id == path.stem:
                            results.append(config.model_dump())
                except (ValueError, OSError):
                    continue
            return sorted(results, key=lambda item: item["updated_at"] or "", reverse=True)

    def get(self, configuration_id: str) -> dict:
        with self._lock:
            return TeamConfiguration.model_validate_json(
                self.path(configuration_id).read_text(encoding="utf-8")
            ).model_dump()

    def save(self, payload: dict) -> dict:
        config = TeamConfiguration.model_validate(payload)
        config.id = config.id or uuid4().hex
        config.updated_at = datetime.now(UTC).isoformat()
        result = config.model_dump()
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            atomic_write(self.path(config.id), json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def delete(self, configuration_id: str) -> None:
        with self._lock:
            self.path(configuration_id).unlink()
