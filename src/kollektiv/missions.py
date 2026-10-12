"""Offline mission catalogue shared by the Python API and bundled frontend."""

import json
from importlib.resources import files
from typing import Literal, NotRequired, TypedDict, cast


class MissionSample(TypedDict):
    id: str
    title: str
    difficulty: Literal["easy", "medium", "elaborate"]
    description: str
    goal: str
    settings: dict
    positions: NotRequired[dict[str, dict[str, float]]]


def mission_samples() -> list[MissionSample]:
    """Fresh data preserves edited teams; the same catalogue works with older servers."""
    return cast(
        list[MissionSample],
        json.loads(files("kollektiv").joinpath("mission_samples.json").read_text(encoding="utf-8")),
    )
