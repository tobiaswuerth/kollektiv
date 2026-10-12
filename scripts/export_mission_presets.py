"""Regenerate provided portable presets from the canonical mission catalogue.

Run with ``uv run python scripts/export_mission_presets.py``. Output paths are
anchored to this script's repository, so invoking it from another directory
produces the same files. No Ollama connection or running server is needed.
"""

import json
from pathlib import Path

from kollektiv import Settings
from kollektiv.configurations import TeamConfiguration
from kollektiv.missions import mission_samples

PRESET_IDS = ("scientific-paper-lab", "fifty-page-novel-studio", "solo-orchestrator")
REPOSITORY = Path(__file__).resolve().parents[1]


def preset_configuration(sample: dict) -> dict:
    """Validate and expand sample overrides into a portable version 1 file."""
    configuration = TeamConfiguration.model_validate(
        {
            "version": 1,
            "name": sample["title"],
            "goal": sample["goal"],
            "settings": {**Settings().model_dump(), **sample["settings"]},
            "positions": sample["positions"],
        }
    )
    return configuration.model_dump(exclude_none=True)


def main() -> None:
    samples = {sample["id"]: sample for sample in mission_samples()}
    destination = REPOSITORY / "presets"
    destination.mkdir(parents=True, exist_ok=True)
    for sample_id in PRESET_IDS:
        path = destination / f"{sample_id}.json"
        content = json.dumps(preset_configuration(samples[sample_id]), ensure_ascii=False, indent=2)
        path.write_text(content + "\n", encoding="utf-8", newline="\n")
        print(path.relative_to(REPOSITORY))


if __name__ == "__main__":
    main()
