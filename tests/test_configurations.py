import json

import pytest

from kollektiv.cli import read_config
from kollektiv.config import Settings
from kollektiv.configurations import ConfigurationStore, validate_positions


@pytest.mark.parametrize(
    "links",
    [
        [{"source": "builder", "target": "ghost"}],
        [{"source": "builder", "target": "builder"}],
        [{"source": "builder", "target": "researcher"}] * 2,
    ],
)
def test_rejects_invalid_communication_topology(links):
    with pytest.raises(ValueError):
        Settings(communication_links=links)


def test_directional_messages_and_permissions():
    settings = Settings(communication_links=[{"source": "builder", "target": "researcher"}])
    assert settings.message_recipients("builder", "all") == ["researcher"]
    assert settings.message_recipients("builder", "researcher") == ["researcher"]
    with pytest.raises(ValueError, match="cannot send"):
        settings.message_recipients("researcher", "builder")
    with pytest.raises(ValueError, match="cannot send"):
        settings.message_recipients("builder", "reviewer")
    settings.agent("builder").tools.remove("send_message")
    assert settings.allowed_recipients("builder") == []
    with pytest.raises(ValueError, match="cannot send"):
        settings.message_recipients("builder", "researcher")
    assert Settings().message_recipients("builder", "all") == ["all"]


def test_configuration_restart_roundtrip_update_and_delete(tmp_path):
    store = ConfigurationStore(tmp_path / "presets")
    settings = Settings(communication_links=[])
    settings.agent("builder").role = "Editor"
    settings.agent("builder").persona = "Use concise language and verify every file."
    saved = store.save(
        {
            "name": "Writing team",
            "goal": "Create story.md",
            "settings": settings.model_dump(),
            "positions": {"builder": {"x": -32, "y": 140}},
        }
    )
    restarted = ConfigurationStore(store.directory)
    assert restarted.list() == [saved]
    assert restarted.get(saved["id"])["settings"]["team"][2]["role"] == "Editor"
    saved["goal"] = "Create a better story.md"
    updated = restarted.save(saved)
    assert updated["id"] == saved["id"]
    assert restarted.get(saved["id"])["goal"] == saved["goal"]
    assert len(restarted.list()) == 1
    restarted.delete(saved["id"])
    assert restarted.list() == []
    with pytest.raises(FileNotFoundError):
        restarted.get(saved["id"])


@pytest.mark.parametrize("bad_id", ["../escape", "..", "D:/escape", "", "name.json"])
def test_configuration_ids_cannot_escape_directory(tmp_path, bad_id):
    store = ConfigurationStore(tmp_path)
    with pytest.raises(ValueError):
        store.get(bad_id)
    with pytest.raises(ValueError):
        store.save({"id": bad_id, "name": "x", "settings": Settings().model_dump()})


@pytest.mark.parametrize(
    "positions",
    [
        {"ghost": {"x": 0, "y": 0}},
        {"builder": {"x": float("nan"), "y": 0}},
        [],
        {"builder": {"x": 0, "y": 1, "extra": 1}},
    ],
)
def test_positions_are_finite_and_belong_to_team(positions):
    with pytest.raises(ValueError):
        validate_positions(positions, Settings())


def test_cli_accepts_exported_team_configuration_and_raw_overrides(tmp_path):
    path = tmp_path / "team.json"
    settings = Settings(communication_links=[]).model_dump()
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "Exported team",
                "goal": "Write story.md",
                "settings": settings,
                "positions": {},
            }
        ),
        encoding="utf-8",
    )
    assert read_config(str(path)) == settings
    path.write_text('{"model":"other-local-model"}', encoding="utf-8-sig")
    assert read_config(str(path)) == {"model": "other-local-model"}
