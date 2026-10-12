import itertools
import json
from pathlib import Path

import pytest

from kollektiv import Settings
from kollektiv.cli import read_config
from kollektiv.configurations import TeamConfiguration
from kollektiv.missions import mission_samples


def test_samples_cover_difficulty_tiers_and_preserve_local_model_configuration():
    samples = mission_samples()
    assert len({sample["id"] for sample in samples}) == len(samples)
    for difficulty in ("easy", "medium", "elaborate"):
        assert len([sample for sample in samples if sample["difficulty"] == difficulty]) >= 2
    custom = Settings(model="my-local-model", max_concurrency=3)
    for sample in samples:
        merged = Settings.model_validate({**custom.model_dump(), **sample["settings"]})
        assert merged.model == custom.model
        assert merged.base_url == custom.base_url
        assert merged.max_concurrency == custom.max_concurrency
        if "team" not in sample["settings"]:
            assert merged.team == custom.team
            assert merged.max_tasks < custom.max_tasks
        assert sample["goal"] and sample["description"]


def test_sample_copies_are_isolated_from_bootstrap_consumers():
    edited = mission_samples()
    edited[0]["settings"]["max_tasks"] = 200
    edited[0]["goal"] = "Changed by a consumer"
    original = mission_samples()[0]
    assert original["settings"]["max_tasks"] == 3
    assert original["goal"] != edited[0]["goal"]


def ambitious_sample(sample_id):
    return next(sample for sample in mission_samples() if sample["id"] == sample_id)


@pytest.mark.parametrize(
    ("sample_id", "agent_count", "publisher", "expected_web"),
    [
        ("scientific-paper-lab", 12, "manuscript_editor", True),
        ("fifty-page-novel-studio", 14, "book_designer", False),
    ],
)
def test_ambitious_samples_load_complete_teams_with_scoped_tools(
    sample_id, agent_count, publisher, expected_web
):
    sample = ambitious_sample(sample_id)
    settings = Settings.model_validate({**Settings().model_dump(), **sample["settings"]})
    assert len(settings.team) == agent_count
    assert len({agent.role for agent in settings.team}) == agent_count
    assert settings.web_enabled == expected_web
    assert {"write_file", "assemble_files"} <= set(settings.agent(publisher).tools)
    assert "write_file" not in settings.agent(settings.reviewer).tools
    assert "assemble_files" not in settings.agent(settings.reviewer).tools
    assert "write_file" not in settings.agent(settings.coordinator).tools
    for agent in settings.team:
        assert len(agent.persona) > 200
        assert len(agent.skills) >= 2
        assert {"read_file", "send_message", "share_note"} <= set(agent.tools)
        if agent.name not in (settings.coordinator, settings.reviewer):
            assert "write_file" in agent.tools
    if not expected_web:
        assert all(not tool.startswith("web_") for agent in settings.team for tool in agent.tools)


@pytest.mark.parametrize("sample_id", ["scientific-paper-lab", "fifty-page-novel-studio"])
def test_ambitious_channels_allow_handoffs_and_feedback_without_all_to_all_messaging(sample_id):
    sample = ambitious_sample(sample_id)
    settings = Settings.model_validate({**Settings().model_dump(), **sample["settings"]})
    links = settings.communication_links
    assert links is not None
    names = {agent.name for agent in settings.team}
    assert len(links) < len(names) * (len(names) - 1) / 2
    for sender in names:
        visited = {sender}
        pending = [sender]
        while pending:
            current = pending.pop()
            for target in settings.message_recipients(current, "all"):
                if target not in visited:
                    visited.add(target)
                    pending.append(target)
        assert visited == names, f"{sender} cannot reach every specialist through handoff channels"


@pytest.mark.parametrize("sample_id", ["scientific-paper-lab", "fifty-page-novel-studio"])
def test_ambitious_graph_positions_cover_every_agent_without_overlapping_cards(sample_id):
    sample = ambitious_sample(sample_id)
    names = {agent["name"] for agent in sample["settings"]["team"]}
    positions = sample["positions"]
    assert set(positions) == names
    for name, position in positions.items():
        assert isinstance(position["x"], (int, float))
        assert isinstance(position["y"], (int, float))
        assert position["x"] >= 0 and position["y"] >= 0, name
    for first, second in itertools.combinations(positions.values(), 2):
        assert abs(first["x"] - second["x"]) >= 214 or abs(first["y"] - second["y"]) >= 146


def test_novel_contract_requires_a_complete_book_and_budgeted_atomic_chapter_tasks():
    sample = ambitious_sample("fifty-page-novel-studio")
    goal = sample["goal"]
    assert "12,600 to 14,400 words" in goal
    assert "700 to 800 words" in goal
    assert "250 to 300 words per page" in goal
    assert "no guaranteed physical page count" in goal
    assert "ALL eighteen accepted final chapters" in goal
    assert "assemble_files" in goal
    for number in range(1, 19):
        assert f"chapters/chapter_{number:02d}.md" in goal
    # Seven foundation files, 18 drafts, 18 final chapters, front matter, book, and README.
    assert sample["settings"]["max_tasks"] >= 46
    assert sample["settings"]["context_chars"] >= 350_000
    assert sample["settings"]["num_ctx"] >= 131_072
    assert sample["settings"]["num_predict"] >= 32_768
    assert "no more than 5,000 words combined" in goal
    assert "each no more than 1,200 words" in goal


def test_scientific_contract_requires_evidence_and_disallows_fake_execution_or_data():
    sample = ambitious_sample("scientific-paper-lab")
    goal = sample["goal"]
    assert "4,500 to 6,500 words" in goal
    for path in (
        "research/evidence.json",
        "research/proofs.md",
        "research/counterexamples.md",
        "research/limitations.md",
        "sections/08-references.md",
        "reproducibility/examples.json",
        "reproducibility/analyze.py",
        "paper.md",
    ):
        assert path in goal
    assert "at least five primary sources" in goal
    assert "NOT EXECUTED" in goal
    assert "No fabricated data" in goal
    assert "assemble_files" in goal
    assert sample["settings"]["num_predict"] >= 16_384
    assert "no more than 1,000 words" in goal
    assert "no more than 6,000 words" in goal


def test_ambitious_sample_team_and_layout_copies_are_isolated():
    sample = ambitious_sample("fifty-page-novel-studio")
    sample["settings"]["team"][0]["persona"] = "Changed"
    sample["settings"]["communication_links"].clear()
    sample["positions"]["showrunner"]["x"] = -1
    original = ambitious_sample("fifty-page-novel-studio")
    assert original["settings"]["team"][0]["persona"] != "Changed"
    assert original["settings"]["communication_links"]
    assert original["positions"]["showrunner"]["x"] >= 0


@pytest.mark.parametrize("sample_id", ["scientific-paper-lab", "fifty-page-novel-studio"])
def test_portable_presets_match_canonical_samples_and_support_editor_and_cli_import(sample_id):
    sample = ambitious_sample(sample_id)
    path = Path(__file__).resolve().parents[1] / "presets" / f"{sample_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"version", "name", "goal", "settings", "positions"}
    expected = TeamConfiguration.model_validate(
        {
            "version": 1,
            "name": sample["title"],
            "goal": sample["goal"],
            "settings": {**Settings().model_dump(), **sample["settings"]},
            "positions": sample["positions"],
        }
    )
    imported = TeamConfiguration.model_validate(payload)
    assert imported.model_dump() == expected.model_dump()
    assert payload == expected.model_dump(exclude_none=True)
    assert imported.id is None and imported.updated_at is None
    assert read_config(str(path)) == expected.settings.model_dump()
