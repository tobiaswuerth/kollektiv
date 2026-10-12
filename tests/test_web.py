"""Test the control room through HTTP and actual checkpointed graph operations."""

import json
import re
import threading
import time

import httpx
import pytest
from test_collective import ScriptedModels

from kollektiv.webapp import ControlRoomServer, RunManager


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    pytest.fail("The expected run boundary was not reached")


@pytest.fixture
def control_room(tmp_path):
    manager = RunManager(tmp_path, models=ScriptedModels())
    server = ControlRoomServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", timeout=10) as client:
        yield client, manager
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_http_room_creation_live_step_admin_and_resume(control_room):
    client, manager = control_room
    page = client.get("/")
    assert page.status_code == 200 and 'id="root"' in page.text
    assets = re.findall(r'(?:src|href)="(/(?:assets/[^\"]+|app.js|style.css))"', page.text)
    assert assets
    assert all(client.get(path).status_code == 200 for path in assets)
    assert client.get("/api/bootstrap").json()["settings"]["model"] == "gemma4:e2b"
    response = client.post(
        "/api/runs",
        json={
            "goal": "Create design.md and a consistent story.md",
            "settings": {"discussion_rounds": 1},
        },
    )
    assert response.status_code == 201
    run_id = response.json()["manifest"]["run_id"]
    path = f"/api/runs/{run_id}"
    assert client.get("/api/runs").json()["runs"][0]["run_id"] == run_id
    step = client.post(path + "/control", json={"action": "step"})
    assert step.status_code == 200, step.text
    managed = manager.get(run_id)
    eventually(
        lambda: managed.control.snapshot()["paused"] and managed.control.snapshot()["waiting"]
    )
    snapshot = client.get(path).json()
    assert snapshot["running"]
    assert len([event for event in snapshot["events"] if event["kind"] == "node_finished"]) == 1
    assert snapshot["state"]["status"] == "discussing"
    cursor = snapshot["cursor"]
    assert client.get(path + f"?cursor={cursor}").json()["events"] == []

    response = client.post(
        path + "/admin",
        json={"action": "inject_message", "recipient": "builder", "content": "ADMIN CONTEXT"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["control"]["paused"]
    assert any(
        message["content"] == "ADMIN CONTEXT"
        for message in response.json()["state"]["messages"].values()
    )
    assert client.post(path + "/control", json={"action": "resume"}).status_code == 200
    eventually(lambda: not managed.running)
    final = client.get(path).json()
    assert not final["error"]
    assert final["state"]["status"] == "completed"
    assert final["control"]["mode"] == "finished"
    assert not final["control"]["running"]
    assert {file["path"] for file in final["artifacts"]} == {"design.md", "story.md"}
    artifact = client.get(path + "/artifact", params={"path": "story.md"})
    assert artifact.status_code == 200 and "COMPLETE" in artifact.json()["content"]
    assert any("ADMIN CONTEXT" in prompt for _, _, prompt in manager.models.prompts)


def test_http_validation_and_artifact_confinement(control_room):
    client, _ = control_room
    assert client.post("/api/runs", json={"goal": ""}).status_code == 400
    assert (
        client.post("/api/runs", json={"goal": "x", "settings": {"max_concurrency": 0}}).status_code
        == 400
    )
    assert (
        client.post(
            "/api/runs", json={"goal": "x"}, headers={"Origin": "https://other.example"}
        ).status_code
        == 403
    )
    assert client.get("/api/runs/no-such-run").status_code == 404
    created = client.post("/api/runs", json={"goal": "A goal"}).json()
    path = f"/api/runs/{created['manifest']['run_id']}"
    assert client.get(path + "/artifact", params={"path": "../run.json"}).status_code == 400
    assert client.post(path + "/control", json={"action": "unknown"}).status_code == 400
    assert client.get("/src/kollektiv/runtime.py").status_code == 404


def test_partial_event_record_is_not_lost(tmp_path):
    first = {"id": "first", "kind": "node_started"}
    second = {"id": "second", "kind": "node_finished"}
    path = tmp_path / "events.jsonl"
    path.write_bytes((json.dumps(first) + "\n" + json.dumps(second)).encode())
    events, cursor = RunManager.read_events(tmp_path)
    assert events == [first]
    with path.open("ab") as stream:
        stream.write(b"\n")
    remaining, cursor = RunManager.read_events(tmp_path, cursor)
    assert remaining == [second]
    assert cursor == path.stat().st_size


def test_fast_event_feed_preserves_bursts_without_reading_checkpoint(control_room, monkeypatch):
    client, manager = control_room
    created = client.post("/api/runs", json={"goal": "An event transport test"}).json()
    run_id = created["manifest"]["run_id"]
    collective = manager.get(run_id).collective

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Fast polling must not load graph state")

    monkeypatch.setattr(collective, "state", unexpected_read)
    expected = []
    for _ in range(40):
        for kind in ("node_started", "tool_result", "node_finished"):
            expected.append(collective.emit(kind, agent="builder", tool="write_file")["id"])
    response = client.get(f"/api/runs/{run_id}/events")
    assert response.status_code == 200, response.text
    batch = response.json()
    assert [e["id"] for e in batch["events"]] == expected
    assert client.get(f"/api/runs/{run_id}/events?cursor={batch['cursor']}").json()["events"] == []


def test_saved_team_configuration_and_run_layout(control_room):
    client, manager = control_room
    settings = client.get("/api/bootstrap").json()["settings"]
    settings["communication_links"] = [{"source": "builder", "target": "researcher"}]
    positions = {"builder": {"x": 40, "y": 220}}
    payload = {
        "version": 1,
        "name": "Focused team",
        "goal": "Create a story",
        "settings": settings,
        "positions": positions,
    }
    saved = client.post("/api/configurations", json=payload)
    assert saved.status_code == 201, saved.text
    configuration_id = saved.json()["id"]
    assert client.get("/api/configurations").json()["configurations"][0] == saved.json()
    assert client.get(f"/api/configurations/{configuration_id}").json() == saved.json()
    updated = client.post("/api/configurations", json={**saved.json(), "name": "Renamed team"})
    assert updated.status_code == 201
    assert len(manager.configurations.list()) == 1
    created = client.post(
        "/api/runs", json={"goal": payload["goal"], "settings": settings, "positions": positions}
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["manifest"]["run_id"]
    assert created.json()["manifest"]["positions"] == positions
    reopened = RunManager(manager.directory, models=manager.models)
    assert reopened.snapshot(run_id)["manifest"]["positions"] == positions
    assert reopened.configurations.get(configuration_id)["name"] == "Renamed team"
    refused = client.delete(
        f"/api/configurations/{configuration_id}", headers={"Origin": "https://other.example"}
    )
    assert refused.status_code == 403
    assert client.delete(f"/api/configurations/{configuration_id}").status_code == 200
    assert client.get("/api/configurations").json()["configurations"] == []
    assert client.get("/api/configurations/not-an-id").status_code == 400


def test_bootstrap_mission_samples_and_invalid_layout(control_room):
    client, _ = control_room
    data = client.get("/api/bootstrap").json()
    assert {sample["difficulty"] for sample in data["mission_samples"]} == {
        "easy",
        "medium",
        "elaborate",
    }
    assert "write_file" in data["tools"]
    response = client.post(
        "/api/runs", json={"goal": "Invalid layout", "positions": {"ghost": {"x": 0, "y": 0}}}
    )
    assert response.status_code == 400
    assert client.get("/api/runs").json()["runs"] == []


def test_run_selector_uses_creation_time_not_random_id_suffix(control_room):
    client, manager = control_room
    ids = sorted(
        client.post("/api/runs", json={"goal": "Sorting test"}).json()["manifest"]["run_id"]
        for _ in range(2)
    )
    for run_id, created_at in zip(
        ids, ["2026-10-12T10:00:00.900000+00:00", "2026-10-12T10:00:00.100000+00:00"], strict=True
    ):
        collective = manager.get(run_id).collective
        collective.manifest["created_at"] = created_at
        (collective.directory / "run.json").write_text(
            json.dumps(collective.manifest), encoding="utf-8"
        )
    assert [run["run_id"] for run in client.get("/api/runs").json()["runs"]] == ids
