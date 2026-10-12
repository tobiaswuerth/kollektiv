"""Test the control room through HTTP and actual checkpointed graph operations."""

import json
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


def test_http_world_creation_live_step_admin_and_resume(control_room):
    client, manager = control_room
    page = client.get("/")
    assert page.status_code == 200 and "The commons" in page.text
    assert "GOD’S EYE" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200
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
