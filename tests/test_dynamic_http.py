"""Exercise live supervision through an isolated HTTP server and real worker graphs."""

import threading
from contextlib import contextmanager

import httpx
from test_dynamic_workflow import MANAGER_TOOLS, FamilyModels
from test_web import eventually

from kollektiv import Settings
from kollektiv.webapp import ControlRoomServer, RunManager


@contextmanager
def dynamic_room(tmp_path, models):
    manager = RunManager(tmp_path, models=models)
    server = ControlRoomServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", timeout=10) as client:
            settings = Settings(max_concurrency=1, max_steps=64, discussion_rounds=1)
            settings.team = [agent for agent in settings.team if agent.name != "researcher"]
            settings.agent("builder").tools += MANAGER_TOOLS
            created = client.post(
                "/api/runs",
                json={
                    "goal": "Produce an actual supervised outcome.",
                    "settings": settings.model_dump(),
                },
            )
            assert created.status_code == 201, created.text
            run_id = created.json()["manifest"]["run_id"]
            managed = manager.get(run_id)
            models.manager = managed.collective.delegation
            yield client, manager, managed, f"/api/runs/{run_id}"
    finally:
        models.release.set()
        manager.close()
        for managed in list(manager._runs.values()):
            if managed.thread:
                managed.thread.join(5)
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_live_http_instruction_preserves_running_control_and_changes_child_next_prompt(tmp_path):
    models = FamilyModels(hold="assistant-before-write")
    with dynamic_room(tmp_path, models) as (client, _, managed, path):
        managed.collective.workspace.write("seed.md", "COMPLETE source for live instructions.")
        response = client.post(path + "/control", json={"action": "start"})
        assert response.status_code == 200, response.text
        assert models.entered.wait(10)
        active_control = managed.control
        response = client.post(
            path + "/admin",
            json={
                "action": "instruct_agent",
                "agent_id": models.manager.get("assistant")["id"],
                "content": "CORRECTED ASSUMPTION: mark the result ADJUSTED after this call.",
            },
        )
        assert response.status_code == 200, response.text
        snapshot = response.json()
        assert snapshot["running"]
        assert snapshot["control"]["mode"] == "running"
        assert managed.control is active_control
        assert snapshot["delegation"]["agents"]["assistant"]["interventions"]
        models.release.set()
        eventually(lambda: not managed.running, timeout=15)
        final = client.get(path).json()
        assert managed.error is None
        assert final["state"]["status"] == "completed"
        assert "ADJUSTED" in managed.collective.workspace.read("combined.md")
        assert any(event["kind"] == "agent_instruction" for event in final["events"])


def test_live_http_fire_retires_family_and_keeps_completed_files_without_replacing_control(
    tmp_path,
):
    models = FamilyModels(nested=True, hold="leaf-after-write")
    with dynamic_room(tmp_path, models) as (client, _, managed, path):
        response = client.post(path + "/control", json={"action": "start"})
        assert response.status_code == 200, response.text
        assert models.entered.wait(10)
        child = models.manager.get("leaf")
        original = managed.collective.workspace.read(child["output"])
        active_control = managed.control
        response = client.post(
            path + "/admin",
            json={
                "action": "fire_agent",
                "agent_id": models.manager.get("assistant")["id"],
                "reason": "Cancel this family.",
            },
        )
        assert response.status_code == 200, response.text
        assert managed.control is active_control
        jobs = response.json()["delegation"]["agents"]
        assert jobs["assistant"]["status"] == jobs["leaf"]["status"] == "retired"
        models.release.set()
        eventually(lambda: not managed.running, timeout=15)
        assert managed.error is None
        assert managed.collective.workspace.read(child["output"]) == original
        assert "FORBIDDEN" not in original
        final = client.get(path).json()
        assert final["state"]["status"] == "completed"
        assert any(event["kind"] == "agent_fired" for event in final["events"])
