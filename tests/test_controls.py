import threading
from collections import Counter

import pytest
from test_collective import ScriptedModels, make_run

from kollektiv import Collective, RunControl
from kollektiv.runtime import RunBusyError


def launch(run, control, events):
    results, errors = [], []

    def target():
        try:
            results.append(run.run(control=control, on_event=events.append))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, results, errors


def test_step_advances_one_node_and_stop_resumes_worker_without_rewriting(tmp_path):
    models = ScriptedModels()
    run = make_run(tmp_path, models)
    control = RunControl(start_paused=True)
    events = []
    thread, results, errors = launch(run, control, events)
    try:
        assert control.wait_until_paused(5)
        assert not [e for e in events if e["kind"] == "node_started"]
        control.step()
        assert control.wait_until_paused(5)
        started = [e for e in events if e["kind"] == "node_started"]
        assert [e["node"] for e in started] == ["begin"]
        assert models.calls["Contribution"] == 0
        for _ in range(30):
            before = len([e for e in events if e["kind"] == "node_started"])
            control.step()
            assert control.wait_until_paused(5)
            after = len([e for e in events if e["kind"] == "node_started"])
            assert after == before + 1
            if events and any(
                e["kind"] == "node_finished" and e["node"] == "worker.act" for e in events
            ):
                break
        else:
            pytest.fail("Did not reach worker.act")
        assert not run.workspace.path("design.md").exists()
        control.step()
        assert control.wait_until_paused(5)
        assert run.workspace.path("design.md").is_file()
        assert models.calls["TaskReview"] == 0
        inspection = run.state(include_workers=True)
        assert len(inspection["worker_states"]) == 1
        worker = inspection["worker_states"][0]
        assert worker["values"]["task"]["id"] == "design"
        assert worker["next"] == ["act"]
        assert any(m["data"].get("name") == "write_file" for m in worker["values"]["history"])
    finally:
        control.stop()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert results[0]["status"] == "stopped"
    assert Collective.open(run.directory, models=models).run()["status"] == "completed"
    assert models.writes == {"design": 1, "story": 1}


def test_admin_message_stops_paused_runner_and_enters_checkpoint(tmp_path):
    models = ScriptedModels()
    run = make_run(tmp_path, models)
    control = RunControl(start_paused=True)
    thread, _, errors = launch(run, control, [])
    try:
        assert control.wait_until_paused(5)
        state = run.admin(
            "inject_message",
            control=control,
            recipient="builder",
            content="OPERATOR: preserve a quiet robot character.",
        )
        thread.join(5)
        assert control.snapshot()["stopped"]
        assert not thread.is_alive() and not errors
        assert any(
            m["sender"] == "admin" and m["recipient"] == "builder"
            for m in state["messages"].values()
        )
        assert run.run()["status"] == "completed"
        assert any(
            "OPERATOR: preserve" in prompt
            for agent, _, prompt in models.prompts
            if agent == "builder"
        )
    finally:
        control.stop()
        thread.join(5)


def test_admin_requires_active_model_call_to_finish(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class SlowModels(ScriptedModels):
        def act(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().act(*args, **kwargs)

    run = make_run(tmp_path, SlowModels())
    control = RunControl()
    thread, _, errors = launch(run, control, [])
    try:
        assert entered.wait(5)
        control.pause()
        with pytest.raises(RunBusyError, match="current node"):
            run.admin("share_note", control=control, content="Should not race")
        release.set()
        assert control.wait_until_paused(5)
        state = run.admin("share_note", control=control, content="Paused editing is safe")
        assert any(n["content"] == "Paused editing is safe" for n in state["notes"].values())
    finally:
        release.set()
        control.stop()
        thread.join(5)
    assert not errors


def test_admin_reassignment_and_artifact_edits_reopen_dependencies(tmp_path):
    models = ScriptedModels()
    run = make_run(tmp_path, models)
    run.run()
    with pytest.raises(ValueError, match="write_file"):
        run.admin("reassign_task", task_id="design", owner="reviewer")
    state = run.admin("reassign_task", task_id="design", owner="builder")
    assert state["tasks"]["design"]["owner"] == "builder"
    assert state["tasks"]["story"]["status"] == "pending"
    assert state["tasks"]["design"]["attempts"] == 0
    assert state["plan"]["tasks"][0]["owner"] == "builder"
    state = run.admin("replace_artifact", path="design.md", content="# COMPLETE operator draft")
    assert "preserve their changes" in state["tasks"]["design"]["feedback"]
    assert run.workspace.read("design.md") == "# COMPLETE operator draft"
    assert run.run()["status"] == "completed"
    assert models.writes == Counter(design=2, story=2)
    state = run.admin("retry_task", task_id="story")
    assert state["tasks"]["design"]["status"] == "completed"
    assert state["tasks"]["story"]["status"] == "pending"


def test_pre_start_goal_config_and_seed_memory_edits(tmp_path):
    models = ScriptedModels()
    run = make_run(tmp_path, models)
    run.admin(
        "revise_goal", goal="The operator's corrected goal", settings={"discussion_rounds": 1}
    )
    assert run.settings.discussion_rounds == 1
    run.admin("share_note", content="SEED MEMORY")
    run.admin("inject_message", recipient="builder", content="SEED MESSAGE")
    resumed = Collective.open(run.directory, models=models)
    state = resumed.run()
    assert state["goal"] == "The operator's corrected goal"
    assert any(n["content"] == "SEED MEMORY" for n in state["notes"].values())
    assert any(
        "SEED MESSAGE" in prompt for agent, _, prompt in models.prompts if agent == "builder"
    )
    with pytest.raises(ValueError, match="before starting"):
        resumed.admin("revise_goal", goal="Cannot change an executed goal")


def test_fresh_run_can_be_inspected_concurrently_during_database_initialization(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    run = make_run(tmp_path, ScriptedModels())
    control = RunControl(start_paused=True)
    thread, _, errors = launch(run, control, [])
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: run.state(include_workers=True), range(18)))
        assert all(isinstance(state, dict) for state in results)
        assert control.wait_until_paused(5)
        control.step()
        assert control.wait_until_paused(5)
        assert run.state()["status"] == "discussing"
    finally:
        control.stop()
        thread.join(5)
    assert not errors
