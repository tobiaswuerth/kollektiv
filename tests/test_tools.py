import json

import pytest

from kollektiv import Settings
from kollektiv.runtime import RunBusyError, run_lock
from kollektiv.tools import Toolkit, visible_messages
from kollektiv.workspace import Workspace, relative_path


@pytest.mark.parametrize(
    "name",
    [
        "../escape.md",
        "/absolute.md",
        "D:/escape.md",
        "x/../../escape.md",
        "x\\..\\escape.md",
        "\\\\server\\share\\file",
        "test.md:stream",
        "CON.md",
        "x/NUL",
        "a./file.md",
        "a//b",
        "./a",
    ],
)
def test_rejects_paths_leaving_workspace_and_windows_special_names(name):
    with pytest.raises(ValueError):
        relative_path(name)


def test_nested_atomic_artifacts_and_syntax_checks(tmp_path):
    workspace = Workspace(tmp_path / "artifacts")
    info = workspace.write("chapters/one.md", "# Chapter\nHello world")
    assert info["words"] == 4
    assert workspace.list() == ["chapters/one.md"]
    workspace.write("bad.py", "if broken")
    workspace.write("bad.json", "{oops}")
    assert workspace.inspect("bad.py")["problems"]
    assert workspace.inspect("bad.json")["problems"]
    assert workspace.write("valid.py", "print('hello')")["problems"] == []
    assert workspace.write("valid.json", '{"ok":true}')["problems"] == []
    with pytest.raises(ValueError, match="context_chars"):
        workspace.context(["chapters/one.md"], 3)
    assert not list(workspace.root.rglob(".kollektiv-*"))


def test_tool_capabilities_and_write_ownership(tmp_path):
    workspace = Workspace(tmp_path)
    settings = Settings()
    toolkit = Toolkit(settings, workspace, "builder", "own.md", lambda *_a, **_k: {})
    tools = {t.name: t for t in toolkit.build()}
    assert "web_search" not in tools
    with pytest.raises(ValueError):
        tools["write_file"].invoke({"path": "someone-elses.md", "content": "x"})
    with pytest.raises(ValueError, match="Unknown teammate"):
        tools["send_message"].invoke({"recipient": "invented", "content": "hi"})
    tools["send_message"].invoke({"recipient": "researcher", "content": "A useful decision"})
    assert len(visible_messages(toolkit.messages, "researcher")) == 1
    assert not visible_messages(toolkit.messages, "reviewer")
    assert json.loads(tools["write_file"].invoke({"content": "actual"}))["problems"] == []
    reviewer_tools = Toolkit(settings, workspace, "reviewer", None, lambda *_a, **_k: {}).build()
    assert "write_file" not in {t.name for t in reviewer_tools}
    settings.web_enabled = True
    assert "web_search" in {
        t.name
        for t in Toolkit(settings, workspace, "researcher", None, lambda *_a, **_k: {}).build()
    }


def test_workspace_rejects_symlink_escape(tmp_path):
    workspace = Workspace(tmp_path / "artifacts")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace.root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit creation of symlinks")
    with pytest.raises(ValueError, match="leaves"):
        workspace.write("link/escape.md", "cannot escape")
    assert not (outside / "escape.md").exists()


def test_run_lock_is_held_by_process_and_released(tmp_path):
    path = tmp_path / ".run.lock"
    with run_lock(path):
        with pytest.raises(RunBusyError):
            with run_lock(path):
                pass
    with run_lock(path):
        assert path.is_file()
