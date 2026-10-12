import json

import pytest
from test_collective import ScriptedModels

from kollektiv import Collective, Settings
from kollektiv.models import Contribution, MessageDraft
from kollektiv.tools import Toolkit, visible_messages
from kollektiv.workspace import Workspace


class RoutedModels(ScriptedModels):
    def structured(self, agent, schema, prompt, validate=None):
        if schema is not Contribution:
            return super().structured(agent, schema, prompt, validate)
        self.prompts.append((agent, schema.__name__, prompt))
        encoded = prompt.split('"allowed_message_recipients":', 1)[1].lstrip()
        recipients = json.JSONDecoder().raw_decode(encoded)[0]
        result = Contribution(
            findings=f"{agent} checks complete artifacts.",
            messages=[MessageDraft(recipient="all", content=f"HELLO FROM {agent}")]
            if recipients
            else [],
        )
        if validate:
            validate(result)
        return result


def test_broadcast_follows_paths_and_rejected_delivery_leaves_no_message(tmp_path):
    settings = Settings(communication_links=[{"source": "builder", "target": "researcher"}])
    emitted = []
    toolkit = Toolkit(
        settings,
        Workspace(tmp_path),
        "builder",
        "story.md",
        lambda kind, **fields: emitted.append((kind, fields)),
    )
    send = next(tool for tool in toolkit.build() if tool.name == "send_message")
    send.invoke({"recipient": "all", "content": "Only the researcher should receive this."})
    assert len(emitted) == 1
    assert len(visible_messages(toolkit.messages, "researcher")) == 1
    assert not visible_messages(toolkit.messages, "reviewer")
    assert not visible_messages(toolkit.messages, "coordinator")
    before = toolkit.messages.copy()
    with pytest.raises(ValueError, match="cannot send"):
        send.invoke({"recipient": "reviewer", "content": "Forbidden"})
    assert toolkit.messages == before


@pytest.mark.parametrize("disable_tool", [False, True])
def test_zero_paths_and_disabled_messages_still_complete_reviewed_work(tmp_path, disable_tool):
    settings = Settings(communication_links=[])
    if disable_tool:
        for agent in settings.team:
            agent.tools = [tool for tool in agent.tools if tool != "send_message"]
    run = Collective.create(
        "Create design.md and story.md", tmp_path, settings, models=RoutedModels()
    )
    final = run.run()
    assert final["status"] == "completed"
    assert final["contributions"]
    assert final["messages"]
    assert all(message.get("system") for message in final["messages"].values())
    assert all(task["status"] == "completed" for task in final["tasks"].values())


def test_directional_graph_delivers_agent_messages_and_mandatory_review_feedback(tmp_path):
    links = [
        {"source": "builder", "target": "researcher"},
        {"source": "researcher", "target": "builder"},
    ]
    run = Collective.create(
        "Create design.md and story.md",
        tmp_path,
        Settings(communication_links=links),
        models=RoutedModels(),
    )
    final = run.run()
    assert final["status"] == "completed"
    optional = [message for message in final["messages"].values() if not message.get("system")]
    assert optional
    assert all(
        {"source": message["sender"], "target": message["recipient"]} in links
        for message in optional
    )
    assert any(
        message.get("system")
        and message["sender"] == "reviewer"
        and message["recipient"] == "builder"
        for message in final["messages"].values()
    )
