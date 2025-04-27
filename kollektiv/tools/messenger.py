import pydantic
from dataclasses import dataclass
import datetime

from .tool import (
    Tool,
    Function,
    ResponseStatus,
    RESPONSE_OK,
)


@dataclass
class Message:
    timestamp: datetime.datetime
    sender: object
    recipients: list[str]
    content: str

class SendMessageInput(pydantic.BaseModel):
    agent_ids: list[int]
    message: str


class SendMessageOutput(pydantic.BaseModel):
    status: ResponseStatus


class Messenger(Tool):

    def __init__(self, agents):
        name = "Messenger"
        description = "A tool to chat with team members."
        super().__init__(name, description)

        self.agents = agents

        self.register_function(
            Function(
                name="send_message",
                description="Get a list of all currently stored files.",
                func=self.send_message,
            )
        )

    def send_message(
        self, agent, input_: SendMessageInput
    ) -> SendMessageOutput:
        sender = [a for a in self.agents if a.id == agent.id][0]
        assert sender, "Sender not found"

        recipients = [a for a in self.agents if a.id in input_.agent_ids]
        if not recipients or len(recipients) != len(input_.agent_ids):
            found = [a.id for a in recipients]
            not_found = [
                a for a in input_.agent_ids if a not in found
            ]
            status = ResponseStatus(
                404, f"Recipient(s) not found: {not_found}"
            )
            return SendMessageOutput(status=status)

        msg = Message(
            timestamp=self.system_state.time,
            sender=sender,
            recipients=[a.name for a in recipients],
            content=input_.message,
        )

        for r in recipients:
            r.inbox.append(msg)
        return SendMessageOutput(status=RESPONSE_OK)
