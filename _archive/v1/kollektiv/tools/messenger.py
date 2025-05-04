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
    agent_names: list[str]
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

    def send_message(self, sender, input_: SendMessageInput) -> SendMessageOutput:
        recipients = [a for a in self.agents if a.name in input_.agent_names]
        if not recipients or len(recipients) != len(input_.agent_names):
            found = [a.name for a in recipients]
            not_found = [a for a in input_.agent_names if a not in found]
            status = ResponseStatus(404, f"Recipient(s) not found: {not_found}")
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
