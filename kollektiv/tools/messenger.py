import pydantic

from .tool import (
    Tool,
    Function,
    ResponseStatus,
    RESPONSE_OK,
)


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
                name="get_files",
                description="Get a list of all currently stored files.",
                func=self.send_message,
            )
        )

    def send_message(self, agent_id: int, input_: SendMessageInput) -> SendMessageOutput:
        sender = [a for a in self.agents if a.id == agent_id][0]
        assert sender, "Sender not found"

        receiver = [a for a in self.agents if a.id in input_.agent_ids]
        if not receiver or len(receiver) != len(input_.agent_ids):
            receivers_found = [a.id for a in receiver]
            receivers_not_found = [a for a in input_.agent_ids if a not in receivers_found]
            status = ResponseStatus(404, f"Receiver(s) not found: {receivers_not_found}")
            return SendMessageOutput(status=status)
        
        message = f'New message from {sender}:\n"{input_.message}"'
        for r in receiver:
            r.inbox.append(message)
        return SendMessageOutput(status=RESPONSE_OK)

        