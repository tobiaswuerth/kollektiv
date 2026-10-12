"""Validated run configuration, persisted with every run for reproducible resumes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ToolName = Literal[
    "list_files",
    "read_file",
    "write_file",
    "assemble_files",
    "check_file",
    "send_message",
    "share_note",
    "web_search",
    "web_browse",
    "hire",
    "fire",
    "monitor_agents",
    "instruct_agent",
]


def validate_hiring_tools(tools: list[str]) -> None:
    if "hire" in tools:
        missing = {"monitor_agents", "instruct_agent"} - set(tools)
        if missing:
            raise ValueError(
                "The hire skill requires explicitly granting monitor_agents and instruct_agent; "
                f"missing: {', '.join(sorted(missing))}"
            )


class Agent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    role: str = Field(min_length=1)
    persona: str = Field(min_length=1)
    skills: list[str] = Field(default_factory=list)
    tools: list[ToolName] = Field(
        default_factory=lambda: [
            "list_files",
            "read_file",
            "write_file",
            "check_file",
            "send_message",
            "share_note",
        ]
    )
    model: str | None = None

    @model_validator(mode="after")
    def validate_supervision(self) -> "Agent":
        validate_hiring_tools(self.tools)
        return self


class CommunicationLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str


def default_team() -> list[Agent]:
    return [
        Agent(
            name="coordinator",
            role="Project coordinator",
            persona="Practical and precise. Plan small tasks, explicit dependencies and measurable acceptance criteria.",
            skills=["planning", "integration"],
            tools=["list_files", "read_file", "send_message", "share_note"],
        ),
        Agent(
            name="researcher",
            role="Researcher and architect",
            persona="Curious and skeptical. Establish facts, record sources, define consistent conventions for the team.",
            skills=["research", "architecture", "writing"],
            tools=[
                "list_files",
                "read_file",
                "write_file",
                "check_file",
                "send_message",
                "share_note",
                "web_search",
                "web_browse",
            ],
        ),
        Agent(
            name="builder",
            role="Builder and author",
            persona="Resourceful and concrete. Produce complete usable files, follow dependencies, and communicate design decisions.",
            skills=["implementation", "writing", "editing"],
        ),
        Agent(
            name="reviewer",
            role="Independent quality reviewer",
            persona="Constructive and exacting. Inspect actual artifacts against every requirement and send actionable corrections.",
            skills=["review", "validation", "consistency"],
            tools=["list_files", "read_file", "check_file", "send_message", "share_note"],
        ),
    ]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "gemma4:e2b"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.2, ge=0, le=2)
    reasoning: bool = False
    num_ctx: int = Field(default=32768, ge=4096)
    num_predict: int = Field(default=8192, ge=256)
    request_timeout: float = Field(default=180, gt=0)
    structured_retries: int = Field(default=2, ge=0, le=5)
    max_tasks: int = Field(default=24, ge=1, le=200)
    max_steps: int = Field(default=12, ge=1, le=100)
    max_task_attempts: int = Field(default=3, ge=1, le=10)
    max_goal_revisions: int = Field(default=2, ge=0, le=10)
    discussion_rounds: int = Field(default=2, ge=1, le=5)
    max_concurrency: int = Field(default=1, ge=1, le=16)
    context_chars: int = Field(default=60000, ge=4000)
    max_file_bytes: int = Field(default=2_000_000, ge=1024)
    max_dynamic_agents: int = Field(default=8, ge=1, le=32)
    max_hires: int = Field(default=48, ge=1, le=200)
    web_enabled: bool = False
    team: list[Agent] = Field(default_factory=default_team, min_length=3)
    coordinator: str = "coordinator"
    reviewer: str = "reviewer"
    communication_links: list[CommunicationLink] | None = None

    @model_validator(mode="after")
    def validate_team(self) -> "Settings":
        names = [a.name for a in self.team]
        if len(names) != len(set(names)):
            raise ValueError("Agent names must be unique")
        if self.coordinator not in names or self.reviewer not in names:
            raise ValueError("coordinator and reviewer must name members of the team")
        if self.coordinator == self.reviewer:
            raise ValueError("The coordinator and reviewer must be different agents")
        if not any("write_file" in a.tools and a.name != self.reviewer for a in self.team):
            raise ValueError("The team needs a worker with write_file access")
        if self.communication_links is not None:
            pairs = [(link.source, link.target) for link in self.communication_links]
            if len(pairs) != len(set(pairs)):
                raise ValueError("Communication paths must be unique")
            if any(source not in names or target not in names for source, target in pairs):
                raise ValueError("Communication paths must connect members of the team")
            if any(source == target for source, target in pairs):
                raise ValueError("Communication paths must connect two different agents")
        return self

    def agent(self, name: str) -> Agent:
        return next(a for a in self.team if a.name == name)

    def allowed_recipients(self, sender: str) -> list[str]:
        """Optional messages follow directed paths and the sender's tool permissions."""
        if "send_message" not in self.agent(sender).tools:
            return []
        targets = (
            [agent.name for agent in self.team]
            if self.communication_links is None
            else [link.target for link in self.communication_links if link.source == sender]
        )
        return [*targets, "all"] if targets else []

    def message_recipients(self, sender: str, recipient: str) -> list[str]:
        if recipient not in self.allowed_recipients(sender):
            raise ValueError(
                f"{sender} cannot send to {recipient!r}; allowed recipients: "
                f"{self.allowed_recipients(sender)}"
            )
        if recipient == "all" and self.communication_links is not None:
            return [link.target for link in self.communication_links if link.source == sender]
        return [recipient]
