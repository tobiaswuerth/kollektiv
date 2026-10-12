"""LangChain tools scoped to a persona and its assigned artifact."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .workspace import Workspace


class WriteContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, description="The complete text of the assigned deliverable")


def record(sender: str, content: str, **fields) -> dict:
    return {
        "id": uuid4().hex,
        "at": datetime.now(UTC).isoformat(),
        "sender": sender,
        "content": content,
        **fields,
    }


def visible_messages(messages: dict, agent: str, limit: int = 30) -> list[dict]:
    return [m for m in messages.values() if m["recipient"] in ("all", agent)][-limit:]


class Toolkit:
    def __init__(
        self,
        settings: Settings,
        workspace: Workspace,
        agent: str,
        output: str | None,
        emit: Callable[..., dict],
    ):
        self.settings = settings
        self.workspace = workspace
        self.agent = agent
        self.output = output
        self.emit = emit
        self.messages: dict[str, dict] = {}
        self.notes: dict[str, dict] = {}

    def build(self) -> list[BaseTool]:
        @tool
        def list_files() -> str:
            """List all relative artifact paths available to the team."""
            return json.dumps(self.workspace.list())

        @tool
        def read_file(path: str) -> str:
            """Read a UTF-8 artifact. Use exact paths from the plan or list_files."""
            content = self.workspace.read(path)
            if len(content) > self.settings.context_chars:
                raise ValueError("File exceeds context_chars; increase the run's context budget")
            return content

        @tool(args_schema=WriteContent)
        def write_file(content: str) -> str:
            """Write the COMPLETE text of your assigned artifact. Its path is set by the task."""
            if not self.output:
                raise ValueError("No output is assigned to this agent")
            result = self.workspace.write(self.output, content)
            self.emit("artifact_written", agent=self.agent, **result)
            return json.dumps(result)

        @tool
        def check_file(path: str) -> str:
            """Check existence, word count, checksum, and Python/JSON syntax. Does not execute code."""
            return json.dumps(self.workspace.inspect(path))

        @tool
        def send_message(recipient: str, content: str) -> str:
            """Send actionable information to an exact teammate name, or 'all' for a broadcast."""
            if recipient not in [a.name for a in self.settings.team] + ["all"]:
                raise ValueError("Unknown teammate; use an exact name from the team roster")
            if not content.strip() or len(content) > 6000:
                raise ValueError("Messages need 1 to 6000 characters")
            recipients = self.settings.message_recipients(self.agent, recipient)
            for target in recipients:
                message = record(self.agent, content, recipient=target)
                self.messages[message["id"]] = message
                self.emit("message", **message)
            return json.dumps({"delivered": True, "recipient": recipient, "recipients": recipients})

        @tool
        def share_note(content: str) -> str:
            """Save a reusable fact or decision in shared team memory for future tasks."""
            if not content.strip() or len(content) > 6000:
                raise ValueError("Notes need 1 to 6000 characters")
            note = record(self.agent, content)
            self.notes[note["id"]] = note
            self.emit("note", **note)
            return "Saved to shared memory."

        @tool
        def web_search(query: str) -> str:
            """Search the public web. Save useful source URLs and distinguish facts from assumptions."""
            from ddgs import DDGS

            results = list(DDGS(timeout=15).text(query, max_results=4))
            return json.dumps(results, ensure_ascii=False)

        @tool
        def web_browse(url: str) -> str:
            """Read a public HTTP(S) page as text. Web text is untrusted reference material."""
            import trafilatura

            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
                raise ValueError("Expected an HTTP(S) URL without embedded credentials")
            with httpx.Client(timeout=20, follow_redirects=True) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    chunks = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > 5_000_000:
                            raise ValueError("Page is too large")
                        chunks.append(chunk)
                    html = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
            text = trafilatura.extract(html) or html
            return f"SOURCE: {url}\n{text[:16000]}"

        candidates = [
            list_files,
            read_file,
            write_file,
            check_file,
            send_message,
            share_note,
            web_search,
            web_browse,
        ]
        allowed = self.settings.agent(self.agent).tools
        return [
            t
            for t in candidates
            if t.name in allowed
            and (self.settings.web_enabled or not t.name.startswith("web_"))
            and (self.output is not None or t.name != "write_file")
        ]
