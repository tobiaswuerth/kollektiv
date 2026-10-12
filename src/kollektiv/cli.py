"""Small CLI for local operation; no hosted API keys or graphical dependencies."""

import argparse
import json
import sys
from pathlib import Path

import httpx

from .config import Settings
from .llm import Models, json_text
from .models import Contribution
from .runtime import Collective
from .workflow import Workflow
from .workspace import Workspace


def read_config(path: str | None) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def progress(event: dict) -> None:
    kind = event["kind"]
    agent = event.get("agent", event.get("sender", "collective"))
    if kind == "model_started":
        print(f"[{agent}] working on {event['purpose']}...", flush=True)
    elif kind == "model":
        usage = event.get("usage") or {}
        print(
            f"[{agent}] {event['purpose']} ({usage.get('output_tokens', '?')} tokens)", flush=True
        )
    elif kind == "artifact_written":
        print(f"[{agent}] wrote {event['path']} ({event['words']} words)", flush=True)
    elif kind in ("task_review", "goal_review"):
        accepted = event.get("accepted", event.get("achieved", False))
        print(
            f"[{agent}] {event.get('task', 'goal')}: {'accepted' if accepted else 'needs changes'}",
            flush=True,
        )
    elif kind in ("plan_created", "task_finished", "goal_revision", "run_finished", "schema_retry"):
        detail = {k: v for k, v in event.items() if k not in ("id", "at", "kind")}
        print(f"[{agent}] {kind}: {json.dumps(detail, ensure_ascii=False)}", flush=True)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Local agents collaborating toward a shared goal")
    commands = cli.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Start an isolated, checkpointed collective")
    goal = run.add_mutually_exclusive_group(required=True)
    goal.add_argument("goal", nargs="?", help="Shared goal in quotes")
    goal.add_argument("--goal-file", help="Read the goal from a UTF-8 text file")
    run.add_argument("--runs-dir", default="runs")
    resume = commands.add_parser("resume", help="Resume the last checkpoint of an existing run")
    resume.add_argument("run_dir")
    resume.add_argument(
        "--retry", action="store_true", help="Give an incomplete run a fresh bounded attempt budget"
    )
    for command in (run, resume):
        command.add_argument("--config", help="JSON settings, or overrides when resuming")
        command.add_argument("--model")
        command.add_argument("--base-url")
        command.add_argument(
            "--web",
            action="store_true",
            default=None,
            help="Enable web tools for agents whose personas allow them",
        )
        command.add_argument("--quiet", action="store_true")
    status = commands.add_parser("status", help="Inspect checkpointed task status")
    status.add_argument("run_dir")
    status.add_argument("--json", action="store_true")
    listing = commands.add_parser("list", help="List local runs")
    listing.add_argument("--runs-dir", default="runs")
    doctor = commands.add_parser("doctor", help="Check Ollama connectivity and model availability")
    doctor.add_argument("--config")
    doctor.add_argument("--model")
    doctor.add_argument("--base-url")
    doctor.add_argument(
        "--generate", action="store_true", help="Also check a real structured model response"
    )
    graph = commands.add_parser("graph", help="Print the workflow as Mermaid")
    graph.add_argument("--output", help="Write Mermaid to this file")
    serve = commands.add_parser("serve", help="Open the agent graph control room")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--runs-dir", default="runs")
    serve.add_argument("--open", action="store_true", help="Open the control room in your browser")
    return cli


def config_overrides(args) -> dict:
    data = read_config(getattr(args, "config", None))
    for flag, key in (("model", "model"), ("base_url", "base_url"), ("web", "web_enabled")):
        value = getattr(args, flag, None)
        if value is not None:
            data[key] = value
    return data


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "serve":
            from .webapp import serve

            serve(host=args.host, port=args.port, runs_dir=args.runs_dir, open_browser=args.open)
            return 0
        if args.command in ("run", "resume"):
            overrides = config_overrides(args)
            if args.command == "run":
                goal = (
                    Path(args.goal_file).read_text(encoding="utf-8-sig")
                    if args.goal_file
                    else args.goal
                )
                run = Collective.create(goal, args.runs_dir, Settings.model_validate(overrides))
            else:
                run = Collective.open(args.run_dir, overrides)
            print(f"Run: {run.directory}", flush=True)
            state = run.run(
                retry=getattr(args, "retry", False), on_event=None if args.quiet else progress
            )
            print(f"Status: {state['status']}\n{state.get('final_summary', '')}")
            print(f"Artifacts: {run.workspace.root}\nReport: {run.directory / 'report.md'}")
            return 0 if state["status"] == "completed" else 2
        if args.command == "status":
            run = Collective.open(args.run_dir)
            state = run.state()
            if args.json:
                print(json_text(state))
            else:
                print(
                    f"{run.directory}\nStatus: {state.get('status', 'unknown')}\nGoal: {run.manifest['goal']}"
                )
                if run.manifest.get("last_error"):
                    print(f"Interruption: {run.manifest['last_error']}")
                for task in state.get("tasks", {}).values():
                    print(f"  {task['id']}: {task['status']} ({task['owner']}) -> {task['output']}")
                print(state.get("final_summary", ""))
            return 0
        if args.command == "list":
            for path in sorted(Path(args.runs_dir).glob("*/run.json")):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                print(f"{path.parent.name}  {manifest['status']}  {manifest['goal'][:100]}")
            return 0
        if args.command == "doctor":
            settings = Settings.model_validate(config_overrides(args))
            with httpx.Client(base_url=settings.base_url, timeout=10) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
                available = {m["name"] for m in response.json()["models"]}
            required = {a.model or settings.model for a in settings.team}
            missing = required - available
            if missing:
                raise ValueError(
                    f"Models missing from Ollama: {', '.join(sorted(missing))}. Use ollama pull MODEL."
                )
            print(
                f"Ollama reachable at {settings.base_url}; models ready: {', '.join(sorted(required))}"
            )
            if args.generate:
                model = Models(settings, lambda kind, **fields: {"kind": kind, **fields})
                response = model.structured(
                    settings.coordinator,
                    Contribution,
                    "Return findings='Ollama is ready' and one message to builder saying 'Ready'.",
                )
                print(f"Structured output OK: {response.findings}")
            return 0
        if args.command == "graph":
            # An in-memory checkpointer keeps graph rendering independent of Ollama and run storage.
            from tempfile import TemporaryDirectory

            from langgraph.checkpoint.memory import InMemorySaver

            with TemporaryDirectory() as tmp:
                workflow = Workflow(Settings(), Workspace(Path(tmp)), lambda *_a, **_k: {})
                diagram = workflow.build(InMemorySaver()).get_graph(xray=True).draw_mermaid()
            if args.output:
                Path(args.output).write_text(diagram, encoding="utf-8")
            else:
                print(diagram)
            return 0
    except KeyboardInterrupt:
        print(
            "Interrupted. Progress is checkpointed; use kollektiv resume RUN_DIR.", file=sys.stderr
        )
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Existing run progress is checkpointed. Inspect status and resume its run directory.",
            file=sys.stderr,
        )
        return 1
    return 0
