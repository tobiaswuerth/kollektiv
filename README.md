# Kollektiv 4

A local collective of agents that discuss a shared goal, break it into tasks, create files,
review each other's work, and revise the result. Runs use **LangGraph** for orchestration,
**LangChain** for models and tools, and **Ollama** with `gemma4:e2b` by default.

The earlier experiments live in `_archive/v1`, `_archive/v2`, and `_archive/v3`.
The current implementation is in `src/kollektiv`.

## Start

Requires Python 3.11+ and Ollama with the model installed. From the repository root:

```powershell
ollama pull gemma4:e2b
uv sync
uv run kollektiv doctor --generate
uv run kollektiv serve --open
```

If Ollama is not already running, start `ollama serve` in a separate terminal.
No cloud credentials are required. Open `http://127.0.0.1:8765` for the virtual-world control room.
To run from the terminal, use `uv run kollektiv run --goal-file examples/smoke.txt`.
The example takes several model calls; progress appears as agents respond. Ollama speed depends
on the machine and the amount of content requested.

With pip instead of uv:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m kollektiv doctor --generate
.\.venv\Scripts\python.exe -m kollektiv run "Write a short poem about a robot in poem.md"
```

For a longer goal, put the full requirements in a UTF-8 file:

```powershell
uv run kollektiv run --goal-file examples/story.txt
uv run kollektiv run --goal-file examples/software.txt
uv run kollektiv run "Research ways to structure a multiplayer Pong project and write a sourced plan in plan.md" --web
```

The examples are deliberately small enough for a local model. Larger projects can use a larger model and larger context and output budgets through configuration.

## How the collective works

The default team has four distinct personas:

| Agent | Responsibilities | Tools |
| --- | --- | --- |
| `coordinator` | Integrate team advice into phases, tasks, dependencies and acceptance criteria | Read files, messages, shared notes |
| `researcher` | Research, architecture, shared conventions and writing | File tools, messages, notes, optional web tools |
| `builder` | Implement and author complete deliverables | File tools, messages, notes |
| `reviewer` | Independently review artifacts and assess the entire goal | Read files, deterministic checks, messages, notes |

1. **Discuss and research.** Every persona contributes recommendations and sends messages.
   A second round lets teammates respond to their inboxes. With `--web`, researchers can search
   and browse before the plan is created.
2. **Plan.** The coordinator produces a validated plan with phases, one file per task, exact
   output paths, suitable owners, acceptance criteria and a dependency DAG.
   Draft/edit steps for the same file are combined while preserving instructions, criteria and
   external dependencies. Cycles and unknown task references fail validation.
3. **Act.** LangGraph's `Send` dispatches ready tasks to worker subgraphs. Workers use native
   LangChain tool calls to read inputs, write their assigned file, send messages and save notes.
   Workers cannot overwrite another task's deliverable.
4. **Review and reflect.** The reviewer reads the actual file and its inputs. Missing, empty,
   or syntactically invalid Python/JSON files fail deterministic checks. Rejected work returns
   to its owner with feedback; accepted work unlocks dependent tasks. Lessons enter shared memory.
5. **Audit the goal.** The reviewer checks all deliverables together against the original goal.
   It can reopen tasks or add missing deliverables. Revising an input also reopens its dependents.
   Only an accepted whole-goal review produces `completed`.

This is a fixed workflow with a dynamic team and task graph; adding a persona or changing the
number of tasks does not require generating graph code. See [workflow.mmd](workflow.mmd), or
run `uv run kollektiv graph` to render the current structure as Mermaid.

Direct messages are addressed to an exact teammate name; `all` broadcasts to the team.
Shared notes, discussion findings, task reflections and review feedback persist in the checkpoint.
Tasks dispatched in the same wave share the wave's starting context. Messages from that wave
become available to subsequent tasks; task dependencies establish ordering when a handoff is needed.

## Virtual-world control room

```powershell
uv run kollektiv serve --open
# Or pick another local port:
uv run kollektiv serve --port 8766
```

The browser displays the team on a small virtual island, with activity and communication animated
from actual runtime events. Pan and zoom the world, select an agent, and inspect the collective
through the live activity, tasks, conversations, artifacts and debug views.

Create a goal and configure its model/team, then start it or step through one graph node at a time.
Steps include individual worker model calls, tool executions and reviews. Pause waits for the
current node to finish; stop preserves a checkpoint you can resume. Existing CLI runs appear in
the same run selector.

God's Eye controls let you send an admin message, add shared memory, retry a task, reassign it to
another worker, or edit an artifact. Pause first. Admin edits stop the paused runner, change its
checkpoint, and prepare it for a fresh step or resume, so edits cannot race an executing node.
Artifact changes reopen the affected task and its dependents for review. Goal and team settings
can be edited before the first graph execution; a different goal needs a new run after work starts.

The state inspector includes active worker checkpoints and tool history. The debug feed exposes
model responses, tool calls/results, checksums, reviews, control changes and failures.

## Inspect and resume

Each invocation creates a unique `runs/<timestamp>-<id>/` directory containing:

| File | Purpose |
| --- | --- |
| `artifacts/` | Actual deliverables; the only location workers can write |
| `run.json` | Original goal, team, model and settings |
| `checkpoints.sqlite` | Durable graph and worker checkpoints |
| `state.json`, `plan.json` | Inspectable snapshots of task progress and the plan |
| `messages.json`, `memory.json` | Team communication and shared memory |
| `events.jsonl` | Model usage, tool results, artifact checksums, reviews and interruptions |
| `report.md` | Result, deliverable links and review evidence |

```powershell
uv run kollektiv list
uv run kollektiv status runs/<run-id>
uv run kollektiv status runs/<run-id> --json
uv run kollektiv resume runs/<run-id>
```

Ctrl+C or a model/network failure leaves progress checkpointed. Resume uses the saved goal and
configuration and continues unfinished graph nodes, including a worker's tool history.
Previously accepted files are verified by checksum; changed or missing files and their dependent
tasks are rebuilt and reviewed. An OS-held lock prevents two processes from advancing the same run.

Retries and revisions are bounded. Exhausting a budget produces `incomplete`, keeps the work and
feedback, and exits with code 2. A model or configuration error exits with code 1. To deliberately
give an incomplete run another attempt budget:

```powershell
uv run kollektiv resume runs/<run-id> --retry
```

## Configure the team and model

```powershell
uv run kollektiv run --goal-file examples/story.txt --config examples/config.json
uv run kollektiv run "Your full goal" --model gemma4:e4b --base-url http://localhost:11434
uv run kollektiv resume runs/<run-id> --config overrides.json
```

Configuration is JSON. Omitted settings use defaults; resume configuration overrides only supplied
top-level fields. See [Settings and Agent](src/kollektiv/config.py) for the full validated schema.

| Setting | Default | Effect |
| --- | --- | --- |
| `model`, `base_url` | `gemma4:e2b`, `http://localhost:11434` | Ollama model and server |
| `reasoning` | `false` | Enable Ollama thinking for models supporting it |
| `num_ctx`, `num_predict` | `32768`, `8192` | Model context and maximum output tokens per call |
| `context_chars` | `60000` | Maximum artifact text supplied for a review |
| `request_timeout` | `180` | Seconds allowed per model request |
| `max_concurrency` | `1` | Graph task concurrency; default suits a single local model |
| `discussion_rounds` | `2` | Team discussion rounds before planning |
| `max_tasks` | `24` | Maximum total tasks, including goal-review additions |
| `max_steps` | `12` | Model/tool steps per task attempt |
| `max_task_attempts` | `3` | Attempts per task per goal revision |
| `max_goal_revisions` | `2` | Whole-goal correction rounds |
| `structured_retries` | `2` | Schema repair retries using validation feedback |
| `web_enabled` | `false` | Permit web tools for personas configured with them |

A custom `team` replaces the whole roster. Each agent has `name`, `role`, `persona`, `skills`,
`tools`, and an optional `model`. The `coordinator` and `reviewer` settings name two different team
members; at least one other member must have `write_file`. For example, copy the roster from
`run.json`, change the personas or per-agent models, and provide it in your config.

Artifacts are text files of any extension, with a configurable size limit. Generated programs
are checked for Python/JSON syntax where applicable and reviewed by the model. They are not
executed by this application, so test generated software before relying on it. Completion is the
collective's evidence-based assessment, and its quality still depends on the model. A small local
model may need more attempts or a more explicit goal for large or complex projects.

If a review would exceed `context_chars`, the run stops with a resumable error instead of
approving partially visible files. Increase both `context_chars` and `num_ctx` when resuming.
If generation reaches `num_predict`, increase that budget. Web lookup failures are reported as
tool results and do not supply invented sources.

## Python API

```python
from kollektiv import Collective, Settings

run = Collective.create(
    "Create design.md and a consistent short story in story.md.",
    runs_dir="runs",
    settings=Settings(model="gemma4:e2b"),
)
result = run.run()
print(result["status"], run.workspace.root)

# Later, including from another process:
result = Collective.open(run.directory).run()
```

## Development and validation

```powershell
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

The automated tests exercise the real graph, tool execution and SQLite persistence with scripted
model responses: dependencies, direct-message delivery, rejection and correction, goal revisions,
missing deliverables, crash recovery after a file write, artifact changes, isolated runs and write
confinement. `doctor --generate` and the smoke goal exercise the installed Ollama model.

The implementation follows the official
[LangGraph orchestrator/worker and evaluator/optimizer patterns](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
and [ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama).
