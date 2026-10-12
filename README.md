# Kollektiv 4

A local collective of agents that discuss a shared goal, break it into tasks, create files,
review each other's work, and revise the result. Runs use **LangGraph** for orchestration,
**LangChain** for models and tools, and **Ollama** with `gemma4:e2b` by default.

The earlier experiments live in `_archive/v1`, `_archive/v2`, and `_archive/v3`.
The current implementation is in `src/kollektiv`.

![Kollektiv: configure a team, step through a mission and inspect live messages](docs/control-room.gif)

This short recording uses scripted model responses with the real graph, tools and checkpoints.

## Start

Requires Python 3.11+ and Ollama with the model installed. From the repository root:

```powershell
ollama pull gemma4:e2b
uv sync
uv run kollektiv doctor --generate
uv run kollektiv serve --open
```

If Ollama is not already running, start `ollama serve` in a separate terminal.
No cloud credentials are required. Open `http://127.0.0.1:8765` for the agent graph control room.
To try a mission, choose **New collective → Mission sample** in the UI. Easy, medium and elaborate
samples load editable goals, task budgets and communication graphs. The task-specific easy samples
start with three agents, medium samples use four, and **Solo orchestrator** starts with one.
They replace a previously loaded studio team with
a compact configuration. A small hiring demo recruits one temporary helper; two advanced studios
provide larger specialized teams.
From the terminal, use `uv run kollektiv run "Write a short poem about a robot in poem.md"`.
A mission takes several model calls; progress appears as agents respond. Ollama speed depends
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
uv run kollektiv run --goal-file mission.txt
uv run kollektiv run "Research ways to structure a multiplayer Pong project and write a sourced plan in plan.md" --web
```

Create `mission.txt` with your own requirements. The UI's mission samples start small; elaborate
missions take longer and may benefit from a larger model or larger context and output budgets.

## How the collective works

The default team has four distinct personas:

| Agent | Responsibilities | Tools |
| --- | --- | --- |
| `coordinator` | Integrate team advice into phases, tasks, dependencies and acceptance criteria | Read files, messages, shared notes |
| `researcher` | Research, architecture, shared conventions and writing | File tools, messages, notes, optional web tools |
| `builder` | Implement and author complete deliverables | File tools, messages, notes |
| `reviewer` | Independently review artifacts and assess the entire goal | Read files, deterministic checks, messages, notes |

1. **Discuss and research.** Every persona contributes recommendations and can send permitted messages.
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

The original team follows this reviewed workflow and can recruit temporary specialists while it
works. Changing the team or task count does not require generating graph code. See [workflow.mmd](workflow.mmd), or
run `uv run kollektiv graph` to render the current structure as Mermaid.

Direct messages are addressed to an exact teammate name; `all` broadcasts to the team.
Shared notes, discussion findings, task reflections and review feedback persist in the checkpoint.
Tasks dispatched in the same wave share the wave's starting context. Messages from that wave
become available to subsequent tasks; task dependencies establish ordering when a handoff is needed.

## Agent graph control room

```powershell
uv run kollektiv serve --open
# Or pick another local port:
uv run kollektiv serve --port 8766
```

The browser combines the team and its live activity in a draggable, zoomable agent graph.
Colored person cards show agent state, task progress and recent tools; directed edges animate
actual messages, and tool calls appear as loops around their agent. Select a person, channel,
task, event or artifact to inspect its details on the right.

The left sidebar holds the mission, phases and task states. A full-width bottom dock has two
independent tabbed panes, so messages and live debug can stay visible together. Each pane can
also show memory, state or artifacts. Resize the dock height and pane split with the handles
or their keyboard arrow controls. On narrow screens, the panes stack and the inspector becomes
a dismissible drawer. Animations respect the operating system's reduced-motion preference.

**Edit team** opens a draft configuration. Add or remove agents; select a person to edit its name,
role, instructions, skills, model and tool permissions. Choose the coordinator and reviewer.
Drag from a source handle to another person's target handle to create a directed communication
path. Use a path's remove button to delete it. Node positions are saved too.
The editor fills the browser window. Its Mission, Agents, Settings, Paths and Library views share
a collapsible sidebar; the properties inspector scrolls independently. Search the role roster
to focus a specialist at a readable zoom, or collapse both panels for an unobstructed graph.

Save drafts to the server's configuration library, reload them, or export/import portable JSON.
Presets live in `runs/_configurations/` by default and survive restarts. Creating a collective
from the draft preserves the current run and starts a new run with its own team and layout.
Saved configurations can omit a mission goal; starting a run requires one.

Unrestricted messaging preserves the default team's behavior. With directed paths, discretionary
messages can only reach permitted outgoing neighbors; broadcasts expand to those neighbors.
Removing `send_message` disables those messages as well. Automatic review feedback, task status
and admin instructions still reach their intended recipients; shared memory stays collective.
These paths control communication, while task dependencies control execution order.

The runtime graph preserves dragged positions across reloads and offers **Reset view** to bring
all people into view. Resizing and run changes fit the graph after its nodes are measured;
moving message packets follow the displayed path as nodes are dragged or the camera changes.

## Advanced mission studios

Choose a studio in **New collective → Mission sample**, or load it from the editor's Mission tab.
These examples replace the draft's team, directed message paths and layout while preserving your
selected Ollama model, server URL and concurrency. The goal and all roles remain editable.

| Example | Team and graph | Deliverables |
| --- | --- | --- |
| **Scientific paper lab** | 12 specialists and 35 directed paths: research director, question design, literature, citation evidence, method, formal analysis, adversarial validation, reproducibility, results, limitations, manuscript integration and independent peer review | A 4,500–6,500-word analytical systems paper on bounded retries and checkpoint recovery, nine manuscript sections, verified primary sources, worked DAG examples and a reproducibility bundle |
| **50-page novel studio** | 14 specialists and 45 directed paths: showrunner, premise, plot, world, characters, chronology, three act authors, scene design, continuity, line editing, book design and independent manuscript review | A complete speculative mystery, *The Lantern Archive*: creative foundations, 18 drafts, 18 polished chapters, front matter and an unabridged assembled book |

The novel targets 700–800 narrative words in each final chapter, or 12,600–14,400 words overall.
“50 pages” describes approximate manuscript length at 250–300 words per page; Markdown does not
enforce physical pagination. The paper is an analytical study with sourced claims and explicit
assumptions. Its generated reproduction program must be labeled unexecuted until you run it.

Both templates plan file-production tasks in dependency-gated phases and retain independent
artifact and whole-goal review. `assemble_files` concatenates accepted, declared dependency files
into the publisher's assigned output, preserving every chapter or section in order. Its response
contains checksums and counts rather than retransmitting a whole book through the model. It keeps
the usual workspace confinement, byte limits, atomic writes and checkpoint recovery.

These are substantial local-model workloads. The presets request a 131,072-token context window,
600-second request timeout and larger planning-output budgets; the novel allows 350,000 characters
of artifact review context. Choose a model and machine that support those settings, or edit the
budgets in the editor. The templates have been validated with real graph/tool/checkpoint fixtures;
they are not pre-generated papers or novels, and completing a run still depends on the model.

Portable configurations are also provided: [scientific-paper-lab.json](presets/scientific-paper-lab.json)
and [fifty-page-novel-studio.json](presets/fifty-page-novel-studio.json). Import them into the editor,
or pass one to CLI `--config` together with your mission goal. Regenerate these files from the
shared catalogue with `uv run python scripts/export_mission_presets.py`.

## Hire, supervise and fire specialists

The `hire` and `fire` skills are native agent tools with explicit permissions in **Edit team**.
Granting `hire` also requires explicit `monitor_agents` and `instruct_agent` permissions, so every
hiring persona can supervise its children. The editor selects those permissions together; imported
configurations must include them explicitly. `fire` is a separate permission.
Choose **Hire a garden specialist** under easy mission samples to try them with a three-agent
starting team. The writer hires one helper, monitors its real results, integrates them into the
garden plan and retires the helper. Other small samples keep a static three- or four-agent team.

For a team built entirely as it works, choose **Solo orchestrator**. Replace `[YOUR TASK HERE]` in
the goal with your actual task and required deliverables. Its generic role instructions tell the
single orchestrator to decompose the goal, hire specialists with exact personas and permissions,
monitor real progress, redirect work when needed and fire helpers once their work is accepted.
The orchestrator delegates authoring and uses `assemble_files` to publish verified direct-child
outputs into its own assigned deliverables, including results produced before planning. It cannot
assemble unfinished, unrelated or changed child outputs. Its coordinator and reviewer roles are
shared: checks of its own integrations are fresh-context self-reviews, and it can hire a separate
read-only critic when independent expertise is useful. No extra agents are silently added.
The portable configuration is [solo-orchestrator.json](presets/solo-orchestrator.json).

`hire` requires the exact child's `name`, `role`, `instructions`, `skills`, `tools`, `assignment`
and `acceptance_criteria`. File-writing children also receive an `output` path; their assigned file
lives under `delegates/<name>/` to keep ownership separate. Read-only children can omit `output`
and return findings as text, including when their tools list is empty. Both kinds of work receive
independent review. The original goal still needs its own complete deliverables and whole-goal audit.

The hierarchy has three levels: original agents are level 1, their hires are level 2, and those
hires' children are level 3. Level-3 agents cannot receive or execute `hire` or `fire`, including
forged tool requests. Names remain unique across the run, including retired helpers. Run settings
bound active helpers (`max_dynamic_agents`, default 8) and lifetime hires (`max_hires`, default 48).

Each parent gets automatic, persistent inbox messages when a child changes status. `monitor_agents`
shows its direct children's assignment, current node, recent tools, results and progress, with an
optional short wait. `instruct_agent` sends a correction to a direct child; the next model call
receives it automatically. Agents can supervise only their own direct children. Parent-child
messages work alongside the configured static communication paths.

`fire` retires a direct child and its descendants. It blocks future writes immediately; an active
model call can finish, but its retired agent cannot execute subsequent tools. Completed artifacts
and review history remain available. The live graph shows dashed family links and agent levels;
the inspector exposes parent reports, child progress, findings, artifacts and retired history.
An operator can send a child instruction or retire a family directly from the inspector while
the collective runs. Other checkpoint and artifact edits still require pausing.

Children reuse the existing LangGraph worker, review and SQLite checkpoint machinery, with a
separate durable delegation ledger. Replayed hires return the same child; scoped writes have
receipts so an interrupted call can recover without repeating the write. Stop and resume preserve
unfinished children. Model calls share the run's concurrency limit, including nested hires, so
a single-model setup does not require parallel Ollama calls. See the official
[LangGraph persistence guide](https://docs.langchain.com/oss/python/langgraph/persistence) for the
checkpoint and shared-store distinction used by this design.

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
While running, the frontend polls the lightweight event feed every 125 ms and refreshes the
full checkpoint approximately every 750 ms. Every event is read from the durable log, including
transitions that finish between polls. Idle and hidden tabs poll less often. Transport latency
appears in the header; model/server health is checked separately.

## Inspect and resume

Each invocation creates a unique `runs/<timestamp>-<id>/` directory containing:

| File | Purpose |
| --- | --- |
| `artifacts/` | Actual deliverables; the only location workers can write |
| `run.json` | Original goal, team, model and settings |
| `checkpoints.sqlite` | Durable graph and worker checkpoints |
| `delegation.sqlite`, `delegate-checkpoints.sqlite` | Hired-agent definitions, status messages, write receipts and private worker checkpoints |
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
uv run kollektiv run --goal-file mission.txt --config team.json
uv run kollektiv run "Your full goal" --model gemma4:e4b --base-url http://localhost:11434
uv run kollektiv resume runs/<run-id> --config overrides.json
```

Configuration is JSON. Omitted settings use defaults; resume configuration overrides only supplied
top-level fields. See [Settings and Agent](src/kollektiv/config.py) for the full validated schema.
`--config` also accepts a team configuration exported from the editor. Its settings are applied;
the CLI goal argument or goal file supplies the mission.

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
| `communication_links` | `null` | Unrestricted optional messaging; a list of `{source, target}` paths restricts it, and `[]` disables it |
| `max_dynamic_agents`, `max_hires` | `8`, `48` | Active hired workers and lifetime hires; hierarchy depth is always limited to three levels |

A custom `team` replaces the whole roster and can contain just one agent. Each agent has `name`,
`role`, `persona`, `skills`, `tools`, and an optional `model`. The `coordinator` and `reviewer`
settings name team members and may name the same agent. A sole agent receives both roles when
those settings are omitted. Deliverables require an eligible author with `write_file` or a
publisher with `assemble_files`; a distinct dedicated reviewer does not own production tasks.
For example, copy the roster from
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

The frontend source is in `frontend/`, built with React, TypeScript and Vite. It uses
[React Flow](https://reactflow.dev/) for graph interaction, Dagre for initial layout,
Zustand for runtime state, and Lucide for icons. Modules separate the transport/store,
graph, mission overview, inspector, dock and intervention forms. Components consume typed
snapshots and events through `api.ts`; `useRuntime.ts` owns polling and reconnection.

The compiled frontend is checked into `src/kollektiv/web/` and included in the Python package.
It ships one JavaScript bundle for compatibility with existing control-room processes.
Running the control room needs no Node installation. To change its source, install Node 20.19+
or 22.12+ and run:

```powershell
cd frontend
npm ci
npm run dev
```

Start `uv run kollektiv serve` from the repository root in another terminal. Vite's development
server proxies `/api` to `127.0.0.1:8765`. After editing, regenerate the packaged assets and check
the frontend:

```powershell
cd frontend
npm run build
npm test
npm run test:e2e
npm run format:check
```

The browser test starts an isolated server with scripted models, real tools and checkpoints;
it does not require Ollama or touch normal runs. It uses installed Microsoft Edge on Windows;
elsewhere, install Playwright Chromium with `npx playwright install chromium`, or set
`BROWSER_PATH` to a browser executable. Port 8767 must be free so the test can create its own
temporary workspace.

To regenerate the README recording with Playwright and FFmpeg, start a separate scripted demo
server from the repository root, then record from `frontend/` in another terminal:

```powershell
uv run python tests/ui_server.py --port 8768 --delay 0.12
# In the second terminal:
cd frontend
npm run demo:record
```

The recorder uses only this temporary server; it never modifies normal runs.

Run the Python checks from the repository root:

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

See [mcp_research.md](mcp_research.md) for researched local MCP hosting options, compatibility
constraints and a staged integration proposal. MCP integration is a future option.
