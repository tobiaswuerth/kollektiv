import { useState } from "react";
import {
  FolderOpen,
  GitBranch,
  Search,
  Settings2,
  Target,
  Trash2,
  Users,
} from "lucide-react";
import { linkId, workflowLinks } from "../graph/model";
import type { MissionSample, TeamConfiguration } from "../types";
import { MissionSamplePicker } from "./MissionSamplePicker";

export type EditorTab = "mission" | "agents" | "settings" | "paths" | "library";

const tabs = [
  { id: "mission", label: "Mission", Icon: Target },
  { id: "agents", label: "Agents", Icon: Users },
  { id: "settings", label: "Settings", Icon: Settings2 },
  { id: "paths", label: "Paths", Icon: GitBranch },
  { id: "library", label: "Library", Icon: FolderOpen },
] as const;

export function CommunicationMode({
  draft,
  change,
}: {
  draft: TeamConfiguration;
  change: (next: TeamConfiguration) => void;
}) {
  return (
    <label className="editor-communication-mode">
      Messaging
      <select
        aria-label="Communication mode"
        value={
          draft.settings.communication_links == null ? "unrestricted" : "custom"
        }
        onChange={(event) =>
          change({
            ...draft,
            settings: {
              ...draft.settings,
              communication_links:
                event.target.value === "unrestricted"
                  ? null
                  : workflowLinks(draft.settings),
            },
          })
        }
      >
        <option value="unrestricted">Unrestricted</option>
        <option value="custom">Directed paths</option>
      </select>
    </label>
  );
}

function TeamRoster({
  draft,
  selected,
  select,
}: {
  draft: TeamConfiguration;
  selected: string;
  select: (name: string) => void;
}) {
  const [query, setQuery] = useState("");
  const matching = draft.settings.team.filter((agent) =>
    `${agent.name} ${agent.role} ${agent.skills.join(" ")}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  return (
    <div className="editor-roster">
      <div className="editor-panel-intro">
        <h3>
          Team roster <span>{draft.settings.team.length}</span>
        </h3>
        <p>Select a specialist to focus its node and edit its instructions.</p>
      </div>
      <label className="editor-search">
        <span>
          <Search size={13} /> Find an agent
        </span>
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Name, role, or skill"
        />
      </label>
      <div className="editor-roster-list">
        {matching.map((agent, index) => (
          <button
            key={agent.name}
            type="button"
            className={agent.name === selected ? "selected" : ""}
            aria-label={`Select agent ${agent.name}`}
            aria-pressed={agent.name === selected}
            onClick={() => select(agent.name)}
          >
            <span className="editor-roster-index">
              {String(index + 1).padStart(2, "0")}
            </span>
            <span className="editor-roster-description">
              <strong>{agent.name}</strong>
              <small>{agent.role}</small>
            </span>
            {agent.name === draft.settings.coordinator ? (
              <span className="editor-roster-designation">Lead</span>
            ) : agent.name === draft.settings.reviewer ? (
              <span className="editor-roster-designation">Review</span>
            ) : null}
          </button>
        ))}
        {!matching.length && (
          <p className="editor-panel-note">No agents match this search.</p>
        )}
      </div>
    </div>
  );
}

function RuntimeSettings({
  draft,
  change,
}: {
  draft: TeamConfiguration;
  change: (next: TeamConfiguration) => void;
}) {
  const update = (fields: Partial<TeamConfiguration["settings"]>) =>
    change({ ...draft, settings: { ...draft.settings, ...fields } });
  return (
    <div className="editor-runtime">
      <div className="editor-panel-intro">
        <h3>Execution settings</h3>
        <p>Control the team leads, local model, and budgets for this draft.</p>
      </div>
      {(["coordinator", "reviewer"] as const).map((field) => (
        <label key={field}>
          {field === "coordinator" ? "Coordinator" : "Reviewer"}
          <select
            value={draft.settings[field]}
            onChange={(event) => update({ [field]: event.target.value })}
          >
            {draft.settings.team.map((entry) => (
              <option key={entry.name} value={entry.name}>
                {entry.name}
              </option>
            ))}
          </select>
        </label>
      ))}
      <label>
        Default model
        <input
          value={draft.settings.model}
          onChange={(event) => update({ model: event.target.value })}
        />
      </label>
      <label>
        Ollama URL
        <input
          value={draft.settings.base_url}
          onChange={(event) => update({ base_url: event.target.value })}
        />
      </label>
      <label>
        Concurrency
        <input
          type="number"
          min={1}
          max={16}
          value={draft.settings.max_concurrency}
          onChange={(event) =>
            update({ max_concurrency: Number(event.target.value) })
          }
        />
      </label>
      <label className="editor-web">
        <input
          type="checkbox"
          checked={draft.settings.web_enabled}
          onChange={(event) => update({ web_enabled: event.target.checked })}
        />
        Allow web tools
      </label>
      <details className="editor-budget-settings">
        <summary>Task and model budgets</summary>
        {[
          {
            field: "max_tasks",
            label: "Maximum tasks",
            fallback: 24,
            min: 1,
            max: 200,
          },
          {
            field: "max_steps",
            label: "Tool steps per task",
            fallback: 12,
            min: 1,
            max: 100,
          },
          {
            field: "max_task_attempts",
            label: "Attempts per task",
            fallback: 3,
            min: 1,
            max: 10,
          },
          {
            field: "max_goal_revisions",
            label: "Mission revision rounds",
            fallback: 2,
            min: 0,
            max: 10,
          },
          {
            field: "num_predict",
            label: "Output token budget",
            fallback: 8192,
            min: 256,
          },
          {
            field: "num_ctx",
            label: "Context token budget",
            fallback: 32768,
            min: 4096,
          },
          {
            field: "context_chars",
            label: "Shared context characters",
            fallback: 60000,
            min: 4000,
          },
        ].map(({ field, label, fallback, min, max }) => (
          <label key={field}>
            {label}
            <input
              type="number"
              min={min}
              max={max}
              value={Number(draft.settings[field] ?? fallback)}
              onChange={(event) =>
                update({ [field]: Number(event.target.value) })
              }
            />
          </label>
        ))}
      </details>
    </div>
  );
}

function CommunicationPaths({
  draft,
  change,
}: {
  draft: TeamConfiguration;
  change: (next: TeamConfiguration) => void;
}) {
  const [query, setQuery] = useState("");
  const links = draft.settings.communication_links;
  const matching = links?.filter((link) =>
    `${link.source} ${link.target}`.includes(query.toLowerCase()),
  );
  return (
    <div className="editor-channels">
      <div className="editor-panel-intro">
        <h3>
          Communication paths <span>{links?.length ?? "All"}</span>
        </h3>
        <p>
          {links == null
            ? "All agents may communicate. Draw a path to enable explicit channels."
            : "Only these directed agent message channels are allowed."}
        </p>
      </div>
      <p className="editor-panel-note">
        Draw from an agent's bottom handle to another agent's top handle. Select
        a line and use Remove selected path above the canvas to delete it.
      </p>
      {links != null && links.length > 0 && (
        <label className="editor-search">
          <span>
            <Search size={13} /> Find a path
          </span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Source or recipient"
          />
        </label>
      )}
      <div className="editor-channel-list">
        {matching?.map((link) => (
          <button
            type="button"
            key={linkId(link)}
            onClick={() =>
              change({
                ...draft,
                settings: {
                  ...draft.settings,
                  communication_links: links!.filter(
                    (entry) => linkId(entry) !== linkId(link),
                  ),
                },
              })
            }
            aria-label={`Remove path ${link.source} to ${link.target}`}
          >
            <span>
              {link.source}
              <small>→ {link.target}</small>
            </span>
            <Trash2 size={13} />
          </button>
        ))}
      </div>
      {links?.length === 0 && (
        <p className="editor-panel-note">
          No message channels are enabled. Draw paths between agent handles to
          add them.
        </p>
      )}
      {links && links.length > 0 && matching?.length === 0 && (
        <p className="editor-panel-note">No paths match this search.</p>
      )}
      <p className="editor-panel-note">
        Task scheduling and required review feedback remain part of the
        execution workflow.
      </p>
    </div>
  );
}

export function EditorSidebar({
  tab,
  setTab,
  draft,
  change,
  selected,
  select,
  samples,
  sample,
  chooseSample,
  configurations,
  load,
}: {
  tab: EditorTab;
  setTab: (tab: EditorTab) => void;
  draft: TeamConfiguration;
  change: (next: TeamConfiguration) => void;
  selected: string;
  select: (name: string) => void;
  samples: MissionSample[];
  sample: string;
  chooseSample: (id: string) => void;
  configurations: TeamConfiguration[];
  load: (id: string) => void;
}) {
  return (
    <>
      <div
        role="tablist"
        aria-label="Collective editor sections"
        className="editor-section-tabs"
      >
        {tabs.map(({ id, label, Icon }) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={id === tab}
            aria-controls={`editor-section-${id}`}
            id={`editor-tab-${id}`}
            tabIndex={id === tab ? 0 : -1}
            onClick={() => setTab(id)}
            onKeyDown={(event) => {
              const index = tabs.findIndex((entry) => entry.id === id);
              const nextIndex =
                event.key === "ArrowRight"
                  ? (index + 1) % tabs.length
                  : event.key === "ArrowLeft"
                    ? (index + tabs.length - 1) % tabs.length
                    : event.key === "Home"
                      ? 0
                      : event.key === "End"
                        ? tabs.length - 1
                        : -1;
              if (nextIndex < 0) return;
              event.preventDefault();
              setTab(tabs[nextIndex].id);
              document
                .getElementById(`editor-tab-${tabs[nextIndex].id}`)
                ?.focus();
            }}
          >
            <Icon size={15} />
            <span>{label}</span>
          </button>
        ))}
      </div>
      <div
        className="editor-sidebar-content"
        role="tabpanel"
        id={`editor-section-${tab}`}
        aria-labelledby={`editor-tab-${tab}`}
      >
        {tab === "mission" && (
          <div className="editor-mission">
            <div className="editor-panel-intro">
              <h3>Mission brief</h3>
              <p>Start from a sample or design your own collective.</p>
            </div>
            <MissionSamplePicker
              samples={samples}
              value={sample}
              onChange={chooseSample}
            />
            <label>
              Mission goal
              <textarea
                value={draft.goal}
                rows={19}
                onChange={(event) =>
                  change({ ...draft, goal: event.target.value })
                }
                placeholder="What should this collective deliver?"
              />
            </label>
            <p className="editor-panel-note">
              The full mission is passed to the coordinator. Include
              deliverables, evidence requirements, and acceptance criteria.
            </p>
          </div>
        )}
        {tab === "agents" && (
          <TeamRoster draft={draft} selected={selected} select={select} />
        )}
        {tab === "settings" && (
          <RuntimeSettings draft={draft} change={change} />
        )}
        {tab === "paths" && (
          <CommunicationPaths draft={draft} change={change} />
        )}
        {tab === "library" && (
          <div className="editor-library-panel">
            <div className="editor-panel-intro">
              <h3>Configuration library</h3>
              <p>
                Save the complete mission, team, paths, and node positions
                together.
              </p>
            </div>
            <label>
              Configuration name
              <input
                value={draft.name}
                onChange={(event) =>
                  change({ ...draft, name: event.target.value })
                }
                placeholder="My collective"
              />
            </label>
            <label>
              Load saved
              <select
                aria-label="Load saved configuration"
                value={draft.id || ""}
                onChange={(event) => load(event.target.value)}
              >
                <option value="">Choose configuration…</option>
                {configurations.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.name}
                  </option>
                ))}
              </select>
            </label>
            <p className="editor-panel-note">
              Use Save in the top toolbar to update this configuration. The copy
              button saves a separate version. Export and import use portable
              JSON files.
            </p>
            <p className="editor-panel-note">
              Loading a configuration replaces the open draft. A sample starts a
              new draft, leaving your saved configurations available here.
            </p>
          </div>
        )}
      </div>
    </>
  );
}
