import { useEffect, useRef, useState } from "react";
import { GitBranch, LayoutGrid, Play, Plus, Trash2, X } from "lucide-react";
import { api } from "../api";
import { useRoom } from "../store";
import { useGraphLayouts } from "../graph/layoutStore";
import {
  addAgent,
  configurationErrors,
  draftFrom,
  parseConfiguration,
  removeAgent,
  updateAgent,
} from "../graph/editorModel";
import {
  deleteConfiguration,
  listConfigurations,
  saveConfiguration,
} from "../graph/configurations";
import { layoutTeam, linkId, workflowLinks } from "../graph/model";
import type { Agent, TeamConfiguration } from "../types";
import { applyMissionSample } from "../missionSamples";
import { download } from "./ui";
import { AgentProperties } from "./AgentProperties";
import { ConfigurationLibrary } from "./ConfigurationLibrary";
import { EditorCanvas } from "./EditorCanvas";
import { MissionSamplePicker } from "./MissionSamplePicker";
import "../editor.css";

const fallbackTools = [
  "list_files",
  "read_file",
  "write_file",
  "check_file",
  "send_message",
  "share_note",
  "web_search",
  "web_browse",
];

export function TeamEditor({ close }: { close: () => void }) {
  const room = useRoom();
  const [draft, setDraft] = useState(() =>
    draftFrom(
      room.snapshot?.manifest.settings || room.defaults!,
      room.snapshot?.manifest.goal || "",
      {
        ...room.snapshot?.manifest.positions,
        ...useGraphLayouts.getState().layouts[room.runId || "defaults"],
      },
    ),
  );
  const [selected, setSelected] = useState(draft.settings.team[0]?.name || "");
  const [configurations, setConfigurations] = useState<TeamConfiguration[]>([]);
  const [tools, setTools] = useState(fallbackTools);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [sample, setSample] = useState("");
  const [viewportRevision, setViewportRevision] = useState(0);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current!;
    element.showModal();
    let disposed = false;
    void Promise.allSettled([
      listConfigurations(),
      api<{ tools?: string[] }>("/api/bootstrap"),
    ]).then(([library, bootstrap]) => {
      if (disposed) return;
      if (library.status === "fulfilled") setConfigurations(library.value);
      else setError(String(library.reason));
      if (bootstrap.status === "fulfilled" && bootstrap.value.tools?.length)
        setTools(bootstrap.value.tools);
    });
    return () => {
      disposed = true;
      element.close();
    };
  }, []);
  const change = (next: TeamConfiguration) => {
    setDraft(next);
    setNotice("");
    setError("");
  };
  const agent = draft.settings.team.find((entry) => entry.name === selected);
  const errors = configurationErrors(draft, tools);
  const act = async (job: () => Promise<void>) => {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      await job();
    } catch (failure) {
      setError(String(failure));
    } finally {
      setBusy(false);
    }
  };
  const update = (fields: Partial<Agent>) => {
    try {
      change(updateAgent(draft, selected, fields));
      if (fields.name !== undefined) setSelected(fields.name);
      return true;
    } catch (failure) {
      setError(String(failure));
      return false;
    }
  };
  const save = (copy = false) =>
    void act(async () => {
      if (errors.length) throw new Error(errors[0]);
      const saved = await saveConfiguration({
        ...draft,
        ...(copy ? { id: undefined } : {}),
      });
      setDraft(saved);
      setConfigurations(await listConfigurations());
      setNotice(`Saved ${saved.name}.`);
    });
  const create = () =>
    void act(async () => {
      if (errors.length) throw new Error(errors[0]);
      if (!draft.goal.trim()) throw new Error("Enter a mission goal.");
      await useRoom
        .getState()
        .createRun(draft.goal, draft.settings, draft.positions);
      close();
    });
  return (
    <dialog
      ref={dialog}
      className="team-editor"
      aria-label="Edit collective"
      onCancel={close}
    >
      <header className="editor-header">
        <div>
          <span className="eyebrow">COLLECTIVE DESIGNER</span>
          <h2>
            <GitBranch size={20} />
            Edit collective
          </h2>
        </div>
        <span className="editor-draft-badge">Configuration draft</span>
        <button
          type="button"
          className="icon-button"
          onClick={close}
          aria-label="Close editor"
        >
          <X size={19} />
        </button>
      </header>
      <ConfigurationLibrary
        draft={draft}
        configurations={configurations}
        busy={busy}
        changeName={(name) => change({ ...draft, name })}
        load={(id) => {
          const entry = configurations.find(
            (configuration) => configuration.id === id,
          );
          if (!id) {
            change({ ...draft, id: undefined });
            return;
          }
          if (entry) {
            change(structuredClone(entry));
            setViewportRevision((revision) => revision + 1);
            setSelected(entry.settings.team[0]?.name || "");
            setSample("");
          }
        }}
        save={save}
        remove={() =>
          void act(async () => {
            if (!draft.id) return;
            await deleteConfiguration(draft.id);
            setConfigurations(await listConfigurations());
            setDraft({ ...draft, id: undefined });
            setNotice("Saved configuration removed. Your draft is still open.");
          })
        }
        exportFile={() =>
          download(
            `${draft.name.replace(/[^a-z0-9_-]/gi, "_") || "collective"}.json`,
            JSON.stringify(draft, null, 2),
          )
        }
        importFile={(file) =>
          void act(async () => {
            change(parseConfiguration(await file.text()));
            setViewportRevision((revision) => revision + 1);
            setSelected("");
            setSample("");
            setNotice(
              "Configuration imported. Save it to keep it in this server's library.",
            );
          })
        }
      />
      <div className="editor-mission">
        <label>
          Mission goal
          <textarea
            value={draft.goal}
            rows={2}
            onChange={(event) => change({ ...draft, goal: event.target.value })}
            placeholder="What should this collective deliver?"
          />
        </label>
        <MissionSamplePicker
          samples={room.missionSamples}
          value={sample}
          onChange={(id) => {
            setSample(id);
            const selectedSample = room.missionSamples.find(
              (entry) => entry.id === id,
            );
            if (selectedSample)
              change({
                ...draft,
                ...applyMissionSample(selectedSample, draft.settings),
              });
          }}
        />
      </div>
      <div className="editor-body">
        <div className="editor-workspace">
          <div className="editor-toolbar">
            <button
              type="button"
              className="button"
              onClick={() => {
                const next = addAgent(draft);
                change(next);
                setSelected(next.settings.team.at(-1)!.name);
              }}
            >
              <Plus size={14} />
              Add agent
            </button>
            <button
              type="button"
              className="button"
              onClick={() => {
                change({ ...draft, positions: layoutTeam(draft.settings) });
                setViewportRevision((revision) => revision + 1);
              }}
            >
              <LayoutGrid size={14} />
              Auto layout
            </button>
            <label>
              Messaging
              <select
                aria-label="Communication mode"
                value={
                  draft.settings.communication_links == null
                    ? "unrestricted"
                    : "custom"
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
          </div>
          <EditorCanvas
            draft={draft}
            change={change}
            selected={selected}
            select={setSelected}
            viewportRevision={viewportRevision}
          />
          <div className="editor-channels">
            <header>
              <strong>Communication paths</strong>
              <small>
                {draft.settings.communication_links == null
                  ? "All agents may communicate"
                  : "Only these directed agent message channels are allowed"}
              </small>
            </header>
            {draft.settings.communication_links == null ? (
              <p>
                Draw a path to switch to explicit channels, or choose Directed
                paths for an initial coordinator → workers → reviewer layout.
              </p>
            ) : draft.settings.communication_links.length ? (
              <div className="editor-channel-list">
                {draft.settings.communication_links.map((link) => (
                  <button
                    type="button"
                    key={linkId(link)}
                    onClick={() =>
                      change({
                        ...draft,
                        settings: {
                          ...draft.settings,
                          communication_links:
                            draft.settings.communication_links!.filter(
                              (entry) => linkId(entry) !== linkId(link),
                            ),
                        },
                      })
                    }
                    aria-label={`Remove path ${link.source} to ${link.target}`}
                  >
                    <span>
                      {link.source} → {link.target}
                    </span>
                    <Trash2 size={12} />
                  </button>
                ))}
              </div>
            ) : (
              <p>
                No agent message channels are enabled. Draw paths between agent
                handles to add them.
              </p>
            )}
            <small>
              Task scheduling and required review feedback remain part of the
              execution workflow.
            </small>
          </div>
        </div>
        <AgentProperties
          agent={agent}
          draft={draft}
          tools={tools}
          update={update}
          remove={() => {
            change(removeAgent(draft, selected));
            setSelected("");
          }}
        />
      </div>
      <div className="editor-runtime">
        <label>
          Coordinator
          <select
            value={draft.settings.coordinator}
            onChange={(event) =>
              change({
                ...draft,
                settings: {
                  ...draft.settings,
                  coordinator: event.target.value,
                },
              })
            }
          >
            {draft.settings.team.map((entry) => (
              <option key={entry.name} value={entry.name}>
                {entry.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Reviewer
          <select
            value={draft.settings.reviewer}
            onChange={(event) =>
              change({
                ...draft,
                settings: { ...draft.settings, reviewer: event.target.value },
              })
            }
          >
            {draft.settings.team.map((entry) => (
              <option key={entry.name} value={entry.name}>
                {entry.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Default model
          <input
            value={draft.settings.model}
            onChange={(event) =>
              change({
                ...draft,
                settings: { ...draft.settings, model: event.target.value },
              })
            }
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
              change({
                ...draft,
                settings: {
                  ...draft.settings,
                  max_concurrency: Number(event.target.value),
                },
              })
            }
          />
        </label>
        <label className="editor-web">
          <input
            type="checkbox"
            checked={draft.settings.web_enabled}
            onChange={(event) =>
              change({
                ...draft,
                settings: {
                  ...draft.settings,
                  web_enabled: event.target.checked,
                },
              })
            }
          />
          Allow web tools
        </label>
      </div>
      <footer className="editor-footer">
        <div>
          {error ? (
            <p role="alert" className="form-error">
              {error}
            </p>
          ) : notice ? (
            <p role="status" className="editor-notice">
              {notice}
            </p>
          ) : errors.length ? (
            <p className="editor-validation">{errors[0]}</p>
          ) : (
            <p>
              A new run uses this draft. The current run keeps its original
              configuration.
            </p>
          )}
        </div>
        <button type="button" className="button" onClick={close}>
          Close
        </button>
        <button
          type="button"
          className="button primary"
          onClick={create}
          disabled={busy || errors.length > 0 || !draft.goal.trim()}
        >
          <Play size={14} />
          {busy ? "Working…" : "Create collective"}
        </button>
      </footer>
    </dialog>
  );
}
