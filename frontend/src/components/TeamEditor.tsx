import { useEffect, useRef, useState } from "react";
import {
  GitBranch,
  LayoutGrid,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Play,
  Plus,
  X,
} from "lucide-react";
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
import { layoutEditorTeam } from "../graph/editorLayout";
import type { Agent, TeamConfiguration } from "../types";
import { applyMissionSample } from "../missionSamples";
import { download } from "./ui";
import { AgentProperties } from "./AgentProperties";
import { ConfigurationLibrary } from "./ConfigurationLibrary";
import { EditorCanvas } from "./EditorCanvas";
import {
  CommunicationMode,
  EditorSidebar,
  type EditorTab,
} from "./EditorSidebar";
import "../editor.css";

const fallbackTools = [
  "list_files",
  "read_file",
  "write_file",
  "assemble_files",
  "hire",
  "fire",
  "monitor_agents",
  "instruct_agent",
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
  const [tab, setTab] = useState<EditorTab>("mission");
  const [leftOpen, setLeftOpen] = useState(() => window.innerWidth > 1000);
  const [rightOpen, setRightOpen] = useState(
    () => window.innerWidth > 1360 && draft.settings.team.length <= 7,
  );
  const [viewportRevision, setViewportRevision] = useState(0);
  const [focusRequest, setFocusRequest] = useState({ id: "", revision: 0 });
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
  const resetFocus = () =>
    setFocusRequest((request) => ({ id: "", revision: request.revision + 1 }));
  const select = (name: string, focus = false) => {
    setSelected(name);
    if (!focus && name !== selected) resetFocus();
    if (name) {
      setRightOpen(true);
      if (window.innerWidth <= 1280) setLeftOpen(false);
      if (focus)
        setFocusRequest((request) => ({
          id: name,
          revision: request.revision + 1,
        }));
    }
  };
  const showTab = (next: EditorTab) => {
    setTab(next);
    setLeftOpen(true);
    if (window.innerWidth <= 1000) setRightOpen(false);
  };
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
      if (fields.name !== undefined) {
        setSelected(fields.name);
        resetFocus();
      }
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
  const load = (id: string) => {
    const entry = configurations.find(
      (configuration) => configuration.id === id,
    );
    if (!id) {
      change({ ...draft, id: undefined });
      return;
    }
    if (entry) {
      change(structuredClone(entry));
      resetFocus();
      setViewportRevision((revision) => revision + 1);
      setSelected(entry.settings.coordinator);
      setSample("");
    }
  };
  const create = () =>
    void act(async () => {
      if (errors.length) throw new Error(errors[0]);
      if (!draft.goal.trim()) {
        showTab("mission");
        throw new Error("Enter a mission goal.");
      }
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
        <h2>
          <GitBranch size={21} />
          <span>
            Edit collective<small>COLLECTIVE DESIGNER</small>
          </span>
        </h2>
        <ConfigurationLibrary
          draft={draft}
          configurations={configurations}
          busy={busy}
          changeName={(name) => change({ ...draft, name })}
          load={load}
          save={save}
          remove={() =>
            void act(async () => {
              if (!draft.id) return;
              await deleteConfiguration(draft.id);
              setConfigurations(await listConfigurations());
              setDraft({ ...draft, id: undefined });
              setNotice(
                "Saved configuration removed. Your draft is still open.",
              );
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
              const imported = parseConfiguration(await file.text());
              change(imported);
              resetFocus();
              setViewportRevision((revision) => revision + 1);
              setSelected(imported.settings.coordinator);
              setSample("");
              setNotice(
                "Configuration imported. Save it to keep it in this server's library.",
              );
            })
          }
        />
        <button
          type="button"
          className="icon-button"
          onClick={close}
          aria-label="Close editor"
        >
          <X size={20} />
        </button>
      </header>
      <div className="editor-toolbar">
        <button
          type="button"
          className="icon-button"
          aria-label={leftOpen ? "Hide design panel" : "Show design panel"}
          aria-expanded={leftOpen}
          onClick={() => {
            setLeftOpen(!leftOpen);
            if (!leftOpen && window.innerWidth <= 1000) setRightOpen(false);
          }}
        >
          {leftOpen ? (
            <PanelLeftClose size={17} />
          ) : (
            <PanelLeftOpen size={17} />
          )}
        </button>
        <button
          type="button"
          className="button"
          onClick={() => showTab("mission")}
        >
          Mission & samples
        </button>
        <button
          type="button"
          className="button"
          onClick={() => {
            const next = addAgent(draft);
            change(next);
            select(next.settings.team.at(-1)!.name, true);
          }}
        >
          <Plus size={14} />
          Add agent
        </button>
        <button
          type="button"
          className="button"
          onClick={() => {
            change({ ...draft, positions: layoutEditorTeam(draft.settings) });
            resetFocus();
            setViewportRevision((revision) => revision + 1);
          }}
        >
          <LayoutGrid size={14} />
          Auto layout
        </button>
        <CommunicationMode draft={draft} change={change} />
        <button
          type="button"
          className="button editor-paths-button"
          onClick={() => showTab("paths")}
        >
          <GitBranch size={14} />
          {draft.settings.communication_links?.length ?? "All"} paths
        </button>
        <span className="editor-agent-count">
          {draft.settings.team.length} agents
        </span>
        <button
          type="button"
          className="icon-button editor-inspector-toggle"
          aria-label={
            rightOpen ? "Hide agent inspector" : "Show agent inspector"
          }
          aria-expanded={rightOpen}
          onClick={() => {
            setRightOpen(!rightOpen);
            if (!rightOpen && window.innerWidth <= 1000) setLeftOpen(false);
          }}
        >
          {rightOpen ? (
            <PanelRightClose size={17} />
          ) : (
            <PanelRightOpen size={17} />
          )}
        </button>
      </div>
      <div
        className={`editor-body ${leftOpen ? "left-open" : ""} ${rightOpen ? "right-open" : ""}`}
      >
        <aside className="editor-sidebar" aria-label="Collective design panel">
          <button
            type="button"
            className="icon-button editor-panel-close"
            aria-label="Close design panel"
            onClick={() => setLeftOpen(false)}
          >
            <X size={16} />
          </button>
          <EditorSidebar
            tab={tab}
            setTab={setTab}
            draft={draft}
            change={change}
            selected={selected}
            select={(name) => select(name, true)}
            samples={room.missionSamples}
            sample={sample}
            configurations={configurations}
            load={load}
            chooseSample={(id) => {
              setSample(id);
              const selectedSample = room.missionSamples.find(
                (entry) => entry.id === id,
              );
              if (selectedSample) {
                const applied = applyMissionSample(
                  selectedSample,
                  draft.settings,
                );
                change({
                  ...draft,
                  ...applied,
                  id: undefined,
                  name: selectedSample.title,
                  positions: applied.positions ?? draft.positions,
                });
                setSelected(applied.settings.coordinator);
                resetFocus();
                if (applied.settings.team.length > 7) setRightOpen(false);
                setViewportRevision((revision) => revision + 1);
              }
            }}
          />
        </aside>
        <div className="editor-workspace">
          <EditorCanvas
            draft={draft}
            change={change}
            selected={selected}
            select={select}
            viewportRevision={viewportRevision}
            focusRequest={focusRequest}
            onPathCreated={() => showTab("paths")}
          />
        </div>
        <aside className="editor-inspector" aria-label="Agent inspector panel">
          <button
            type="button"
            className="icon-button editor-panel-close"
            aria-label="Close agent inspector"
            onClick={() => setRightOpen(false)}
          >
            <X size={16} />
          </button>
          <AgentProperties
            agent={agent}
            draft={draft}
            tools={tools}
            update={update}
            remove={() => {
              change(removeAgent(draft, selected));
              setSelected("");
              resetFocus();
            }}
          />
        </aside>
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
              A new run uses this draft. Your current run keeps its
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
