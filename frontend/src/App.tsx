import { lazy, Suspense, useEffect, useState } from "react";
import {
  CircleAlert,
  Network,
  Pause,
  Play,
  Plus,
  Radio,
  RotateCcw,
  Square,
  StepForward,
  X,
} from "lucide-react";
import { useRoom } from "./store";
import { useRuntime } from "./hooks/useRuntime";
import { Overview } from "./components/Overview";
import { AgentGraph } from "./components/AgentGraph";
import { Inspector } from "./components/Inspector";
import { DataDock } from "./components/DataDock";
import { ActionDialog, type Action } from "./components/ActionDialog";
import { EditorBoundary } from "./components/EditorBoundary";
const TeamEditor = lazy(() =>
  import("./components/TeamEditor").then((module) => ({
    default: module.TeamEditor,
  })),
);
export function App() {
  useRuntime();
  const room = useRoom();
  const [action, setAction] = useState<Action | null>(null);
  const [editing, setEditing] = useState(false);
  const snapshot = room.snapshot,
    running = !!snapshot?.running,
    paused = snapshot?.control?.mode === "paused",
    completed = snapshot?.state.status === "completed";
  const admin = (kind: string, fields?: Record<string, string>) =>
    setAction({ kind, fields });
  const run = () =>
    void room.control(running ? (paused ? "resume" : "pause") : "resume");
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (
        (e.target as HTMLElement).closest(
          'input,textarea,select,button,[role="button"]',
        ) ||
        action ||
        editing ||
        e.ctrlKey ||
        e.metaKey ||
        e.altKey
      )
        return;
      if (e.code === "Space" && snapshot && !completed) {
        e.preventDefault();
        run();
      }
      if (
        e.key === "." &&
        snapshot &&
        !completed &&
        (!running || (snapshot.control?.paused && snapshot.control.waiting))
      )
        void room.control("step");
      if (e.key === "Escape") room.select(null);
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [action, editing, snapshot, completed, running]);
  return (
    <>
      <div className="app-shell">
        <header className="app-header">
          <a href="/" className="brand">
            <span>k.</span>kollektiv<small>04</small>
          </a>
          <span className="header-caption">
            <Network size={13} />
            AGENT CONTROL ROOM
          </span>
          <div className="header-end">
            <span
              className={`connection ${room.connected ? "" : "offline"}`}
              title={`Last request: ${room.latency} ms`}
            >
              <Radio size={13} />
              {!room.connected
                ? "Reconnecting"
                : room.health?.connected
                  ? running
                    ? "LIVE · 125 ms"
                    : "Ollama connected"
                  : "Ollama offline"}
            </span>
            <button
              id="new-run"
              className="button primary"
              disabled={!room.defaults}
              onClick={() => admin("create")}
            >
              <Plus size={14} />
              New collective
            </button>
          </div>
        </header>
        <Overview editGoal={() => admin("revise_goal")} />
        <main className="main">
          <AgentGraph onEdit={() => setEditing(true)} />
          <div className="transport">
            <div>
              <button
                id="run-control"
                className="button primary"
                disabled={!snapshot || room.busy || (completed && !running)}
                onClick={run}
              >
                {running && !paused ? <Pause size={14} /> : <Play size={14} />}
                {completed && !running
                  ? "Complete"
                  : running
                    ? paused
                      ? "Resume"
                      : "Pause"
                    : snapshot?.control?.paused
                      ? "Resume"
                      : "Run collective"}
              </button>
              <button
                id="step-control"
                className="button"
                disabled={
                  !snapshot ||
                  room.busy ||
                  completed ||
                  (running &&
                    !(snapshot.control?.paused && snapshot.control.waiting))
                }
                onClick={() => void room.control("step")}
              >
                <StepForward size={15} />
                Step
              </button>
              <button
                className="icon-button"
                aria-label="Stop collective"
                disabled={!running || room.busy}
                onClick={() => void room.control("stop")}
              >
                <Square size={13} />
              </button>
              {snapshot &&
                !running &&
                ["incomplete", "stopped", "interrupted"].includes(
                  snapshot.state.status || "",
                ) && (
                  <button
                    className="text-button"
                    onClick={() => void room.control("resume", true)}
                  >
                    <RotateCcw size={12} />
                    Retry
                  </button>
                )}
            </div>
            <small>
              {running
                ? paused
                  ? snapshot?.control?.paused && snapshot.control.waiting
                    ? "Paused at a node boundary"
                    : "Waiting for the current node…"
                  : "Every transition is checkpointed"
                : completed
                  ? "All deliverables reviewed"
                  : "One graph node per step"}
              <kbd>Space</kbd>
              <kbd>.</kbd>
            </small>
          </div>
        </main>
        <Inspector admin={admin} />
        <DataDock admin={admin} />
      </div>
      {room.error && (
        <div className="error-toast" role="alert">
          <CircleAlert size={16} />
          {room.error}
          <button
            className="icon-button"
            aria-label="Dismiss error"
            onClick={() => useRoom.setState({ error: "" })}
          >
            <X size={14} />
          </button>
        </div>
      )}
      {action && room.defaults && (
        <ActionDialog
          key={action.kind + (action.fields?.task_id || "")}
          action={action}
          close={() => setAction(null)}
        />
      )}
      {editing && room.defaults && (
        <EditorBoundary close={() => setEditing(false)}>
          <Suspense
            fallback={
              <div role="status" className="error-toast">
                Loading team editor…
              </div>
            }
          >
            <TeamEditor close={() => setEditing(false)} />
          </Suspense>
        </EditorBoundary>
      )}
    </>
  );
}
