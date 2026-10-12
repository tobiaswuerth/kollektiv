import {
  Check,
  Circle,
  CircleAlert,
  Cog,
  FileText,
  LockKeyhole,
  RefreshCw,
} from "lucide-react";
import { useRoom } from "../store";
import { canAdmin, taskStatus } from "../selectors";
import { Badge, Empty } from "./ui";
export function Overview({ editGoal }: { editGoal: () => void }) {
  const {
    snapshot,
    defaults,
    runs,
    runId,
    chooseRun,
    refreshRuns,
    select,
    selection,
  } = useRoom();
  const state = snapshot?.state,
    tasks = Object.values(state?.tasks || {});
  const done = tasks.filter((t) => t.status === "completed").length;
  const status = snapshot?.running
    ? snapshot.control?.paused && snapshot.control.waiting
      ? "paused"
      : snapshot.control?.mode === "paused"
        ? "pausing"
        : "working"
    : state?.status || "created";
  const phases = [...new Set(tasks.map((t) => t.phase))];
  return (
    <aside className="overview">
      <header className="section-heading">
        <span>MISSION CONTROL</span>
        <button
          className="icon-button"
          aria-label="Refresh collectives"
          onClick={() => void refreshRuns()}
        >
          <RefreshCw size={14} />
        </button>
      </header>
      <select
        id="run-select"
        aria-label="Select a collective"
        value={runId || ""}
        onChange={(e) => void chooseRun(e.target.value)}
      >
        <option value="">Choose a collective</option>
        {runs.map((r) => (
          <option key={r.run_id} value={r.run_id}>
            {r.goal.slice(0, 75)}
          </option>
        ))}
      </select>
      <section className="mission">
        <div className="section-heading">
          <span>SHARED GOAL</span>
          {snapshot && state?.status === "created" && !state.run_id && (
            <button
              className="text-button"
              disabled={!canAdmin(snapshot)}
              onClick={editGoal}
            >
              Edit
            </button>
          )}
        </div>
        <h1 title={snapshot?.manifest.goal}>
          {snapshot?.manifest.goal || "One goal. Many minds."}
        </h1>
        <div className="mission-meta">
          <Badge status={status} />
          {state?.goal_revision ? (
            <small>Revision {state.goal_revision}</small>
          ) : null}
        </div>
        <details className="plan-context">
          <summary>Plan & context</summary>
          <p>
            {state?.plan?.summary ||
              "Discuss a goal, divide the work, and review the result together."}
          </p>
        </details>
        <div className="progress-meta">
          <span>Accepted tasks</span>
          <strong>
            {done} / {tasks.length}
          </strong>
        </div>
        <div className="progress-track">
          <i
            style={{
              width: `${tasks.length ? (done / tasks.length) * 100 : 0}%`,
            }}
          />
        </div>
      </section>
      <section className="tasks-overview">
        <header className="section-heading">
          <span>TASKS & DEPENDENCIES</span>
          <small>{tasks.length}</small>
        </header>
        {tasks.length ? (
          phases.map((phase) => (
            <div className="task-phase" key={phase}>
              <div className="phase-label">{phase}</div>
              {tasks
                .filter((t) => t.phase === phase)
                .map((t) => {
                  const status = taskStatus(t, snapshot);
                  return (
                    <button
                      className={`task-row ${selection?.id === t.id ? "selected" : ""}`}
                      key={t.id}
                      data-task={t.id}
                      onClick={() => select({ type: "task", id: t.id })}
                    >
                      <span className={`task-indicator ${status}`}>
                        {status === "completed" ? (
                          <Check size={14} />
                        ) : status === "failed" ? (
                          <CircleAlert size={14} />
                        ) : status === "blocked" ? (
                          <LockKeyhole size={13} />
                        ) : (
                          <Circle size={13} />
                        )}
                      </span>
                      <span>
                        <strong>{t.title}</strong>
                        <small>
                          <FileText size={11} /> {t.output}
                        </small>
                        <span className="task-meta">
                          <span>{t.owner}</span>
                          <Badge status={status} />
                        </span>
                        {t.dependencies.length > 0 && (
                          <small className="task-deps">
                            Needs {t.dependencies.join(", ")}
                          </small>
                        )}
                      </span>
                    </button>
                  );
                })}
            </div>
          ))
        ) : (
          <Empty title="A plan comes first">
            Tasks and their dependencies appear after the team discussion.
          </Empty>
        )}
      </section>
      <footer className="overview-footer">
        <span className="runtime-dot" />
        <span>
          <strong>LOCAL RUNTIME</strong>
          <small>
            {(snapshot?.manifest.settings || defaults)?.model || "Ollama"}
          </small>
        </span>
        <button
          className="icon-button"
          aria-label="Inspect configuration"
          onClick={() => select({ type: "config", id: "config" })}
        >
          <Cog size={16} />
        </button>
      </footer>
    </aside>
  );
}
