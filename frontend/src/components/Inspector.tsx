import { useEffect, useState } from "react";
import {
  ArrowUpRight,
  Braces,
  CheckCheck,
  Download,
  Eye,
  MessageSquare,
  RotateCcw,
  Save,
  Wrench,
  X,
} from "lucide-react";
import { useRoom } from "../store";
import { api } from "../api";
import { canAdmin, nice, records, time } from "../selectors";
import { Badge, Empty, JsonTree, download } from "./ui";
export function Inspector({
  admin,
}: {
  admin: (action: string, fields?: Record<string, string>) => void;
}) {
  const { snapshot, defaults, selection, select, events, intervene } =
    useRoom();
  const config = snapshot?.manifest.settings || defaults,
    allowed = canAdmin(snapshot);
  const [artifact, setArtifact] = useState(""),
    [loading, setLoading] = useState(false),
    [error, setError] = useState(""),
    [saving, setSaving] = useState(false);
  useEffect(() => {
    if (selection?.type !== "artifact" || !snapshot) return;
    let cancelled = false;
    setLoading(true);
    setError("");
    api<{ content: string }>(
      `/api/runs/${snapshot.manifest.run_id}/artifact?path=${encodeURIComponent(selection.id)}`,
    )
      .then((r) => {
        if (!cancelled) setArtifact(r.content);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selection?.type, selection?.id, snapshot?.manifest.run_id]);
  const task = snapshot?.state.tasks?.[selection?.id || ""];
  const agent = config?.team.find((a) => a.name === selection?.id);
  const event = events.find((e) => e.id === selection?.id);
  const save = async () => {
    setSaving(true);
    setError("");
    try {
      await intervene("replace_artifact", {
        path: selection!.id,
        content: artifact,
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };
  return (
    <aside className={`inspector ${selection ? "open" : ""}`}>
      <header className="section-heading">
        <span>
          <Eye size={14} />
          INSPECTOR
        </span>
        <small>GOD’S EYE</small>
        <button
          className="icon-button"
          aria-label="Close inspector"
          onClick={() => select(null)}
        >
          <X size={16} />
        </button>
      </header>
      <div className="inspector-body">
        {!selection && (
          <>
            <div className="inspector-welcome">
              <Eye size={32} />
              <h3>See the whole picture.</h3>
              <p>
                Select an agent, task or flowing connection to inspect its real
                state and intervene.
              </p>
            </div>
            <section>
              <h4>CONTROL SHORTCUTS</h4>
              <p>
                <kbd>Space</kbd> Run / pause
              </p>
              <p>
                <kbd>.</kbd> One graph step
              </p>
              <p>Drag nodes to arrange the graph. Scroll to zoom.</p>
            </section>
            <section>
              <h4>LIVE BY DESIGN</h4>
              <p>
                125 ms event polling captures fast transitions. Both bottom
                panels are independent.
              </p>
            </section>
          </>
        )}
        {selection?.type === "agent" && agent && (
          <>
            <div className="detail-heading">
              <span className="detail-avatar">
                {agent.name.slice(0, 2).toUpperCase()}
              </span>
              <div>
                <h3>{agent.name}</h3>
                <p>{agent.role}</p>
              </div>
            </div>
            <div className="metric-grid">
              <span>
                <strong>
                  {
                    Object.values(snapshot?.state.tasks || {}).filter(
                      (t) => t.owner === agent.name,
                    ).length
                  }
                </strong>
                Assigned tasks
              </span>
              <span>
                <strong>
                  {
                    events.filter(
                      (e) => e.kind === "model" && e.agent === agent.name,
                    ).length
                  }
                </strong>
                Observed calls
              </span>
            </div>
            <button
              className="button primary"
              disabled={!allowed}
              onClick={() => admin("inject_message", { recipient: agent.name })}
            >
              <MessageSquare size={14} />
              Send instruction
            </button>
            <section>
              <h4>PERSONA</h4>
              <p>{agent.persona}</p>
            </section>
            <section>
              <h4>SKILLS & TOOLS</h4>
              <div className="tags">
                {agent.skills.map((s) => (
                  <span key={s}>{s}</span>
                ))}
              </div>
              <div className="tags tool-tags">
                {agent.tools.map((t) => (
                  <span key={t}>
                    <Wrench size={10} />
                    {t}
                  </span>
                ))}
              </div>
            </section>
            <section>
              <h4>ASSIGNED WORK</h4>
              {Object.values(snapshot?.state.tasks || {})
                .filter((t) => t.owner === agent.name)
                .map((t) => (
                  <button
                    className="inspect-row"
                    key={t.id}
                    onClick={() => select({ type: "task", id: t.id })}
                  >
                    {t.title}
                    <Badge status={t.status} />
                  </button>
                ))}
            </section>
            <section>
              <h4>RECENT ACTIVITY</h4>
              {events
                .filter(
                  (e) => e.agent === agent.name || e.sender === agent.name,
                )
                .slice(-6)
                .reverse()
                .map((e) => (
                  <button
                    className="inspect-row"
                    key={e.id}
                    onClick={() => select({ type: "event", id: e.id })}
                  >
                    <span>
                      {nice(e.kind)}
                      <small>
                        {e.tool || e.purpose || e.node || time(e.at)}
                      </small>
                    </span>
                    <ArrowUpRight size={12} />
                  </button>
                ))}
            </section>
          </>
        )}
        {selection?.type === "task" && task && (
          <>
            <h3>{task.title}</h3>
            <p className="subtitle">
              {task.id} · {task.phase}
            </p>
            <Badge status={task.status} />
            <section>
              <h4>ASSIGNMENT</h4>
              <p>{task.description}</p>
              <dl>
                <dt>Owner</dt>
                <dd>{task.owner}</dd>
                <dt>Output</dt>
                <dd>{task.output}</dd>
                <dt>Attempts</dt>
                <dd>{task.attempts || 0}</dd>
              </dl>
            </section>
            <section>
              <h4>ACCEPTANCE CRITERIA</h4>
              <ul>
                {task.acceptance_criteria.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </section>
            <section>
              <h4>DEPENDENCIES</h4>
              {task.dependencies.length ? (
                task.dependencies.map((d) => (
                  <button
                    className="text-button"
                    key={d}
                    onClick={() => select({ type: "task", id: d })}
                  >
                    {d} ↗
                  </button>
                ))
              ) : (
                <p>No prerequisites</p>
              )}
            </section>
            {task.feedback && (
              <section>
                <h4>
                  <CheckCheck size={13} />
                  INDEPENDENT REVIEW
                </h4>
                <p>{task.feedback}</p>
              </section>
            )}
            <div className="action-row">
              <button
                className="button"
                disabled={!allowed}
                onClick={() => admin("reassign_task", { task_id: task.id })}
              >
                Reassign
              </button>
              <button
                className="button primary"
                disabled={!allowed}
                onClick={() => admin("retry_task", { task_id: task.id })}
              >
                <RotateCcw size={13} />
                Reopen
              </button>
            </div>
            <button
              className="button wide"
              onClick={() => select({ type: "artifact", id: task.output })}
            >
              <ArrowUpRight size={14} />
              Open artifact
            </button>
          </>
        )}
        {selection?.type === "event" && event && (
          <>
            <h3>{nice(event.kind)}</h3>
            <p className="subtitle">
              {event.agent || event.sender || "collective"} · {time(event.at)}
            </p>
            <section>
              <h4>ACTUAL EVENT PAYLOAD</h4>
              <pre className="json-view">{JSON.stringify(event, null, 2)}</pre>
            </section>
          </>
        )}
        {selection?.type === "artifact" && (
          <>
            <h3>{selection.id.split("/").at(-1)}</h3>
            <p className="subtitle">{selection.id}</p>
            {loading ? (
              <Empty title="Reading artifact…" />
            ) : (
              <>
                <div className="artifact-stats">
                  {new TextEncoder().encode(artifact).length} bytes ·{" "}
                  {artifact.trim().split(/\s+/).filter(Boolean).length} words
                </div>
                <textarea
                  className="artifact-editor"
                  aria-label="Artifact contents"
                  value={artifact}
                  readOnly={!allowed}
                  onChange={(e) => setArtifact(e.target.value)}
                  spellCheck={false}
                />
                <div className="action-row">
                  <button
                    className="button primary"
                    disabled={!allowed || saving}
                    onClick={() => void save()}
                  >
                    <Save size={13} />
                    Save & reopen
                  </button>
                  <button
                    className="button"
                    onClick={() =>
                      download(selection.id.split("/").at(-1)!, artifact)
                    }
                  >
                    <Download size={13} />
                    Download
                  </button>
                </div>
                <p className="help">
                  Saving reopens this task and its dependent work for review.
                </p>
              </>
            )}
            {error && <p className="error-text">{error}</p>}
          </>
        )}
        {selection?.type === "config" && (
          <>
            <h3>Runtime configuration</h3>
            <section>
              <JsonTree value={config || {}} label="settings" />
            </section>
          </>
        )}
        {selection?.type === "channel" &&
          (() => {
            const [source, target] = selection.id.split("→");
            const messages = records("messages", "message").filter(
              (m) =>
                m.sender === source &&
                (m.recipient === target || m.recipient === "all"),
            );
            return (
              <>
                <h3>
                  {source} → {target}
                </h3>
                <p className="subtitle">
                  Communication channel · {messages.length} messages
                </p>
                {messages
                  .slice()
                  .reverse()
                  .map((m) => (
                    <section key={m.id}>
                      <h4>{time(m.at)}</h4>
                      <p>{m.content}</p>
                    </section>
                  ))}
                {!messages.length && (
                  <Empty title="Workflow handoff">
                    Accepted work unlocks the next agent’s tasks.
                  </Empty>
                )}
              </>
            );
          })()}
      </div>
      <footer>
        <span className={`runtime-dot ${allowed ? "" : "muted"}`} />
        {allowed
          ? "Interventions available"
          : snapshot?.running
            ? "Pause at a boundary to intervene"
            : "Live checkpoint inspection"}
      </footer>
    </aside>
  );
}
