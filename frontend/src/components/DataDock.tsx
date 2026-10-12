import { useEffect, useRef, useState } from "react";
import {
  Archive,
  Braces,
  Copy,
  FileText,
  MessageSquare,
  Plus,
  Radio,
  Search,
} from "lucide-react";
import { useRoom } from "../store";
import { canAdmin, nice, records, time } from "../selectors";
import type { Tab } from "../types";
import { Empty, JsonTree, download } from "./ui";
const tabs: { id: Tab; label: string; icon: typeof MessageSquare }[] = [
  { id: "messages", label: "Messages", icon: MessageSquare },
  { id: "memory", label: "Memory", icon: Archive },
  { id: "debug", label: "Live debug", icon: Radio },
  { id: "state", label: "State", icon: Braces },
  { id: "artifacts", label: "Artifacts", icon: FileText },
];
function DataPane({
  id,
  initial,
  admin,
}: {
  id: string;
  initial: Tab;
  admin: (action: string) => void;
}) {
  const { snapshot, events, select, selection, runId } = useRoom();
  const [tab, setTab] = useState(initial),
    [query, setQuery] = useState(""),
    [kind, setKind] = useState("");
  const panel = useRef<HTMLDivElement>(null),
    follow = useRef(true);
  const messages = records("messages", "message"),
    memory = records("notes", "note");
  useEffect(() => {
    follow.current = true;
    if (panel.current)
      panel.current.scrollTop =
        tab === "messages" ? panel.current.scrollHeight : 0;
  }, [tab, runId]);
  useEffect(() => {
    if (tab === "messages" && follow.current && panel.current)
      panel.current.scrollTop = panel.current.scrollHeight;
  }, [messages.length, tab]);
  const filtered = events
    .filter(
      (e) =>
        (!kind || e.kind === kind) &&
        (!query ||
          JSON.stringify(e).toLowerCase().includes(query.toLowerCase())),
    )
    .slice(-350)
    .reverse();
  const kinds = [...new Set(events.map((e) => e.kind))].sort();
  return (
    <section className="dock-pane" data-pane={id}>
      <header>
        <span className="pane-number">{id === "primary" ? "01" : "02"}</span>
        <nav aria-label={`${id} panel views`}>
          {tabs.map(({ id: value, label, icon: Icon }) => (
            <button
              key={value}
              data-tab={value}
              className={tab === value ? "active" : ""}
              onClick={() => setTab(value)}
            >
              <Icon size={13} />
              {label}
              {value === "messages" ? (
                <small>{messages.length}</small>
              ) : value === "memory" ? (
                <small>{memory.length}</small>
              ) : null}
            </button>
          ))}
        </nav>
        {["messages", "memory"].includes(tab) && (
          <button
            className="icon-button"
            title={tab === "memory" ? "Share a note" : "Send an instruction"}
            aria-label={
              tab === "memory" ? "Share a note" : "Send an instruction"
            }
            disabled={!canAdmin(snapshot)}
            onClick={() =>
              admin(tab === "memory" ? "share_note" : "inject_message")
            }
          >
            <Plus size={15} />
          </button>
        )}
      </header>
      <div
        className={`pane-body ${tab}`}
        ref={panel}
        onScroll={() => {
          if (panel.current)
            follow.current =
              panel.current.scrollHeight -
                panel.current.scrollTop -
                panel.current.clientHeight <
              45;
        }}
      >
        {tab === "messages" &&
          (messages.length ? (
            messages.map((m) => (
              <article className="message" key={m.id}>
                <span className="sender-avatar">
                  {(m.sender || "?").slice(0, 2).toUpperCase()}
                </span>
                <div>
                  <header>
                    <strong>{m.sender}</strong>
                    <span>
                      → {m.recipient === "all" ? "everyone" : m.recipient}
                    </span>
                    <time>{time(m.at)}</time>
                  </header>
                  <p>{m.content}</p>
                </div>
              </article>
            ))
          ) : (
            <Empty title="The conversation starts here">
              Agent messages and handoffs appear as the collective works.
            </Empty>
          ))}
        {tab === "memory" &&
          (memory.length ? (
            memory
              .slice()
              .reverse()
              .map((m) => (
                <article className="memory-card" key={m.id}>
                  <header>
                    <Archive size={13} />
                    <strong>{m.sender}</strong>
                    <time>{time(m.at)}</time>
                  </header>
                  <p>{m.content}</p>
                </article>
              ))
          ) : (
            <Empty title="Shared knowledge">
              Conventions, decisions and lessons remain available to every
              agent.
            </Empty>
          ))}
        {tab === "debug" && (
          <>
            <div className="debug-filters">
              <label>
                <Search size={13} />
                <input
                  aria-label="Search debug events"
                  placeholder="Search agents, tools, events…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </label>
              <select
                aria-label="Filter event kind"
                value={kind}
                onChange={(e) => setKind(e.target.value)}
              >
                <option value="">All events</option>
                {kinds.map((k) => (
                  <option key={k} value={k}>
                    {nice(k)}
                  </option>
                ))}
              </select>
            </div>
            {filtered.length ? (
              filtered.map((e) => (
                <button
                  className={`debug-event ${e.status === "error" || e.error ? "failed" : ""} ${selection?.id === e.id ? "selected" : ""}`}
                  key={e.id}
                  data-event={e.id}
                  onClick={() => select({ type: "event", id: e.id })}
                >
                  <time>{time(e.at)}</time>
                  <span className={`event-kind ${e.kind}`}>{nice(e.kind)}</span>
                  <strong>{e.agent || e.sender || "collective"}</strong>
                  <span className="event-detail">
                    {e.tool ||
                      e.purpose ||
                      e.node ||
                      e.task ||
                      e.status ||
                      (e.action as string) ||
                      ""}
                  </span>
                </button>
              ))
            ) : (
              <Empty title="Waiting for runtime events">
                Every transition is recorded, including events between state
                polls.
              </Empty>
            )}
          </>
        )}
        {tab === "state" && (
          <>
            <div className="state-toolbar">
              <span>Actual LangGraph checkpoint</span>
              <button
                className="text-button"
                onClick={() =>
                  void navigator.clipboard
                    .writeText(JSON.stringify(snapshot?.state, null, 2))
                    .catch(() =>
                      download(
                        "state.json",
                        JSON.stringify(snapshot?.state, null, 2),
                      ),
                    )
                }
              >
                <Copy size={12} />
                Copy JSON
              </button>
            </div>
            <JsonTree value={snapshot?.state || {}} />
          </>
        )}
        {tab === "artifacts" &&
          (snapshot?.artifacts.length ? (
            snapshot.artifacts.map((file) => (
              <button
                key={file.path}
                className="artifact-row"
                data-artifact={file.path}
                onClick={() => select({ type: "artifact", id: file.path })}
              >
                <FileText size={15} />
                <span>
                  <strong>{file.path}</strong>
                  <small>{(file.bytes / 1024).toFixed(1)} KB</small>
                </span>
                <span>↗</span>
              </button>
            ))
          ) : (
            <Empty title="No artifacts yet">
              Completed files appear here. Select one to inspect or edit it.
            </Empty>
          ))}
      </div>
    </section>
  );
}
export function DataDock({ admin }: { admin: (action: string) => void }) {
  const [ratio, setRatio] = useState(50),
    [height, setHeight] = useState(32);
  const resizeCleanup = useRef<() => void>(() => {});
  useEffect(() => () => resizeCleanup.current(), []);
  const startResize = (e: React.PointerEvent, vertical: boolean) => {
    resizeCleanup.current();
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    const move = (event: PointerEvent) => {
      if (vertical)
        setHeight(
          Math.min(
            60,
            Math.max(22, ((innerHeight - event.clientY) / innerHeight) * 100),
          ),
        );
      else
        setRatio(
          Math.max(27, Math.min(73, (event.clientX / innerWidth) * 100)),
        );
    };
    const stop = () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", stop);
      document.removeEventListener("pointercancel", stop);
      document.body.classList.remove("resizing");
      resizeCleanup.current = () => {};
    };
    resizeCleanup.current = stop;
    document.body.classList.add("resizing");
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", stop, { once: true });
    document.addEventListener("pointercancel", stop, { once: true });
  };
  useEffect(() => {
    document.documentElement.style.setProperty("--dock-height", `${height}dvh`);
  }, [height]);
  return (
    <section
      className="data-dock"
      aria-label="Two independent data panels"
      style={{ "--split": `${ratio}%` } as React.CSSProperties}
    >
      <div
        className="dock-resize"
        role="separator"
        aria-label="Resize bottom dock"
        aria-orientation="horizontal"
        aria-valuemin={22}
        aria-valuemax={60}
        aria-valuenow={height}
        tabIndex={0}
        onPointerDown={(e) => startResize(e, true)}
        onKeyDown={(e) => {
          if (e.key === "ArrowUp") setHeight((h) => Math.min(60, h + 2));
          if (e.key === "ArrowDown") setHeight((h) => Math.max(22, h - 2));
        }}
      />
      <DataPane id="primary" initial="messages" admin={admin} />
      <div
        className="pane-divider"
        role="separator"
        aria-label="Resize panel split"
        aria-orientation="vertical"
        aria-valuemin={27}
        aria-valuemax={73}
        aria-valuenow={ratio}
        tabIndex={0}
        onPointerDown={(e) => startResize(e, false)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setRatio((r) => Math.max(27, r - 2));
          if (e.key === "ArrowRight") setRatio((r) => Math.min(73, r + 2));
        }}
      />
      <DataPane id="secondary" initial="debug" admin={admin} />
    </section>
  );
}
