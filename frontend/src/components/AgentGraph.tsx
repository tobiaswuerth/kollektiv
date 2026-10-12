import { memo, useMemo, useRef } from "react";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  ReactFlowProvider,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import {
  Code2,
  Expand,
  GitBranch,
  Network,
  Pencil,
  Search,
  ShieldCheck,
  UsersRound,
  Wrench,
  Zap,
} from "lucide-react";
import { useRoom } from "../store";
import {
  activeTeam,
  agentDesignation,
  actor,
  nice,
  specialistMode,
  specialists,
  tones,
} from "../selectors";
import { useClock } from "../hooks/useRuntime";
import type { Agent, RuntimeEvent, Specialist } from "../types";
import { Badge } from "./ui";
import { FlowEdge } from "../graph/FlowEdge";
import { GraphViewport } from "../graph/GraphViewport";
import { useGraphLayouts } from "../graph/layoutStore";
import { layoutRuntimeTeam, linkId, workflowLinks } from "../graph/model";
import { useNodeMeasurements } from "../graph/useNodeMeasurements";
import "../delegation.css";

type Person = Node<
  {
    agent: Agent;
    tone: string;
    mode: string;
    activity: string;
    tasks: number;
    completed: number;
    calls: number;
    tool?: RuntimeEvent;
    pulse?: RuntimeEvent;
    specialist?: Specialist;
    children: number;
    designation: string;
  },
  "person"
>;
const PersonCard = memo(({ data, selected }: NodeProps<Person>) => {
  const select = useRoom((room) => room.select);
  const Icon = data.designation.startsWith("Coordinator")
    ? Network
    : data.designation === "Reviewer"
      ? ShieldCheck
      : /review/.test(data.agent.role.toLowerCase())
        ? ShieldCheck
        : /research|architect/.test(data.agent.role.toLowerCase())
          ? Search
          : /coordinat/.test(data.agent.role.toLowerCase())
            ? Network
            : Code2;
  return (
    <div
      className={`person-card ${data.mode} ${selected ? "selected" : ""} ${data.pulse ? "event-pulse" : ""}`}
      style={{ "--tone": data.tone } as React.CSSProperties}
      data-agent={data.agent.name}
      data-mode={data.mode}
      data-parent={data.specialist?.parent}
      data-level={data.specialist?.level || 1}
    >
      <Handle type="target" position={Position.Top} id="in" />
      <Handle type="source" position={Position.Bottom} id="out" />
      <Handle type="source" position={Position.Right} id="tool-out" />
      <Handle type="target" position={Position.Right} id="tool-in" />
      <div className="person-heading">
        <span className="person-icon">
          <Icon size={19} />
        </span>
        <span>
          <strong>{data.agent.name}</strong>
          <small>{data.agent.role}</small>
        </span>
        <span className={`node-dot ${data.mode}`} />
      </div>
      <div className="person-activity" title={data.activity}>
        <Zap size={12} />
        {data.activity}
      </div>
      {data.designation === "Coordinator & reviewer" && (
        <div className="person-hierarchy">
          <Network size={11} />
          <span>{data.designation}</span>
        </div>
      )}
      {(data.specialist || data.children > 0) && (
        <div className="person-hierarchy">
          <GitBranch size={11} />
          <span>
            {data.specialist
              ? `Level ${data.specialist.level} · ${data.specialist.parent}`
              : `${data.children} specialist${data.children === 1 ? "" : "s"}`}
          </span>
          {data.specialist && data.children > 0 && (
            <small>{data.children} reports</small>
          )}
        </div>
      )}
      <footer>
        <span>
          {data.specialist
            ? nice(data.specialist.status)
            : `${data.completed}/${data.tasks} tasks`}
        </span>
        <span>{data.calls} calls</span>
      </footer>
      {data.tool && (
        <button
          className="tool-pill nodrag"
          onClick={(e) => {
            e.stopPropagation();
            select({ type: "event", id: data.tool!.id });
          }}
          title="Inspect the actual tool call"
        >
          <Wrench size={11} />
          <span>
            {data.tool.tool || data.tool.tool_calls?.[0]?.name || "tool call"}
          </span>
          <span
            className={data.tool.status === "error" ? "failure" : "success"}
          >
            {data.tool.status === "error" ? "!" : "↗"}
          </span>
        </button>
      )}
    </div>
  );
});
const nodeTypes = { person: PersonCard };
const edgeTypes = { flow: FlowEdge };

function GraphContent() {
  const { snapshot, defaults, events, pulses, selection, select, runId } =
    useRoom();
  const config = snapshot?.manifest.settings || defaults;
  const jobs = specialists(snapshot);
  const jobByName = new Map(jobs.map((agent) => [agent.name, agent]));
  const team = activeTeam(snapshot, config?.team || []),
    tasks = Object.values(snapshot?.state.tasks || {}),
    now = useClock();
  const key = `${runId || "defaults"}:${team.map((a) => a.name).join("|")}`;
  const container = useRef<HTMLDivElement>(null);
  const positions = useGraphLayouts(
    (state) => state.layouts[runId || "defaults"],
  );
  const moveNodes = useGraphLayouts((state) => state.moveNodes);
  const { measurements, acceptMeasurements } = useNodeMeasurements();
  const layoutKey = JSON.stringify([
    team.map((a) => a.name),
    config?.coordinator,
    config?.reviewer,
    config?.communication_links,
    jobs
      .filter((agent) => agent.status !== "retired")
      .map((agent) => [agent.name, agent.parent]),
    snapshot?.manifest.positions,
    positions,
  ]);
  const layout = useMemo(
    () =>
      config
        ? layoutRuntimeTeam(config, jobs, {
            ...snapshot?.manifest.positions,
            ...positions,
          })
        : {},
    [layoutKey],
  );
  const activePulses = pulses.filter((p) => p.start <= now && p.end >= now);
  const nodes: Person[] = team.map((agent, index) => {
    const specialist = jobByName.get(agent.name);
    const recent = activePulses
      .filter((p) => actor(p.event) === agent.name)
      .at(-1)?.event;
    const active =
      snapshot?.running &&
      snapshot.control?.active_nodes.find(
        (n) =>
          actor({ id: "", at: "", kind: "node_started", ...n }) === agent.name,
      );
    const tool = events
      .filter(
        (e) =>
          actor(e) === agent.name &&
          (e.kind === "tool_result" || !!e.tool_calls?.length),
      )
      .at(-1);
    const mode = specialist
      ? specialistMode(specialist)
      : active
        ? /review/.test(active.node)
          ? "reviewing"
          : "working"
        : snapshot?.running && snapshot.control?.mode === "paused"
          ? "paused"
          : snapshot?.running && recent
            ? /review/.test(recent.purpose || recent.node || "")
              ? "reviewing"
              : recent.status === "error" || recent.kind === "node_failed"
                ? "failed"
                : "working"
            : "idle";
    const assigned = tasks.filter((t) => t.owner === agent.name);
    const latest =
      recent ||
      events
        .filter(
          (e) =>
            actor(e) === agent.name &&
            [
              "model_started",
              "tool_result",
              "task_review",
              "node_started",
            ].includes(e.kind),
        )
        .at(-1);
    const activity = specialist
      ? specialist.current_node
        ? nice(specialist.current_node)
        : specialist.last_tool || specialist.summary || nice(specialist.status)
      : latest
        ? latest.tool ||
          latest.purpose ||
          nice(latest.node) ||
          nice(latest.kind)
        : snapshot?.state.status === "completed"
          ? "Mission complete"
          : "Ready to collaborate";
    return {
      id: agent.name,
      type: "person",
      position: positions?.[agent.name] ||
        snapshot?.manifest.positions?.[agent.name] ||
        layout[agent.name] || { x: 0, y: 0 },
      selected: selection?.type === "agent" && selection.id === agent.name,
      measured: measurements[agent.name],
      data: {
        agent,
        tone: tones[index % tones.length],
        mode,
        activity,
        tasks: assigned.length,
        completed: assigned.filter((t) => t.status === "completed").length,
        calls: events.filter(
          (e) => e.kind === "model" && e.agent === agent.name,
        ).length,
        tool,
        pulse: recent,
        specialist,
        children: jobs.filter(
          (child) => child.parent === agent.name && child.status !== "retired",
        ).length,
        designation: config ? agentDesignation(config, agent.name) : "",
      },
    };
  });
  const links = new Map<
    string,
    { source: string; target: string; count: number }
  >();
  const add = (source: string, target: string, count = 0) => {
    if (
      !team.some((a) => a.name === source) ||
      !team.some((a) => a.name === target) ||
      source === target
    )
      return;
    const id = linkId({ source, target });
    const link = links.get(id) || { source, target, count: 0 };
    link.count += count;
    links.set(id, link);
  };
  if (config)
    (config.communication_links ?? workflowLinks(config)).forEach(
      ({ source, target }) => add(source, target),
    );
  jobs
    .filter((agent) => agent.status !== "retired")
    .forEach((agent) => add(agent.parent, agent.name));
  events
    .filter((e) => e.kind === "message")
    .forEach((e) => {
      if (e.recipient === "all")
        team.forEach((a) => {
          if (
            e.system ||
            config?.communication_links == null ||
            config.communication_links.some(
              (link) => link.source === e.sender && link.target === a.name,
            )
          )
            add(e.sender!, a.name, 1);
        });
      else add(e.sender!, e.recipient!, 1);
    });
  const edges = [...links.entries()].map(([id, link]) => {
    const live = activePulses.filter(
      (p) =>
        p.event.kind === "message" &&
        p.event.sender === link.source &&
        (p.event.recipient === link.target || p.event.recipient === "all"),
    );
    const tone =
      tones[
        Math.max(
          0,
          team.findIndex((a) => a.name === link.source),
        ) % tones.length
      ];
    return {
      id,
      ...link,
      type: "flow",
      sourceHandle: "out",
      targetHandle: "in",
      markerEnd: { type: MarkerType.ArrowClosed, color: tone },
      data: {
        count: link.count,
        pulses: live,
        tone,
        hierarchy: jobByName.get(link.target)?.parent === link.source,
      },
    };
  });
  team.forEach((a, i) => {
    const live = activePulses.filter(
      (p) =>
        actor(p.event) === a.name &&
        (p.event.kind === "tool_result" || !!p.event.tool_calls?.length),
    );
    if (live.length)
      edges.push({
        id: `${a.name}:tool`,
        source: a.name,
        target: a.name,
        count: 0,
        type: "flow",
        sourceHandle: "tool-out",
        targetHandle: "tool-in",
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: tones[i % tones.length],
        },
        data: {
          count: 0,
          pulses: live,
          tone: tones[i % tones.length],
          hierarchy: false,
          label:
            live.at(-1)!.event.tool || live.at(-1)!.event.tool_calls?.[0]?.name,
        },
      } as (typeof edges)[number]);
  });
  return (
    <div ref={container} className="graph-canvas">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodeClick={(_, n) => select({ type: "agent", id: n.id })}
        onNodesChange={(changes) => {
          acceptMeasurements(changes);
          const moved = changes.filter(
            (c) => c.type === "position" && c.position,
          );
          if (moved.length) {
            const next: Record<string, { x: number; y: number }> = {};
            for (const change of moved)
              if (change.type === "position" && change.position)
                next[change.id] = change.position;
            moveNodes(runId || "defaults", next);
          }
        }}
        onEdgeClick={(_, edge) => select({ type: "channel", id: edge.id })}
        nodesConnectable={false}
        deleteKeyCode={null}
        minZoom={0.12}
        maxZoom={1.8}
        colorMode="dark"
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#27314c" gap={22} size={1} />
        <Controls showInteractive={false} />
        <GraphViewport
          container={container}
          identity={key}
          count={team.length}
        />
      </ReactFlow>
    </div>
  );
}

export function AgentGraph({ onEdit }: { onEdit?: () => void }) {
  const { snapshot, defaults, events, runId, select } = useRoom();
  const jobs = specialists(snapshot);
  const teamKey =
    (snapshot?.manifest.settings || defaults)?.team
      .map((agent) => agent.name)
      .join("|") || "empty";
  const latest = events.at(-1);
  return (
    <section className="graph-card">
      <header className="graph-heading">
        <div>
          <span className="eyebrow">THE COLLECTIVE</span>
          <h2>
            <GitBranch size={16} />
            Agent graph{" "}
            <span className="count">
              {activeTeam(snapshot, defaults?.team || []).length}
            </span>
          </h2>
        </div>
        <div className="graph-phase">
          {jobs.length > 0 && (
            <button
              className="button"
              aria-label="Inspect specialist history"
              onClick={() => select({ type: "delegation", id: "all" })}
            >
              <UsersRound size={13} />
              {jobs.length} specialists
            </button>
          )}
          {onEdit && (
            <button className="button" onClick={onEdit} disabled={!defaults}>
              <Pencil size={13} />
              Edit team
            </button>
          )}
          <Badge
            status={
              snapshot?.running
                ? snapshot.control?.paused && snapshot.control.waiting
                  ? "paused"
                  : "working"
                : snapshot?.state.status || "ready"
            }
          />
          <button
            className="icon-button"
            aria-label="Full screen graph"
            onClick={() => {
              const view = document.querySelector(".graph-card")!;
              if (document.fullscreenElement) void document.exitFullscreen();
              else void view.requestFullscreen();
            }}
          >
            <Expand size={15} />
          </button>
        </div>
      </header>
      <div className="graph-viewport">
        <ReactFlowProvider key={`${runId || "defaults"}:${teamKey}`}>
          <GraphContent />
        </ReactFlowProvider>
        <div className="graph-legend">
          <span>
            <i className="working" />
            Working
          </span>
          <span>
            <i className="reviewing" />
            Reviewing
          </span>
          <span>
            <i className="paused" />
            Paused
          </span>
          <span>→ Information flow</span>
          {jobs.length > 0 && (
            <span>
              <GitBranch size={11} /> Specialist reports
            </span>
          )}
        </div>
      </div>
      <div className="graph-event">
        <span className="live-dot" />
        <span>
          {latest
            ? `${actor(latest) || "collective"} · ${nice(latest.kind)}${latest.tool ? ` · ${latest.tool}` : ""}`
            : "Actual agent activity, tools and messages appear here"}
        </span>
        <small>
          {events.filter((e) => e.kind === "node_finished").length} steps
        </small>
      </div>
    </section>
  );
}
