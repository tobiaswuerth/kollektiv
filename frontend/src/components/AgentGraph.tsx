import { memo, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  BaseEdge,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  ReactFlowProvider,
  getBezierPath,
  useReactFlow,
  type EdgeProps,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import {
  Code2,
  Expand,
  GitBranch,
  Network,
  Search,
  ShieldCheck,
  Wrench,
  Zap,
} from "lucide-react";
import { useRoom } from "../store";
import { actor, nice, tones } from "../selectors";
import { useClock } from "../hooks/useRuntime";
import type { Agent, Pulse, RuntimeEvent } from "../types";
import { Badge } from "./ui";

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
  },
  "person"
>;
const PersonCard = memo(({ data, selected }: NodeProps<Person>) => {
  const select = useRoom((room) => room.select);
  const Icon = /review/.test(data.agent.role.toLowerCase())
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
      <footer>
        <span>
          {data.completed}/{data.tasks} tasks
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
function FlowEdge(props: EdgeProps) {
  const self = props.source === props.target;
  const [normal, labelX, labelY] = getBezierPath(props);
  const path = self
    ? `M ${props.sourceX} ${props.sourceY} C ${props.sourceX + 90} ${props.sourceY - 70}, ${props.targetX + 90} ${props.targetY + 80}, ${props.targetX} ${props.targetY}`
    : normal;
  const data = props.data as {
    pulses?: Pulse[];
    count?: number;
    tone: string;
    label?: string;
  };
  return (
    <>
      <BaseEdge
        {...props}
        path={path}
        style={{
          stroke: data.tone,
          opacity: data.pulses?.length ? 0.9 : 0.35,
          strokeWidth: data.pulses?.length ? 2 : 1.2,
        }}
      />
      {data.pulses?.map((p) => (
        <circle
          key={p.event.id}
          r="4"
          fill={data.tone}
          className="flow-packet"
          data-flow-event={p.event.id}
        >
          <animateMotion dur="1.1s" repeatCount="indefinite" path={path} />
        </circle>
      ))}
      {!self && !!data.count && (
        <g className="edge-label" transform={`translate(${labelX},${labelY})`}>
          <rect x="-13" y="-9" width="26" height="18" rx="8" />
          <text textAnchor="middle" dy="4">
            {data.count}
          </text>
        </g>
      )}
      {self && data.pulses?.length ? (
        <text
          x={props.sourceX + 30}
          y={props.sourceY - 26}
          className="tool-edge-label"
        >
          {data.label}
        </text>
      ) : null}
    </>
  );
}
const edgeTypes = { flow: FlowEdge };

function GraphContent() {
  const { snapshot, defaults, events, pulses, selection, select } = useRoom();
  const config = snapshot?.manifest.settings || defaults;
  const team = config?.team || [],
    tasks = Object.values(snapshot?.state.tasks || {}),
    now = useClock();
  const key = team.map((a) => a.name).join("|");
  const { fitView } = useReactFlow();
  const container = useRef<HTMLDivElement>(null);
  const [positions, setPositions] = useState<
    Record<string, { x: number; y: number }>
  >({});
  const layout = useMemo(() => {
    const graph = new dagre.graphlib.Graph()
      .setGraph({ rankdir: "TB", ranksep: 38, nodesep: 95 })
      .setDefaultEdgeLabel(() => ({}));
    team.forEach((a) => graph.setNode(a.name, { width: 214, height: 132 }));
    const workers = team.filter(
      (a) => a.name !== config?.coordinator && a.name !== config?.reviewer,
    );
    workers.forEach((a) => {
      graph.setEdge(config!.coordinator, a.name);
      graph.setEdge(a.name, config!.reviewer);
    });
    dagre.layout(graph);
    return Object.fromEntries(
      team.map((a) => [
        a.name,
        { x: graph.node(a.name).x - 107, y: graph.node(a.name).y - 66 },
      ]),
    );
  }, [key, config?.coordinator, config?.reviewer]);
  useEffect(() => {
    setPositions({});
  }, [key]);
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    const fit = () => {
      clearTimeout(timer);
      timer = setTimeout(
        () => void fitView({ padding: 0.12, duration: 200 }),
        120,
      );
    };
    const observer = new ResizeObserver(fit);
    observer.observe(container.current!);
    fit();
    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, [key, fitView]);
  const activePulses = pulses.filter((p) => p.start <= now && p.end >= now);
  const nodes: Person[] = team.map((agent, index) => {
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
    const mode = active
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
    const activity = latest
      ? latest.tool || latest.purpose || nice(latest.node) || nice(latest.kind)
      : snapshot?.state.status === "completed"
        ? "Mission complete"
        : "Ready to collaborate";
    return {
      id: agent.name,
      type: "person",
      position: positions[agent.name] || layout[agent.name] || { x: 0, y: 0 },
      selected: selection?.type === "agent" && selection.id === agent.name,
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
    const id = `${source}→${target}`;
    const link = links.get(id) || { source, target, count: 0 };
    link.count += count;
    links.set(id, link);
  };
  team
    .filter(
      (a) => a.name !== config?.coordinator && a.name !== config?.reviewer,
    )
    .forEach((a) => {
      add(config!.coordinator, a.name);
      add(a.name, config!.reviewer);
    });
  events
    .filter((e) => e.kind === "message")
    .forEach((e) => {
      if (e.recipient === "all") team.forEach((a) => add(e.sender!, a.name, 1));
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
      data: { count: link.count, pulses: live, tone },
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
          const moved = changes.filter(
            (c) => c.type === "position" && c.position,
          );
          if (moved.length)
            setPositions((previous) => {
              const next = { ...previous };
              for (const change of moved)
                if (change.type === "position" && change.position)
                  next[change.id] = change.position;
              return next;
            });
        }}
        onEdgeClick={(_, edge) => select({ type: "channel", id: edge.id })}
        nodesConnectable={false}
        fitView
        minZoom={0.3}
        maxZoom={1.8}
        colorMode="dark"
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#27314c" gap={22} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}

export function AgentGraph() {
  const { snapshot, defaults, events } = useRoom();
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
              {(snapshot?.manifest.settings || defaults)?.team.length || 0}
            </span>
          </h2>
        </div>
        <div className="graph-phase">
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
        <ReactFlowProvider>
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
