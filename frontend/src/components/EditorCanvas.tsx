import { memo, useRef } from "react";
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
import { Code2, Network, ShieldCheck } from "lucide-react";
import { tones } from "../selectors";
import type { Agent, TeamConfiguration } from "../types";
import { addCommunicationLink, removeAgent } from "../graph/editorModel";
import { GraphViewport } from "../graph/GraphViewport";
import { linkId } from "../graph/model";
import { useNodeMeasurements } from "../graph/useNodeMeasurements";

type DraftNode = Node<
  { agent: Agent; tone: string; designation: string },
  "draftAgent"
>;
const DraftAgent = memo(({ data, selected }: NodeProps<DraftNode>) => {
  const Icon =
    data.designation === "Coordinator"
      ? Network
      : data.designation === "Reviewer"
        ? ShieldCheck
        : Code2;
  return (
    <div
      className={`draft-agent ${selected ? "selected" : ""}`}
      data-draft-agent={data.agent.name}
      style={{ "--tone": data.tone } as React.CSSProperties}
    >
      <Handle
        type="target"
        position={Position.Top}
        id="in"
        aria-label={`Receive messages at ${data.agent.name}`}
      />
      <div className="draft-agent-heading">
        <span>
          <Icon size={18} />
        </span>
        <div>
          <strong>{data.agent.name || "Unnamed agent"}</strong>
          <small>{data.agent.role || "Set a role"}</small>
        </div>
      </div>
      <div className="draft-agent-details">
        <span>{data.designation || "Worker"}</span>
        <span>{data.agent.tools.length} tools</span>
      </div>
      <p>{data.agent.persona}</p>
      <Handle
        type="source"
        position={Position.Bottom}
        id="out"
        aria-label={`Send messages from ${data.agent.name}`}
      />
    </div>
  );
});
const nodeTypes = { draftAgent: DraftAgent };

function Canvas({
  draft,
  change,
  selected,
  select,
  viewportRevision,
}: {
  draft: TeamConfiguration;
  change: (draft: TeamConfiguration) => void;
  selected: string;
  select: (name: string) => void;
  viewportRevision: number;
}) {
  const container = useRef<HTMLDivElement>(null);
  const { measurements, acceptMeasurements } = useNodeMeasurements();
  const nodes: DraftNode[] = draft.settings.team.map((agent, index) => ({
    id: agent.name,
    type: "draftAgent",
    selected: agent.name === selected,
    measured: measurements[agent.name],
    position: draft.positions[agent.name] || { x: 0, y: 0 },
    data: {
      agent,
      tone: tones[index % tones.length],
      designation:
        agent.name === draft.settings.coordinator
          ? "Coordinator"
          : agent.name === draft.settings.reviewer
            ? "Reviewer"
            : "",
    },
  }));
  const edges = (draft.settings.communication_links || []).map((link) => ({
    id: linkId(link),
    ...link,
    sourceHandle: "out",
    targetHandle: "in",
    type: "smoothstep",
    style: { stroke: "#7085c0", strokeWidth: 1.5 },
    markerEnd: { type: MarkerType.ArrowClosed, color: "#7085c0" },
  }));
  return (
    <div className="editor-canvas" ref={container}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        colorMode="dark"
        minZoom={0.2}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, node) => select(node.id)}
        onPaneClick={() => select("")}
        onNodesChange={(changes) => {
          acceptMeasurements(changes);
          let next = draft;
          for (const entry of changes) {
            if (entry.type === "position" && entry.position)
              next = {
                ...next,
                positions: { ...next.positions, [entry.id]: entry.position },
              };
            if (entry.type === "remove") next = removeAgent(next, entry.id);
          }
          if (next !== draft) change(next);
        }}
        onEdgesChange={(changes) => {
          const removed = changes
            .filter((entry) => entry.type === "remove")
            .map((entry) => entry.id);
          if (removed.length)
            change({
              ...draft,
              settings: {
                ...draft.settings,
                communication_links: (
                  draft.settings.communication_links || []
                ).filter((link) => !removed.includes(linkId(link))),
              },
            });
        }}
        onConnect={(connection) =>
          change(
            addCommunicationLink(draft, {
              source: connection.source,
              target: connection.target,
            }),
          )
        }
        isValidConnection={(connection) =>
          connection.source !== connection.target &&
          !edges.some(
            (edge) =>
              edge.source === connection.source &&
              edge.target === connection.target,
          )
        }
        deleteKeyCode={null}
      >
        <Background gap={22} size={1} color="#293551" />
        <Controls showInteractive={false} />
        <GraphViewport
          container={container}
          identity={`${viewportRevision}:${draft.settings.team.map((agent) => agent.name).join("|")}`}
          count={nodes.length}
        />
      </ReactFlow>
      <div className="editor-canvas-hint">
        Drag agents to arrange · Draw from the bottom handle to the top handle
      </div>
    </div>
  );
}

export function EditorCanvas(props: Parameters<typeof Canvas>[0]) {
  return (
    <ReactFlowProvider>
      <Canvas {...props} />
    </ReactFlowProvider>
  );
}
