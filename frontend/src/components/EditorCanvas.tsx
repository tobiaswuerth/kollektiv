import { memo, useEffect, useRef, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Panel,
  Position,
  ReactFlow,
  ReactFlowProvider,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import {
  Code2,
  Focus,
  Network,
  Route,
  ShieldCheck,
  Trash2,
} from "lucide-react";
import { agentDesignation, tones } from "../selectors";
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
  const Icon = data.designation.startsWith("Coordinator")
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
      <p title={data.agent.skills.join(", ")}>
        {data.agent.skills.slice(0, 2).join(" · ") ||
          "Select to edit instructions"}
      </p>
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

function SelectionControls({
  selected,
  selectedEdge,
  removePath,
  focusedPaths,
  setFocusedPaths,
  focusSelected,
}: {
  selected: string;
  selectedEdge: string;
  removePath: () => void;
  focusedPaths: boolean;
  setFocusedPaths: (value: boolean) => void;
  focusSelected: () => void;
}) {
  return (
    <Panel position="top-right" className="editor-selection-controls">
      <button
        type="button"
        className="button"
        disabled={!selected}
        onClick={focusSelected}
        aria-label="Focus selected agent"
      >
        <Focus size={14} />
        Focus agent
      </button>
      <button
        type="button"
        className="button"
        disabled={!selected}
        aria-pressed={focusedPaths}
        onClick={() => setFocusedPaths(!focusedPaths)}
        aria-label="Highlight selected agent paths"
      >
        <Route size={14} />
        {focusedPaths ? "All paths" : "Agent paths"}
      </button>
      {selectedEdge && (
        <button
          type="button"
          className="button danger"
          onClick={removePath}
          aria-label="Remove selected path"
        >
          <Trash2 size={14} />
          Remove path
        </button>
      )}
    </Panel>
  );
}

function Canvas({
  draft,
  change,
  selected,
  select,
  viewportRevision,
  focusRequest,
  onPathCreated,
}: {
  draft: TeamConfiguration;
  change: (draft: TeamConfiguration) => void;
  selected: string;
  select: (name: string, focus?: boolean) => void;
  viewportRevision: number;
  focusRequest: { id: string; revision: number };
  onPathCreated: () => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [selectedEdge, setSelectedEdge] = useState("");
  const [focusedPaths, setFocusedPaths] = useState(false);
  useEffect(() => {
    setSelectedEdge("");
    setFocusedPaths(false);
  }, [viewportRevision]);
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
      designation: agentDesignation(draft.settings, agent.name),
    },
  }));
  const edges = (draft.settings.communication_links || []).map((link) => ({
    id: linkId(link),
    ...link,
    sourceHandle: "out",
    targetHandle: "in",
    type: "smoothstep",
    selected: selectedEdge === linkId(link),
    style: {
      stroke: selectedEdge === linkId(link) ? "#b0adff" : "#7085c0",
      strokeWidth: selectedEdge === linkId(link) ? 2.5 : 1.5,
      opacity:
        focusedPaths &&
        selected &&
        link.source !== selected &&
        link.target !== selected
          ? 0.1
          : 1,
    },
    markerEnd: { type: MarkerType.ArrowClosed, color: "#7085c0" },
  }));
  return (
    <div className="editor-canvas" ref={container}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        colorMode="dark"
        minZoom={0.18}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, node) => {
          select(node.id);
          setSelectedEdge("");
        }}
        onEdgeClick={(_, edge) => setSelectedEdge(edge.id)}
        onPaneClick={() => {
          select("");
          setSelectedEdge("");
        }}
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
        onConnect={(connection) => {
          change(
            addCommunicationLink(draft, {
              source: connection.source,
              target: connection.target,
            }),
          );
          onPathCreated();
        }}
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
        <MiniMap
          pannable
          zoomable
          position="bottom-right"
          nodeColor="#69799f"
          maskColor="#080f208f"
          aria-label="Graph overview map"
        />
        <GraphViewport
          container={container}
          identity={`${viewportRevision}:${draft.settings.team.map((agent) => agent.name).join("|")}`}
          count={nodes.length}
          focusRequest={focusRequest}
        />
        <SelectionControls
          selected={selected}
          selectedEdge={
            edges.some((edge) => edge.id === selectedEdge) ? selectedEdge : ""
          }
          removePath={() => {
            change({
              ...draft,
              settings: {
                ...draft.settings,
                communication_links: (
                  draft.settings.communication_links || []
                ).filter((link) => linkId(link) !== selectedEdge),
              },
            });
            setSelectedEdge("");
          }}
          focusedPaths={focusedPaths}
          setFocusedPaths={setFocusedPaths}
          focusSelected={() => select(selected, true)}
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
