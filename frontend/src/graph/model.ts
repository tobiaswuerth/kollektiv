import dagre from "@dagrejs/dagre";
import type {
  CommunicationLink,
  GraphPosition,
  Settings,
  Specialist,
} from "../types";

export function linkId(link: CommunicationLink) {
  return `${link.source}→${link.target}`;
}

export function workflowLinks(settings: Settings): CommunicationLink[] {
  return settings.team
    .filter(
      (agent) =>
        ![settings.coordinator, settings.reviewer].includes(agent.name),
    )
    .flatMap((agent) => [
      { source: settings.coordinator, target: agent.name },
      { source: agent.name, target: settings.reviewer },
    ]);
}

export function layoutTeam(settings: Settings): Record<string, GraphPosition> {
  const graph = new dagre.graphlib.Graph()
    .setGraph({ rankdir: "TB", ranksep: 45, nodesep: 75 })
    .setDefaultEdgeLabel(() => ({}));
  const names = new Set(settings.team.map((agent) => agent.name));
  settings.team.forEach((agent) =>
    graph.setNode(agent.name, { width: 214, height: 146 }),
  );
  const links = settings.communication_links ?? workflowLinks(settings);
  links.forEach(({ source, target }) => {
    if (names.has(source) && names.has(target) && source !== target)
      graph.setEdge(source, target);
  });
  dagre.layout(graph);
  return Object.fromEntries(
    settings.team.map((agent) => {
      const point = graph.node(agent.name);
      return [agent.name, { x: point.x - 107, y: point.y - 73 }];
    }),
  );
}

export function layoutRuntimeTeam(
  settings: Settings,
  specialists: Specialist[],
  anchors: Record<string, GraphPosition> = {},
) {
  const active = specialists.filter((agent) => agent.status !== "retired");
  const names = new Set(
    [...settings.team, ...active].map((agent) => agent.name),
  );
  const positions = {
    ...layoutTeam(settings),
    ...Object.fromEntries(
      Object.entries(anchors).filter(([name]) => names.has(name)),
    ),
  };
  // A hire must not move its parent or collide with saved configuration positions.
  // Place new family members beneath their actual parent, then find a free column.
  const collides = (point: GraphPosition) =>
    Object.values(positions).some(
      (other) =>
        Math.abs(point.x - other.x) < 244 && Math.abs(point.y - other.y) < 186,
    );
  active
    .slice()
    .sort((a, b) => a.level - b.level)
    .forEach((agent) => {
      if (positions[agent.name]) return;
      const parent = positions[agent.parent] || { x: 0, y: 0 };
      const point = { x: parent.x, y: parent.y + 211 };
      while (collides(point)) point.x += 289;
      positions[agent.name] = point;
    });
  return positions;
}

export function pulseProgress(start: number, end: number, now: number) {
  return Math.max(0, Math.min(1, (now - start) / Math.max(1, end - start)));
}

export function canFitGraph(
  width: number,
  height: number,
  initialized: boolean,
  count: number,
) {
  return (
    initialized &&
    count > 0 &&
    Number.isFinite(width) &&
    Number.isFinite(height) &&
    width > 1 &&
    height > 1
  );
}
