import { useRoom } from "./store";
import type { Agent, RuntimeEvent, Snapshot, Specialist, Task } from "./types";
export const tones = [
  "#8b8fff",
  "#43c5e5",
  "#f3ae62",
  "#bd89f6",
  "#ee79a6",
  "#79a4f5",
  "#cfc26d",
];
export const time = (at?: string) =>
  at ? new Date(at).toLocaleTimeString([], { hour12: false }) : "";
export const nice = (text?: string) => (text || "").replaceAll("_", " ");
export function specialists(snapshot: Snapshot | null): Specialist[] {
  return Object.values(snapshot?.delegation?.agents || {});
}
export function activeTeam(
  snapshot: Snapshot | null,
  fallback: Agent[] = [],
): Agent[] {
  return [
    ...(snapshot?.manifest.settings.team || fallback),
    ...specialists(snapshot).filter((agent) => agent.status !== "retired"),
  ];
}
export function agentHistory(
  agent: Specialist,
  events: RuntimeEvent[],
): RuntimeEvent[] {
  const history = new Map<string, RuntimeEvent>();
  (agent.history || []).forEach((event) => history.set(event.id, event));
  events
    .filter(
      (event) =>
        event.agent === agent.name ||
        event.sender === agent.name ||
        event.recipient === agent.name ||
        event.child === agent.id,
    )
    .forEach((event) => history.set(event.id, event));
  return [...history.values()].sort((a, b) =>
    (a.at || "").localeCompare(b.at || ""),
  );
}
export function canGuideSpecialist(agent: Specialist) {
  return !["completed", "failed", "retired"].includes(agent.status);
}
export function specialistMode(agent: Specialist) {
  return {
    pending: "pending",
    running: "working",
    awaiting_children: "pending",
    reviewing: "reviewing",
    completed: "completed",
    failed: "failed",
    interrupted: "interrupted",
    retired: "retired",
  }[agent.status];
}
export function canAdmin(snapshot: Snapshot | null) {
  return (
    !!snapshot &&
    (!snapshot.running ||
      !!(
        snapshot.control?.paused &&
        snapshot.control.waiting &&
        !snapshot.control.active_nodes.length
      ))
  );
}
export function actor(event: RuntimeEvent) {
  const { snapshot, defaults } = useRoom.getState();
  const config = snapshot?.manifest.settings || defaults;
  if (
    event.purpose === "TaskReview" ||
    event.purpose === "GoalReview" ||
    event.node?.endsWith("review")
  )
    return config?.reviewer;
  return (
    event.agent ||
    event.sender ||
    snapshot?.state.tasks?.[event.task || ""]?.owner ||
    ([
      "begin",
      "plan",
      "schedule",
      "collect",
      "revise",
      "discussion_join",
    ].includes(event.node || "") || event.kind === "plan_created"
      ? config?.coordinator
      : undefined)
  );
}
export function taskStatus(task: Task, snapshot: Snapshot | null) {
  if (task.status !== "pending") return task.status;
  if (
    snapshot?.running &&
    snapshot.control?.active_nodes.some((n) => n.task === task.id)
  )
    return "working";
  return task.dependencies.some(
    (id) => snapshot?.state.tasks?.[id]?.status !== "completed",
  )
    ? "blocked"
    : "pending";
}
export function records(key: "messages" | "notes", kind: "message" | "note") {
  const { snapshot, events } = useRoom.getState();
  const values = { ...snapshot?.state[key] };
  events
    .filter((e) => e.kind === kind)
    .forEach((e) => {
      values[e.id] = e;
    });
  return Object.values(values).sort((a, b) =>
    (a.at || "").localeCompare(b.at || ""),
  );
}
