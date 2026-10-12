import { useRoom } from "./store";
import type { RuntimeEvent, Snapshot, Task } from "./types";
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
