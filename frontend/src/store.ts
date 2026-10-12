import { create } from "zustand";
import { api } from "./api";
import type {
  EventBatch,
  MissionSample,
  Pulse,
  Run,
  RuntimeEvent,
  Selection,
  Settings,
  Snapshot,
} from "./types";

export function mergeEvents(
  existing: RuntimeEvent[],
  incoming: RuntimeEvent[],
) {
  const seen = new Set(existing.map((e) => e.id));
  const fresh = incoming.filter((e) => {
    if (seen.has(e.id)) return false;
    seen.add(e.id);
    return true;
  });
  return { events: [...existing, ...fresh].slice(-5000), fresh };
}
type Room = {
  defaults: Settings | null;
  missionSamples: MissionSample[];
  runs: Run[];
  runId: string | null;
  snapshot: Snapshot | null;
  events: RuntimeEvent[];
  pulses: Pulse[];
  selection: Selection;
  error: string;
  busy: boolean;
  connected: boolean;
  latency: number;
  mutationEpoch: number;
  health: { connected: boolean; model: string; models: string[] } | null;
  accept: (
    data: Snapshot | EventBatch,
    animate?: boolean,
    authoritative?: boolean,
  ) => void;
  select: (selection: Selection) => void;
  chooseRun: (id: string) => Promise<void>;
  refreshRuns: () => Promise<void>;
  control: (action: string, retry?: boolean) => Promise<void>;
  intervene: (action: string, fields: Record<string, unknown>) => Promise<void>;
  createRun: (
    goal: string,
    settings: Settings,
    positions?: Record<string, { x: number; y: number }>,
  ) => Promise<void>;
};
export const useRoom = create<Room>((set, get) => ({
  defaults: null,
  missionSamples: [],
  runs: [],
  runId: null,
  snapshot: null,
  events: [],
  pulses: [],
  selection: null,
  error: "",
  busy: false,
  connected: true,
  latency: 0,
  mutationEpoch: 0,
  health: null,
  select: (selection) => set({ selection }),
  accept: (data, animate = true, authoritative = false) => {
    const room = get();
    const { events, fresh } = mergeEvents(room.events, data.events);
    const now = Date.now();
    const pulses = animate
      ? fresh
          .filter((e) => now - Date.parse(e.at) < 10_000)
          .slice(-60)
          .map((event, i) => ({
            event,
            start: now + Math.min(i * 40, 1600),
            end: now + Math.min(i * 40, 1600) + 1800,
          }))
      : [];
    const stale =
      !authoritative && !!room.snapshot && data.cursor < room.snapshot.cursor;
    const snapshot = stale
      ? room.snapshot
      : "manifest" in data
        ? data
        : room.snapshot
          ? {
              ...room.snapshot,
              running: data.running,
              control: data.control,
              error: data.error || room.snapshot.error,
              delegation: data.delegation ?? room.snapshot.delegation,
            }
          : null;
    set({
      snapshot: snapshot
        ? {
            ...snapshot,
            cursor: Math.max(
              snapshot.cursor,
              data.cursor,
              room.snapshot?.cursor || 0,
            ),
          }
        : null,
      events,
      pulses: [...room.pulses.filter((p) => p.end > now), ...pulses].slice(
        -160,
      ),
      connected: true,
    });
  },
  chooseRun: async (id) => {
    const epoch = get().mutationEpoch + 1;
    set({
      runId: id || null,
      snapshot: null,
      events: [],
      pulses: [],
      selection: null,
      error: "",
      busy: false,
      mutationEpoch: epoch,
    });
    if (!id) return;
    try {
      const data = await api<Snapshot>(`/api/runs/${id}`);
      if (get().runId === id && get().mutationEpoch === epoch)
        get().accept(data, false, true);
    } catch (e) {
      if (get().runId === id && get().mutationEpoch === epoch)
        set({ error: String(e), connected: false });
    }
  },
  refreshRuns: async () => {
    const data = await api<{ runs: Run[] }>("/api/runs");
    set({ runs: data.runs });
  },
  control: async (action, retry = false) => {
    const id = get().runId;
    if (!id || get().busy) return;
    const epoch = get().mutationEpoch + 1;
    set({ busy: true, error: "", mutationEpoch: epoch });
    try {
      const data = await api<Snapshot>(`/api/runs/${id}/control`, {
        action,
        retry,
      });
      if (get().runId === id && get().mutationEpoch === epoch)
        get().accept(data, true, true);
    } catch (e) {
      if (get().runId === id && get().mutationEpoch === epoch)
        set({ error: String(e) });
    } finally {
      if (get().mutationEpoch === epoch)
        set({ busy: false, mutationEpoch: epoch + 1 });
    }
  },
  intervene: async (action, fields) => {
    const id = get().runId;
    if (!id) return;
    if (get().busy)
      throw new Error("Wait for the current control action to finish.");
    const epoch = get().mutationEpoch + 1;
    set({ busy: true, mutationEpoch: epoch });
    try {
      const data = await api<Snapshot>(`/api/runs/${id}/admin`, {
        action,
        ...fields,
      });
      if (get().runId === id && get().mutationEpoch === epoch) {
        get().accept(data, true, true);
        await get().refreshRuns();
      }
    } finally {
      if (get().mutationEpoch === epoch)
        set({ busy: false, mutationEpoch: epoch + 1 });
    }
  },
  createRun: async (goal, settings, positions) => {
    const data = await api<Snapshot>("/api/runs", {
      goal,
      settings,
      positions,
    });
    await get().refreshRuns();
    await get().chooseRun(data.manifest.run_id);
  },
}));
