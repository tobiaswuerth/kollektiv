import { create } from "zustand";
import { api } from "./api";
import type {
  EventBatch,
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
  health: { connected: boolean; model: string; models: string[] } | null;
  accept: (data: Snapshot | EventBatch, animate?: boolean) => void;
  select: (selection: Selection) => void;
  chooseRun: (id: string) => Promise<void>;
  refreshRuns: () => Promise<void>;
  control: (action: string, retry?: boolean) => Promise<void>;
  intervene: (action: string, fields: Record<string, unknown>) => Promise<void>;
  createRun: (goal: string, settings: Settings) => Promise<void>;
};
export const useRoom = create<Room>((set, get) => ({
  defaults: null,
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
  health: null,
  select: (selection) => set({ selection }),
  accept: (data, animate = true) => {
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
    const snapshot =
      "manifest" in data
        ? data
        : room.snapshot
          ? {
              ...room.snapshot,
              running: data.running,
              control: data.control,
              error: data.error || room.snapshot.error,
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
    set({
      runId: id || null,
      snapshot: null,
      events: [],
      pulses: [],
      selection: null,
      error: "",
    });
    if (!id) return;
    try {
      const data = await api<Snapshot>(`/api/runs/${id}`);
      if (get().runId === id) get().accept(data, false);
    } catch (e) {
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
    set({ busy: true, error: "" });
    try {
      const data = await api<Snapshot>(`/api/runs/${id}/control`, {
        action,
        retry,
      });
      if (get().runId === id) get().accept(data);
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ busy: false });
    }
  },
  intervene: async (action, fields) => {
    const id = get().runId;
    if (!id) return;
    const data = await api<Snapshot>(`/api/runs/${id}/admin`, {
      action,
      ...fields,
    });
    if (get().runId === id) {
      get().accept(data);
      await get().refreshRuns();
    }
  },
  createRun: async (goal, settings) => {
    const data = await api<Snapshot>("/api/runs", { goal, settings });
    await get().refreshRuns();
    await get().chooseRun(data.manifest.run_id);
  },
}));
