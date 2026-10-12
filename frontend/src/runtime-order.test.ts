import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useRoom } from "./store";
import type { Snapshot } from "./types";

const snapshot = (cursor: number, running = false): Snapshot => ({
  manifest: {
    run_id: "current",
    goal: "Write a story",
    status: "created",
    settings: {
      model: "test",
      base_url: "http://localhost:11434",
      team: [],
      coordinator: "c",
      reviewer: "r",
      max_concurrency: 1,
      web_enabled: false,
    },
  },
  state: { status: running ? "working" : "paused", tasks: {} },
  control: null,
  error: null,
  running,
  cursor,
  events: [],
  artifacts: [],
});
beforeEach(() => useRoom.setState(useRoom.getInitialState(), true));
afterEach(() => vi.unstubAllGlobals());

describe("overlapping runtime responses", () => {
  it("keeps newer control state when an older polling snapshot arrives", () => {
    useRoom.getState().accept(snapshot(200));
    useRoom.getState().accept({
      ...snapshot(100, true),
      events: [
        { id: "earlier", at: "2026-01-01T00:00:00Z", kind: "node_started" },
      ],
    });
    expect(useRoom.getState().snapshot?.running).toBe(false);
    expect(useRoom.getState().snapshot?.state.status).toBe("paused");
    expect(useRoom.getState().snapshot?.cursor).toBe(200);
    expect(useRoom.getState().events).toHaveLength(1);
  });

  it("accepts an authoritative control response with a shorter historical event batch", async () => {
    useRoom.setState({ runId: "current", snapshot: snapshot(1000, true) });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify(snapshot(500)))),
    );
    const epoch = useRoom.getState().mutationEpoch;
    await useRoom.getState().control("pause");
    expect(useRoom.getState().snapshot?.running).toBe(false);
    expect(useRoom.getState().snapshot?.cursor).toBe(1000);
    expect(useRoom.getState().mutationEpoch).toBeGreaterThan(epoch);
    expect(useRoom.getState().busy).toBe(false);
  });

  it("ignores errors from an old run after switching to a healthy run", async () => {
    let rejectOld!: (error: Error) => void;
    const old = new Promise<Response>((_, reject) => {
      rejectOld = reject;
    });
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockReturnValueOnce(old)
        .mockResolvedValueOnce(
          new Response(
            JSON.stringify({
              ...snapshot(0),
              manifest: { ...snapshot(0).manifest, run_id: "new" },
            }),
          ),
        ),
    );
    const pending = useRoom.getState().chooseRun("old");
    await useRoom.getState().chooseRun("new");
    rejectOld(new Error("The old connection failed"));
    await pending;
    expect(useRoom.getState().runId).toBe("new");
    expect(useRoom.getState().error).toBe("");
    expect(useRoom.getState().connected).toBe(true);
  });
});
