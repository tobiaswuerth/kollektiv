import { describe, expect, it } from "vitest";
import { mergeEvents } from "./store";
import type { RuntimeEvent } from "./types";
describe("incremental runtime events", () => {
  it("preserves rapid start/finish transitions and deduplicates overlapping state and event polls", () => {
    const events: RuntimeEvent[] = [
      {
        id: "start",
        at: "2026-01-01T00:00:00Z",
        kind: "node_started",
        agent: "builder",
      },
      {
        id: "tool",
        at: "2026-01-01T00:00:00Z",
        kind: "tool_result",
        tool: "write_file",
      },
      {
        id: "end",
        at: "2026-01-01T00:00:00Z",
        kind: "node_finished",
        agent: "builder",
      },
    ];
    const first = mergeEvents([], events);
    const second = mergeEvents(first.events, [events[0], events[1], events[2]]);
    expect(first.fresh.map((e) => e.kind)).toEqual([
      "node_started",
      "tool_result",
      "node_finished",
    ]);
    expect(second.events).toEqual(events);
    expect(second.fresh).toEqual([]);
  });
});
