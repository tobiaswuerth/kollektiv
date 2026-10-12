import { beforeEach, describe, expect, it } from "vitest";
import {
  activeTeam,
  agentHistory,
  canGuideSpecialist,
  specialistMode,
  specialists,
} from "./selectors";
import { layoutRuntimeTeam } from "./graph/model";
import { useRoom } from "./store";
import type { Agent, Snapshot, Specialist } from "./types";

const agent = (name: string): Agent => ({
  name,
  role: "Author",
  persona: "Deliver usable text",
  skills: ["writing"],
  tools: ["read_file", "write_file", "hire", "monitor_agents"],
});
const child = (name: string, parent: string, level: 2 | 3 = 2): Specialist => ({
  ...agent(name),
  id: `${name}-id`,
  parent,
  parent_id: level === 3 ? `${parent}-id` : null,
  level,
  status: "running",
  assignment: "Write a small specialist contribution",
  output: `delegates/${name}/contribution.md`,
  input_files: [],
  acceptance_criteria: ["Complete text"],
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
});
const makeSnapshot = (jobs: Specialist[]): Snapshot => ({
  manifest: {
    run_id: "current",
    goal: "Build a coherent document",
    status: "working",
    settings: {
      model: "local",
      base_url: "http://localhost:11434",
      team: [agent("lead"), agent("worker"), agent("review")],
      coordinator: "lead",
      reviewer: "review",
      max_concurrency: 1,
      web_enabled: false,
    },
  },
  state: { status: "working", tasks: {} },
  running: true,
  control: null,
  error: null,
  events: [],
  cursor: 0,
  artifacts: [],
  delegation: {
    agents: Object.fromEntries(jobs.map((job) => [job.name, job])),
    max_depth: 3,
  },
});

beforeEach(() => useRoom.setState(useRoom.getInitialState(), true));

describe("runtime specialists", () => {
  it("renders children and grandchildren without changing the configured team; retains retired history", () => {
    const snapshot = makeSnapshot([
      child("research", "worker"),
      child("sources", "research", 3),
      { ...child("retired", "worker"), status: "retired" },
    ]);
    expect(activeTeam(snapshot).map((agent) => agent.name)).toEqual([
      "lead",
      "worker",
      "review",
      "research",
      "sources",
    ]);
    expect(snapshot.manifest.settings.team.map((agent) => agent.name)).toEqual([
      "lead",
      "worker",
      "review",
    ]);
    expect(specialists(snapshot)).toHaveLength(3);
  });
  it("updates the graph registry from a fast event response and ignores stale retirement", () => {
    const snapshot = makeSnapshot([child("research", "worker")]);
    useRoom.getState().accept(snapshot);
    useRoom.getState().accept({
      events: [],
      cursor: 5,
      running: true,
      control: null,
      error: null,
      delegation: {
        agents: {
          research: { ...child("research", "worker"), status: "retired" },
        },
        max_depth: 3,
      },
    });
    expect(activeTeam(useRoom.getState().snapshot)).toHaveLength(3);
    useRoom.getState().accept({ ...snapshot, cursor: 4 });
    expect(activeTeam(useRoom.getState().snapshot)).toHaveLength(3);
    expect(specialists(useRoom.getState().snapshot)[0].status).toBe("retired");
  });
  it("combines durable specialist history with live messages without duplicate records", () => {
    const event = {
      id: "first",
      at: "2026-01-01T00:00:01Z",
      kind: "agent_status",
      agent: "research",
    };
    const job = { ...child("research", "worker"), history: [event] };
    expect(
      agentHistory(job, [
        event,
        {
          id: "instruction",
          at: "2026-01-01T00:00:02Z",
          kind: "message",
          sender: "admin",
          recipient: "research",
          content: "Check your assumptions.",
        },
        {
          id: "other",
          at: "2026-01-01T00:00:03Z",
          kind: "model",
          agent: "unrelated",
        },
      ]).map((event) => event.id),
    ).toEqual(["first", "instruction"]);
  });
  it("lays out a finite three-level delegation hierarchy and omits retired agents", () => {
    const snapshot = makeSnapshot([
      child("research", "worker"),
      child("sources", "research", 3),
      { ...child("old", "worker"), status: "retired" },
    ]);
    const layout = layoutRuntimeTeam(
      snapshot.manifest.settings,
      specialists(snapshot),
    );
    expect(Object.keys(layout)).toHaveLength(5);
    expect(layout.old).toBeUndefined();
    expect(layout.research.y).toBeGreaterThan(layout.worker.y);
    expect(layout.sources.y).toBeGreaterThan(layout.research.y);
    Object.values(layout).forEach((position) => {
      expect(Number.isFinite(position.x)).toBe(true);
      expect(Number.isFinite(position.y)).toBe(true);
    });
  });
  it("uses the actual specialist status independently from model event pulses", () => {
    expect(
      specialistMode({ ...child("research", "worker"), status: "reviewing" }),
    ).toBe("reviewing");
    expect(
      specialistMode({
        ...child("research", "worker"),
        status: "awaiting_children",
      }),
    ).toBe("pending");
    expect(
      specialistMode({ ...child("research", "worker"), status: "retired" }),
    ).toBe("retired");
  });
  it("keeps saved parent positions and places a new family outside occupied nodes", () => {
    const snapshot = makeSnapshot([
      child("research", "worker"),
      child("sources", "research", 3),
    ]);
    const anchors = {
      lead: { x: 0, y: 0 },
      worker: { x: 400, y: 200 },
      review: { x: 400, y: 411 },
    };
    const layout = layoutRuntimeTeam(
      snapshot.manifest.settings,
      specialists(snapshot),
      anchors,
    );
    expect(layout.worker).toEqual(anchors.worker);
    expect(layout.review).toEqual(anchors.review);
    expect(layout.research.x).toBeGreaterThan(layout.review.x + 214);
    expect(layout.research.y).toBeGreaterThan(layout.worker.y);
    expect(layout.sources.y).toBeGreaterThan(layout.research.y);
  });
  it("preserves read-only specialists and reports without inventing file permissions", () => {
    const report = {
      ...child("analysis", "worker"),
      tools: [],
      output: null,
      summary: "The assumptions need a counterexample.",
      status: "completed" as const,
    };
    const snapshot = makeSnapshot([report]);
    expect(activeTeam(snapshot).at(-1)?.tools).toEqual([]);
    expect(specialists(snapshot)[0].summary).toContain("counterexample");
    expect(specialists(snapshot)[0].output).toBeNull();
    expect(canGuideSpecialist(report)).toBe(false);
    expect(canGuideSpecialist(child("analysis", "worker"))).toBe(true);
  });
  it("keeps durable independent-review events when their actor is the main reviewer", () => {
    const review = {
      id: "independent-review",
      at: "2026-01-01T00:00:01Z",
      kind: "model",
      agent: "review",
    };
    expect(
      agentHistory({ ...child("research", "worker"), history: [review] }, []),
    ).toEqual([review]);
  });
});
