import { describe, expect, it } from "vitest";
import type { Settings } from "../types";
import {
  addAgent,
  addCommunicationLink,
  configurationErrors,
  draftFrom,
  parseConfiguration,
  removeAgent,
  updateAgent,
} from "./editorModel";
import { canFitGraph, layoutTeam, pulseProgress } from "./model";

const settings: Settings = {
  model: "test-model",
  base_url: "http://localhost:11434",
  coordinator: "lead",
  reviewer: "review",
  max_concurrency: 1,
  web_enabled: false,
  team: [
    {
      name: "lead",
      role: "Coordinator",
      persona: "Plan carefully",
      skills: ["planning"],
      tools: ["send_message"],
    },
    {
      name: "worker",
      role: "Builder",
      persona: "Deliver files",
      skills: ["building"],
      tools: ["write_file", "send_message"],
    },
    {
      name: "review",
      role: "Reviewer",
      persona: "Review files",
      skills: ["review"],
      tools: ["read_file"],
    },
  ],
};

describe("configuration editing", () => {
  it("saves reusable teams without a mission while requiring one to create a run", () => {
    const draft = draftFrom(settings);
    expect(configurationErrors(draft)).toEqual([]);
    expect(configurationErrors(draft, [], true)).toContain(
      "Enter a mission goal.",
    );
  });
  it("forks an immutable runtime configuration", () => {
    const draft = draftFrom(settings, "Make a document");
    draft.settings.team[1].tools.push("read_file");
    expect(settings.team[1].tools).toEqual(["write_file", "send_message"]);
    expect(configurationErrors(draft)).toEqual([]);
  });
  it("renames an agent atomically through designation, paths, and saved positions", () => {
    const draft = draftFrom(
      {
        ...settings,
        communication_links: [{ source: "lead", target: "worker" }],
      },
      "Mission",
    );
    const position = draft.positions.lead;
    const result = updateAgent(draft, "lead", { name: "planner", tools: [] });
    expect(result.settings.coordinator).toBe("planner");
    expect(result.settings.communication_links).toEqual([
      { source: "planner", target: "worker" },
    ]);
    expect(result.positions.planner).toEqual(position);
    expect(result.positions.lead).toBeUndefined();
    expect(draft.settings.coordinator).toBe("lead");
    expect(result.settings.team[0].tools).toEqual([]);
  });
  it("rejects collisions without damaging the draft", () => {
    const draft = draftFrom(settings, "Mission");
    expect(() => updateAgent(draft, "lead", { name: "worker" })).toThrow(
      "unique",
    );
    expect(draft.settings.team[0].name).toBe("lead");
    expect(() => updateAgent(draft, "lead", { name: "" })).toThrow("lowercase");
  });
  it("removes a node with its paths and layout and repairs required designations", () => {
    const draft = addAgent(
      draftFrom(
        {
          ...settings,
          communication_links: [
            { source: "lead", target: "worker" },
            { source: "worker", target: "review" },
          ],
        },
        "Mission",
      ),
    );
    const result = removeAgent(draft, "lead");
    expect(result.settings.coordinator).toBe("worker");
    expect(result.settings.reviewer).toBe("review");
    expect(result.settings.communication_links).toEqual([
      { source: "worker", target: "review" },
    ]);
    expect(result.positions.lead).toBeUndefined();
    expect(configurationErrors(result)).toEqual([]);
  });
  it("switches unrestricted messaging to explicit paths and deduplicates connections", () => {
    const draft = draftFrom(settings, "Mission");
    const link = { source: "lead", target: "worker" };
    const result = addCommunicationLink(draft, link);
    expect(result.settings.communication_links).toEqual([link]);
    expect(addCommunicationLink(result, link)).toBe(result);
    expect(
      addCommunicationLink(result, { source: "worker", target: "worker" }),
    ).toBe(result);
  });
  it("prevents a team from launching without a worker's actual write permission", () => {
    const draft = updateAgent(draftFrom(settings, "Mission"), "worker", {
      tools: [],
    });
    expect(configurationErrors(draft)).toContain(
      "At least one agent other than the reviewer needs write_file access.",
    );
  });
  it("round trips configuration and keeps finite positions for current agents", () => {
    const draft = draftFrom(settings, "Mission", {
      worker: { x: 345, y: 123 },
    });
    const result = parseConfiguration(
      JSON.stringify({
        ...draft,
        positions: {
          ...draft.positions,
          ghost: { x: 0, y: 0 },
          lead: { x: "wrong", y: 2 },
        },
      }),
    );
    expect(result.positions.worker).toEqual({ x: 345, y: 123 });
    expect(result.positions.ghost).toBeUndefined();
    expect(Number.isFinite(result.positions.lead.x)).toBe(true);
    expect(result.settings).toEqual(settings);
  });
  it("rejects malformed imports before they reach the graph", () => {
    expect(() => parseConfiguration(JSON.stringify({ version: 2 }))).toThrow(
      "version 1",
    );
    const draft = draftFrom(settings, "Mission");
    expect(() =>
      parseConfiguration(
        JSON.stringify({
          ...draft,
          settings: { ...settings, team: [{ name: "bad" }] },
        }),
      ),
    ).toThrow("invalid agents");
    expect(() =>
      parseConfiguration(
        JSON.stringify({
          ...draft,
          settings: { ...settings, communication_links: [{ source: "lead" }] },
        }),
      ),
    ).toThrow("invalid communication paths");
  });
});

describe("graph geometry regressions", () => {
  it("does not fit an unmeasured, empty, zero-size, or nonfinite graph", () => {
    expect(canFitGraph(900, 400, true, 3)).toBe(true);
    expect(canFitGraph(900, 400, false, 3)).toBe(false);
    expect(canFitGraph(900, 400, true, 0)).toBe(false);
    expect(canFitGraph(0, 400, true, 3)).toBe(false);
    expect(canFitGraph(900, Number.NaN, true, 3)).toBe(false);
  });
  it("gives deterministic finite layouts for disconnected and cyclic channels", () => {
    for (const links of [
      [],
      [
        { source: "lead", target: "worker" },
        { source: "worker", target: "lead" },
      ],
    ]) {
      const layout = layoutTeam({ ...settings, communication_links: links });
      expect(Object.keys(layout)).toEqual(["lead", "worker", "review"]);
      expect(
        Object.values(layout).every(
          (point) => Number.isFinite(point.x) && Number.isFinite(point.y),
        ),
      ).toBe(true);
    }
  });
  it("clamps a packet to the displayed path endpoints across delayed frames", () => {
    expect(pulseProgress(100, 1900, 0)).toBe(0);
    expect(pulseProgress(100, 1900, 1000)).toBe(0.5);
    expect(pulseProgress(100, 1900, 2000)).toBe(1);
    expect(pulseProgress(100, 100, 101)).toBe(1);
  });
});
