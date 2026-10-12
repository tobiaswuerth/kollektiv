import { describe, expect, it } from "vitest";
import {
  applyMissionSample,
  bundledMissionSamples,
  missionCatalogue,
} from "./missionSamples";
import type { MissionSample, Settings } from "./types";

describe("mission samples", () => {
  it("provides every difficulty when an older server omits or empties the catalogue", () => {
    for (const samples of [undefined, []]) {
      const loaded = missionCatalogue(samples);
      expect(loaded).toHaveLength(8);
      expect(new Set(loaded.map((sample) => sample.difficulty))).toEqual(
        new Set(["easy", "medium", "elaborate"]),
      );
    }
  });
  it("includes new studios even when a running server still returns the old six samples", () => {
    const loaded = missionCatalogue(bundledMissionSamples.slice(0, 6));
    expect(loaded.map((sample) => sample.id)).toContain("scientific-paper-lab");
    expect(loaded.map((sample) => sample.id)).toContain(
      "fifty-page-novel-studio",
    );
  });
  it("loads each complete studio graph and layout while preserving the selected model and concurrency", () => {
    const current = {
      model: "my-local-model",
      base_url: "http://localhost:9999",
      max_concurrency: 2,
      web_enabled: false,
      coordinator: "lead",
      reviewer: "reviewer",
      team: [],
      communication_links: [],
    } satisfies Settings;
    for (const [id, count] of [
      ["scientific-paper-lab", 12],
      ["fifty-page-novel-studio", 14],
    ] as const) {
      const sample = bundledMissionSamples.find((entry) => entry.id === id)!;
      const result = applyMissionSample(sample, current);
      expect(result.settings.team).toHaveLength(count);
      expect(result.settings.communication_links!.length).toBeGreaterThan(
        count,
      );
      expect(Object.keys(result.positions!)).toHaveLength(count);
      expect(result.settings.model).toBe(current.model);
      expect(result.settings.base_url).toBe(current.base_url);
      expect(result.settings.max_concurrency).toBe(current.max_concurrency);
      const originalRole = sample.settings!.team![0].role;
      result.settings.team[0].role = "Changed in draft";
      expect(sample.settings!.team![0].role).toBe(originalRole);
      expect(current.team).toHaveLength(0);
    }
  });
  it("loads the mission's budget while preserving the edited graph, model, and concurrency", () => {
    const settings = {
      model: "my-local-model",
      base_url: "http://localhost:11434",
      max_concurrency: 3,
      web_enabled: true,
      coordinator: "planner",
      reviewer: "critic",
      team: [
        {
          name: "maker",
          role: "Custom role",
          persona: "Custom instructions",
          skills: [],
          tools: ["write_file"],
        },
      ],
      communication_links: [{ source: "planner", target: "maker" }],
      max_tasks: 24,
    } satisfies Settings;
    const sample = {
      id: "simple",
      title: "Simple mission",
      difficulty: "easy",
      description: "One file",
      goal: "Create result.md",
      settings: { max_tasks: 3 },
    } satisfies MissionSample;
    const loaded = applyMissionSample(sample, settings);
    expect(loaded.goal).toBe(sample.goal);
    expect(loaded.settings.max_tasks).toBe(3);
    expect(loaded.settings.team).toBe(settings.team);
    expect(loaded.settings.communication_links).toBe(
      settings.communication_links,
    );
    expect(loaded.settings.model).toBe(settings.model);
    expect(loaded.settings.max_concurrency).toBe(3);
    expect(settings.max_tasks).toBe(24);
  });
});
