import { describe, expect, it } from "vitest";
import { applyMissionSample, missionCatalogue } from "./missionSamples";
import type { MissionSample, Settings } from "./types";

describe("mission samples", () => {
  it("provides every difficulty when an older server omits or empties the catalogue", () => {
    for (const samples of [undefined, []]) {
      const loaded = missionCatalogue(samples);
      expect(loaded).toHaveLength(6);
      expect(new Set(loaded.map((sample) => sample.difficulty))).toEqual(
        new Set(["easy", "medium", "elaborate"]),
      );
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
