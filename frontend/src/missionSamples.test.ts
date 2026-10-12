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
      expect(loaded).toHaveLength(bundledMissionSamples.length);
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
  it("replaces either large studio with a genuinely small easy or medium team", () => {
    const local = {
      model: "my-local-model",
      base_url: "http://localhost:9999",
      max_concurrency: 2,
      web_enabled: false,
      coordinator: "lead",
      reviewer: "reviewer",
      team: [],
      communication_links: [],
    } satisfies Settings;
    for (const studioId of [
      "scientific-paper-lab",
      "fifty-page-novel-studio",
    ]) {
      const studio = bundledMissionSamples.find(
        (sample) => sample.id === studioId,
      )!;
      const current = applyMissionSample(studio, local).settings;
      for (const [sampleId, count] of [
        ["pip-field-card", 3],
        ["picnic-plan", 3],
        ["pip-silent-adventure", 4],
        ["board-game-kit", 4],
      ] as const) {
        const sample = bundledMissionSamples.find(
          (entry) => entry.id === sampleId,
        )!;
        const result = applyMissionSample(sample, current);
        const names = new Set(result.settings.team.map((agent) => agent.name));
        expect(result.settings.team).toHaveLength(count);
        expect(new Set(Object.keys(result.positions!))).toEqual(names);
        expect(result.settings.communication_links).toEqual(
          sample.settings!.communication_links,
        );
        expect(result.settings.max_tasks).toBeLessThanOrEqual(6);
        expect(result.settings.num_ctx).toBe(32768);
        expect(result.settings.num_predict).toBe(4096);
        expect(result.settings.context_chars).toBe(24000);
        expect(result.settings.request_timeout).toBe(180);
        expect(result.settings.web_enabled).toBe(false);
        expect(result.settings.model).toBe(local.model);
        expect(result.settings.base_url).toBe(local.base_url);
        expect(result.settings.max_concurrency).toBe(local.max_concurrency);
        expect(current.team).toHaveLength(studio.settings!.team!.length);
      }
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
  it("provides a three-agent easy hiring mission with one narrowly permitted manager", () => {
    const sample = bundledMissionSamples.find(
      (entry) => entry.id === "hire-a-garden-specialist",
    )!;
    expect(sample.difficulty).toBe("easy");
    expect(sample.settings!.team).toHaveLength(3);
    const writer = sample.settings!.team!.find(
      (agent) => agent.name === "builder",
    )!;
    expect(writer.tools).toEqual(
      expect.arrayContaining([
        "hire",
        "fire",
        "monitor_agents",
        "instruct_agent",
      ]),
    );
    for (const agent of sample.settings!.team!.filter(
      (entry) => entry.name !== "builder",
    ))
      expect(agent.tools).not.toContain("hire");
    expect(sample.settings!.max_dynamic_agents).toBe(1);
    expect(sample.goal).toContain("delegates/herb_specialist/notes.md");
    expect(sample.goal).toContain(
      "retire the completed helper while retaining its notes",
    );
    expect(Object.keys(sample.positions!)).toHaveLength(3);
  });
  it("loads a generic one-orchestrator template without authoring tools or studio carryover", () => {
    const studio = bundledMissionSamples.find(
      (sample) => sample.id === "fifty-page-novel-studio",
    )!;
    const solo = bundledMissionSamples.find(
      (sample) => sample.id === "solo-orchestrator",
    )!;
    const local = {
      model: "local-model",
      base_url: "http://localhost:9999",
      max_concurrency: 2,
    } as Settings;
    const large = applyMissionSample(studio, local).settings;
    const loaded = applyMissionSample(solo, large);
    expect(loaded.goal).toMatch(/^\[YOUR TASK HERE\]/);
    expect(loaded.settings.team).toHaveLength(1);
    expect(loaded.settings.coordinator).toBe("orchestrator");
    expect(loaded.settings.reviewer).toBe("orchestrator");
    expect(loaded.settings.team[0].tools).not.toContain("write_file");
    expect(loaded.settings.team[0].tools).toEqual(
      expect.arrayContaining([
        "hire",
        "fire",
        "monitor_agents",
        "instruct_agent",
        "assemble_files",
      ]),
    );
    expect(loaded.settings.communication_links).toEqual([]);
    expect(loaded.positions).toEqual({ orchestrator: { x: 0, y: 0 } });
    expect(loaded.settings.model).toBe(local.model);
    expect(loaded.settings.base_url).toBe(local.base_url);
    expect(loaded.settings.max_concurrency).toBe(local.max_concurrency);
    expect(loaded.settings.num_ctx).toBe(32768);
    expect(loaded.settings.num_predict).toBe(8192);
    expect(loaded.settings.context_chars).toBe(60000);
    expect(large.team).toHaveLength(14);
  });
});
