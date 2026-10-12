import { describe, expect, it } from "vitest";
import { bundledMissionSamples } from "../missionSamples";
import { layoutEditorTeam } from "./editorLayout";
import type { Settings } from "../types";

describe("large-team editor layout", () => {
  it("keeps a cyclic fourteen-agent studio broad, distinct and bounded", () => {
    const studio = bundledMissionSamples.find(
      (sample) => sample.id === "fifty-page-novel-studio",
    )!;
    const positions = layoutEditorTeam(studio.settings as Settings);
    expect(Object.keys(positions)).toHaveLength(14);
    expect(
      new Set(Object.values(positions).map(({ x, y }) => `${x}:${y}`)).size,
    ).toBe(14);
    const points = Object.values(positions);
    expect(Math.max(...points.map((point) => point.x))).toBeLessThanOrEqual(
      780,
    );
    expect(Math.max(...points.map((point) => point.y))).toBeLessThanOrEqual(
      648,
    );
    expect(positions[studio.settings!.coordinator!].y).toBe(0);
    expect(positions[studio.settings!.reviewer!].y).toBeGreaterThan(
      positions[studio.settings!.team![4].name].y,
    );
  });
  it("keeps shared coordination and review in one top-level node for a large team", () => {
    const studio = bundledMissionSamples.find(
      (sample) => sample.id === "fifty-page-novel-studio",
    )!;
    const settings = {
      ...studio.settings,
      reviewer: studio.settings!.coordinator,
    } as Settings;
    const positions = layoutEditorTeam(settings);
    expect(Object.keys(positions)).toHaveLength(settings.team.length);
    expect(positions[settings.coordinator].y).toBe(0);
    expect(
      new Set(Object.values(positions).map(({ x, y }) => `${x}:${y}`)).size,
    ).toBe(settings.team.length);
  });
});
