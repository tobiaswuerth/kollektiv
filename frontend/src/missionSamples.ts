import type { MissionSample, Settings } from "./types";
import catalogue from "../../src/kollektiv/mission_samples.json";

export const bundledMissionSamples = catalogue as MissionSample[];

export function missionCatalogue(samples?: MissionSample[]) {
  return samples?.length ? samples : bundledMissionSamples;
}

export const missionLevels = [
  { value: "easy", label: "Easy · one artifact" },
  { value: "medium", label: "Medium · shared work" },
  { value: "elaborate", label: "Elaborate · multiple artifacts" },
] as const;

export function applyMissionSample(sample: MissionSample, settings: Settings) {
  return {
    goal: sample.goal,
    settings: { ...settings, ...sample.settings } as Settings,
  };
}
