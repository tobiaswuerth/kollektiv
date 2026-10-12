import type { MissionSample, Settings } from "./types";
import catalogue from "../../src/kollektiv/mission_samples.json";

export const bundledMissionSamples = catalogue as MissionSample[];

export function missionCatalogue(samples?: MissionSample[]) {
  // A server started before a frontend update can have an older catalogue.
  const catalogue = new Map(
    bundledMissionSamples.map((sample) => [sample.id, sample]),
  );
  for (const sample of samples || [])
    if (!catalogue.has(sample.id)) catalogue.set(sample.id, sample);
  return [...catalogue.values()];
}

export const missionLevels = [
  { value: "easy", label: "Easy · one artifact" },
  { value: "medium", label: "Medium · shared work" },
  { value: "elaborate", label: "Elaborate · multiple artifacts" },
] as const;

export function applyMissionSample(sample: MissionSample, settings: Settings) {
  const loaded = {
    ...settings,
    ...structuredClone(sample.settings),
  } as Settings;
  return {
    goal: sample.goal,
    settings: loaded,
    ...(sample.positions
      ? { positions: structuredClone(sample.positions) }
      : {}),
  };
}
