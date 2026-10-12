import { layoutTeam } from "./model";
import type { GraphPosition, Settings } from "../types";

/** A broad specialist grid keeps long, cyclic message networks readable. */
export function layoutEditorTeam(
  settings: Settings,
): Record<string, GraphPosition> {
  if (settings.team.length <= 7) return layoutTeam(settings);
  const workers = settings.team.filter(
    (agent) => ![settings.coordinator, settings.reviewer].includes(agent.name),
  );
  const columns = Math.min(4, Math.ceil(Math.sqrt(workers.length)));
  const rows = Math.ceil(workers.length / columns);
  const width = 260;
  const height = 162;
  const positions: Record<string, GraphPosition> = {};
  positions[settings.coordinator] = { x: ((columns - 1) * width) / 2, y: 0 };
  workers.forEach((agent, index) => {
    positions[agent.name] = {
      x: (index % columns) * width,
      y: (Math.floor(index / columns) + 1) * height,
    };
  });
  if (settings.reviewer !== settings.coordinator)
    positions[settings.reviewer] = {
      x: ((columns - 1) * width) / 2,
      y: (rows + 1) * height,
    };
  return positions;
}
