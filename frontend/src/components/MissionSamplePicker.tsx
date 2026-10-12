import { useId } from "react";
import { missionLevels } from "../missionSamples";
import type { MissionSample } from "../types";
import "./missionSamples.css";

export function MissionSamplePicker({
  samples,
  value,
  onChange,
}: {
  samples: MissionSample[];
  value: string;
  onChange: (id: string) => void;
}) {
  const selectId = useId();
  const descriptionId = `${selectId}-description`;
  const sample = samples.find((item) => item.id === value);
  return (
    <div className="mission-samples">
      <label htmlFor={selectId}>Mission sample</label>
      <select
        id={selectId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-describedby={descriptionId}
      >
        <option value="">Custom mission</option>
        {missionLevels.map((level) => (
          <optgroup key={level.value} label={level.label}>
            {samples
              .filter((item) => item.difficulty === level.value)
              .map((item) => (
                <option key={item.id} value={item.id}>
                  {item.title}
                </option>
              ))}
          </optgroup>
        ))}
      </select>
      <p id={descriptionId} aria-live="polite">
        {sample
          ? `${sample.description} The goal below is editable; your team, model, and concurrency are preserved.`
          : "Choose a starting mission or write your own. Samples adjust task budgets and keep your team and model."}
      </p>
    </div>
  );
}
