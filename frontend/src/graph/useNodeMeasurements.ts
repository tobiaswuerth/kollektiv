import { useCallback, useState } from "react";
import type { NodeChange } from "@xyflow/react";

/** Controlled nodes must retain React Flow's measured dimensions across renders. */
export function useNodeMeasurements() {
  const [measurements, setMeasurements] = useState<
    Record<string, { width: number; height: number }>
  >({});
  const acceptMeasurements = useCallback((changes: NodeChange[]) => {
    const dimensions = changes.filter(
      (change) => change.type === "dimensions" && change.dimensions,
    );
    if (!dimensions.length) return;
    setMeasurements((previous) => {
      let next = previous;
      for (const change of dimensions) {
        if (change.type !== "dimensions" || !change.dimensions) continue;
        if (
          previous[change.id]?.width === change.dimensions.width &&
          previous[change.id]?.height === change.dimensions.height
        )
          continue;
        if (next === previous) next = { ...previous };
        next[change.id] = change.dimensions;
      }
      return next;
    });
  }, []);
  return { measurements, acceptMeasurements };
}
