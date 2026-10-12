import { useEffect, useRef, type RefObject } from "react";
import { Panel, useNodesInitialized, useReactFlow } from "@xyflow/react";
import { Scan } from "lucide-react";
import { canFitGraph } from "./model";

/** Fit only measured nodes in a visible viewport; never race animated camera jobs. */
export function GraphViewport({
  container,
  identity,
  count,
}: {
  container: RefObject<HTMLDivElement | null>;
  identity: string;
  count: number;
}) {
  const initialized = useNodesInitialized();
  const { fitView } = useReactFlow();
  const fitRef = useRef(fitView);
  fitRef.current = fitView;
  const lastSize = useRef("");
  const fit = () => {
    const box = container.current?.getBoundingClientRect();
    if (box && canFitGraph(box.width, box.height, initialized, count)) {
      void fitRef.current({ padding: 0.14, duration: 0 });
    }
  };
  useEffect(() => {
    const element = container.current;
    if (!element || !initialized || !count) return;
    let frame = 0;
    const schedule = () => {
      const box = element.getBoundingClientRect();
      const size = `${Math.round(box.width)}:${Math.round(box.height)}`;
      if (
        !canFitGraph(box.width, box.height, initialized, count) ||
        size === lastSize.current
      )
        return;
      lastSize.current = size;
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(
        () => void fitRef.current({ padding: 0.14, duration: 0 }),
      );
    };
    lastSize.current = "";
    const observer = new ResizeObserver(schedule);
    observer.observe(element);
    schedule();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [container, identity, initialized, count]);
  return (
    <Panel position="top-left">
      <button
        type="button"
        className="graph-fit button"
        onClick={fit}
        aria-label="Reset graph view"
      >
        <Scan size={13} />
        Fit graph
      </button>
    </Panel>
  );
}
