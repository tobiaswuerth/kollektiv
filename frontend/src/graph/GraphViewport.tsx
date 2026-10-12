import { useEffect, useRef, type RefObject } from "react";
import { Panel, useNodesInitialized, useReactFlow } from "@xyflow/react";
import { Scan } from "lucide-react";
import { canFitGraph } from "./model";

/** Fit only measured nodes in a visible viewport; never race animated camera jobs. */
export function GraphViewport({
  container,
  identity,
  count,
  focusRequest,
}: {
  container: RefObject<HTMLDivElement | null>;
  identity: string;
  count: number;
  focusRequest?: { id: string; revision: number };
}) {
  const initialized = useNodesInitialized();
  const { fitView } = useReactFlow();
  const fitRef = useRef(fitView);
  fitRef.current = fitView;
  const lastSize = useRef("");
  const preferredNode = useRef("");
  const cameraFrame = useRef(0);
  const scheduleFit = () => {
    const box = container.current?.getBoundingClientRect();
    if (!box || !canFitGraph(box.width, box.height, initialized, count)) return;
    cancelAnimationFrame(cameraFrame.current);
    cameraFrame.current = requestAnimationFrame(() => {
      const id = preferredNode.current;
      void fitRef.current(
        id
          ? {
              nodes: [{ id }],
              padding: 0.6,
              minZoom: 0.9,
              maxZoom: 1.15,
              duration: 0,
            }
          : { padding: 0.14, duration: 0 },
      );
    });
  };
  const fit = () => {
    preferredNode.current = "";
    scheduleFit();
  };
  useEffect(() => {
    preferredNode.current = "";
    const element = container.current;
    if (!element || !initialized || !count) return;
    const schedule = () => {
      const box = element.getBoundingClientRect();
      const size = `${Math.round(box.width)}:${Math.round(box.height)}`;
      if (
        !canFitGraph(box.width, box.height, initialized, count) ||
        size === lastSize.current
      )
        return;
      lastSize.current = size;
      scheduleFit();
    };
    lastSize.current = "";
    const observer = new ResizeObserver(schedule);
    observer.observe(element);
    schedule();
    return () => {
      cancelAnimationFrame(cameraFrame.current);
      observer.disconnect();
    };
  }, [container, identity, initialized, count]);
  useEffect(() => {
    preferredNode.current = focusRequest?.id || "";
    if (!preferredNode.current || !initialized) return;
    scheduleFit();
  }, [focusRequest?.id, focusRequest?.revision, initialized]);
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
