import { useLayoutEffect, useRef } from "react";
import { BaseEdge, getBezierPath, type EdgeProps } from "@xyflow/react";
import type { Pulse } from "../types";
import { pulseProgress } from "./model";

export type FlowEdgeData = {
  pulses?: Pulse[];
  count?: number;
  tone: string;
  label?: string;
};

/** Read the displayed path itself so packets remain attached while endpoints move. */
export function FlowEdge(props: EdgeProps) {
  const self = props.source === props.target;
  const [normal, labelX, labelY] = getBezierPath(props);
  const path = self
    ? `M ${props.sourceX} ${props.sourceY} C ${props.sourceX + 90} ${props.sourceY - 70}, ${props.targetX + 90} ${props.targetY + 80}, ${props.targetX} ${props.targetY}`
    : normal;
  const data = props.data as FlowEdgeData;
  const geometry = useRef<SVGPathElement>(null);
  const packets = useRef(new Map<string, SVGCircleElement>());
  useLayoutEffect(() => {
    const line = geometry.current;
    if (!line || !data.pulses?.length) return;
    const length = line.getTotalLength();
    let frame = 0;
    const draw = () => {
      const now = Date.now();
      let pending = false;
      for (const pulse of data.pulses || []) {
        const circle = packets.current.get(pulse.event.id);
        if (!circle) continue;
        const progress = pulseProgress(pulse.start, pulse.end, now);
        const point = line.getPointAtLength(length * progress);
        circle.setAttribute("cx", String(point.x));
        circle.setAttribute("cy", String(point.y));
        circle.dataset.flowProgress = String(progress);
        pending ||= progress < 1;
      }
      if (pending) frame = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [path, data.pulses]);
  return (
    <>
      <BaseEdge
        {...props}
        path={path}
        style={{
          stroke: data.tone,
          opacity: data.pulses?.length ? 0.9 : 0.35,
          strokeWidth: data.pulses?.length ? 2 : 1.2,
        }}
      />
      <path
        ref={geometry}
        d={path}
        fill="none"
        stroke="none"
        className="flow-geometry"
        data-flow-path={props.id}
      />
      {data.pulses?.map((pulse) => (
        <circle
          key={pulse.event.id}
          ref={(element) => {
            if (element) packets.current.set(pulse.event.id, element);
            else packets.current.delete(pulse.event.id);
          }}
          r="4"
          fill={data.tone}
          className="flow-packet"
          data-flow-event={pulse.event.id}
          data-flow-edge={props.id}
        />
      ))}
      {!self && !!data.count && (
        <g className="edge-label" transform={`translate(${labelX},${labelY})`}>
          <rect x="-13" y="-9" width="26" height="18" rx="8" />
          <text textAnchor="middle" dy="4">
            {data.count}
          </text>
        </g>
      )}
      {self && data.pulses?.length ? (
        <text
          x={props.sourceX + 30}
          y={props.sourceY - 26}
          className="tool-edge-label"
        >
          {data.label}
        </text>
      ) : null}
    </>
  );
}
