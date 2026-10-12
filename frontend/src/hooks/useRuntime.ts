import { useEffect, useState } from "react";
import { api } from "../api";
import { useRoom } from "../store";
import { missionCatalogue } from "../missionSamples";
import type { EventBatch, MissionSample, Settings, Snapshot } from "../types";

export function useRuntime() {
  useEffect(() => {
    let disposed = false,
      timer: ReturnType<typeof setTimeout>,
      healthTimer: ReturnType<typeof setTimeout>;
    let lastSnapshot = 0,
      lastRuns = 0,
      lightweight = true;
    const refreshHealth = async () => {
      try {
        const id = useRoom.getState().runId;
        const health = await api<{
          connected: boolean;
          model: string;
          models: string[];
        }>(`/api/health${id ? `?run=${id}` : ""}`);
        if (!disposed && id === useRoom.getState().runId)
          useRoom.setState({ health });
      } catch {
        /* Transport status comes from event polling. */
      }
      if (!disposed) healthTimer = setTimeout(refreshHealth, 15_000);
    };
    const poll = async () => {
      const { runId, snapshot, mutationEpoch } = useRoom.getState();
      const start = performance.now();
      try {
        if (runId) {
          const cursor = snapshot?.cursor || 0;
          if (lightweight) {
            try {
              const events = await api<EventBatch>(
                `/api/runs/${runId}/events?cursor=${cursor}`,
              );
              if (
                !disposed &&
                useRoom.getState().runId === runId &&
                useRoom.getState().mutationEpoch === mutationEpoch
              )
                useRoom.getState().accept(events);
            } catch (e) {
              if ((e as { status?: number }).status === 404)
                lightweight = false;
              else throw e;
            }
          }
          if (
            useRoom.getState().runId === runId &&
            useRoom.getState().mutationEpoch === mutationEpoch &&
            (!lightweight ||
              Date.now() - lastSnapshot > (snapshot?.running ? 750 : 1800))
          ) {
            const data = await api<Snapshot>(
              `/api/runs/${runId}?cursor=${useRoom.getState().snapshot?.cursor || 0}`,
            );
            if (
              !disposed &&
              useRoom.getState().runId === runId &&
              useRoom.getState().mutationEpoch === mutationEpoch
            )
              useRoom.getState().accept(data);
            lastSnapshot = Date.now();
          }
        }
        if (Date.now() - lastRuns > 10_000) {
          await useRoom.getState().refreshRuns();
          lastRuns = Date.now();
        }
        if (
          !disposed &&
          useRoom.getState().runId === runId &&
          useRoom.getState().mutationEpoch === mutationEpoch
        )
          useRoom.setState({
            connected: true,
            latency: Math.round(performance.now() - start),
          });
      } catch {
        if (
          !disposed &&
          useRoom.getState().runId === runId &&
          useRoom.getState().mutationEpoch === mutationEpoch
        )
          useRoom.setState({ connected: false });
      }
      if (!disposed)
        timer = setTimeout(
          poll,
          document.hidden
            ? 1500
            : useRoom.getState().snapshot?.running
              ? 125
              : 500,
        );
    };
    (async () => {
      try {
        const data = await api<{
          settings: Settings;
          mission_samples?: MissionSample[];
        }>("/api/bootstrap");
        if (disposed) return;
        useRoom.setState({
          defaults: data.settings,
          missionSamples: missionCatalogue(data.mission_samples),
        });
        await useRoom.getState().refreshRuns();
        const runs = useRoom.getState().runs;
        if (runs.length && !useRoom.getState().runId)
          await useRoom.getState().chooseRun(runs[0].run_id);
      } catch (e) {
        useRoom.setState({ error: String(e), connected: false });
      }
      if (!disposed) {
        void poll();
        void refreshHealth();
      }
    })();
    return () => {
      disposed = true;
      clearTimeout(timer);
      clearTimeout(healthTimer);
    };
  }, []);
}

export function useClock(interval = 100) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), interval);
    return () => clearInterval(timer);
  }, [interval]);
  return now;
}
