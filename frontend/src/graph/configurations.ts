import { api } from "../api";
import type { TeamConfiguration } from "../types";

export async function listConfigurations() {
  return (
    await api<{ configurations: TeamConfiguration[] }>("/api/configurations")
  ).configurations;
}

export function saveConfiguration(configuration: TeamConfiguration) {
  return api<TeamConfiguration>("/api/configurations", configuration);
}

export async function deleteConfiguration(id: string) {
  const response = await fetch(
    `/api/configurations/${encodeURIComponent(id)}`,
    { method: "DELETE", signal: AbortSignal.timeout(30_000) },
  );
  const result = await response.json();
  if (!response.ok)
    throw new Error(result.error || "Could not remove the configuration.");
}
