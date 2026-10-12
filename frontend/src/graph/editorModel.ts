import type {
  Agent,
  CommunicationLink,
  GraphPosition,
  Settings,
  TeamConfiguration,
} from "../types";
import { layoutTeam } from "./model";

export function draftFrom(
  settings: Settings,
  goal = "",
  positions: TeamConfiguration["positions"] = {},
): TeamConfiguration {
  return {
    version: 1,
    name: "My collective",
    goal,
    settings: structuredClone(settings),
    positions: { ...layoutTeam(settings), ...structuredClone(positions) },
  };
}

export function addAgent(draft: TeamConfiguration): TeamConfiguration {
  const names = new Set(draft.settings.team.map((agent) => agent.name));
  let number = 1;
  while (names.has(`agent_${number}`)) number++;
  const name = `agent_${number}`;
  const agent: Agent = {
    name,
    role: "Specialist",
    persona:
      "Work carefully, use available tools, and communicate useful results to the team.",
    skills: [],
    tools: [
      "list_files",
      "read_file",
      "write_file",
      "check_file",
      "send_message",
      "share_note",
    ],
  };
  return {
    ...draft,
    settings: { ...draft.settings, team: [...draft.settings.team, agent] },
    positions: {
      ...draft.positions,
      [name]: { x: 70 + names.size * 35, y: 70 + names.size * 25 },
    },
  };
}

export function updateAgent(
  draft: TeamConfiguration,
  name: string,
  fields: Partial<Agent>,
): TeamConfiguration {
  const nextName = fields.name ?? name;
  if (fields.name !== undefined && !/^[a-z][a-z0-9_-]{0,39}$/.test(fields.name))
    throw new Error(
      "Agent names must start with a lowercase letter and use at most 40 lowercase letters, numbers, underscores or hyphens.",
    );
  if (
    nextName !== name &&
    draft.settings.team.some((agent) => agent.name === nextName)
  )
    throw new Error("Agent names must be unique.");
  const positions = { ...draft.positions };
  if (name !== nextName) {
    positions[nextName] = positions[name] || { x: 0, y: 0 };
    delete positions[name];
  }
  const rename = (value: string) => (value === name ? nextName : value);
  return {
    ...draft,
    positions,
    settings: {
      ...draft.settings,
      team: draft.settings.team.map((agent) =>
        agent.name === name ? { ...agent, ...fields } : agent,
      ),
      coordinator: rename(draft.settings.coordinator),
      reviewer: rename(draft.settings.reviewer),
      communication_links:
        draft.settings.communication_links?.map((link) => ({
          source: rename(link.source),
          target: rename(link.target),
        })) ?? null,
    },
  };
}

export function removeAgent(
  draft: TeamConfiguration,
  name: string,
): TeamConfiguration {
  const team = draft.settings.team.filter((agent) => agent.name !== name);
  const positions = { ...draft.positions };
  delete positions[name];
  const coordinator =
    draft.settings.coordinator === name
      ? team[0]?.name || ""
      : draft.settings.coordinator;
  const reviewer =
    draft.settings.reviewer === name
      ? team.find((agent) => agent.name !== coordinator)?.name || ""
      : draft.settings.reviewer;
  return {
    ...draft,
    positions,
    settings: {
      ...draft.settings,
      team,
      coordinator,
      reviewer,
      communication_links:
        draft.settings.communication_links?.filter(
          (link) => link.source !== name && link.target !== name,
        ) ?? null,
    },
  };
}

export function addCommunicationLink(
  draft: TeamConfiguration,
  link: CommunicationLink,
): TeamConfiguration {
  const links = draft.settings.communication_links || [];
  if (
    link.source === link.target ||
    links.some(
      (entry) => entry.source === link.source && entry.target === link.target,
    )
  )
    return draft;
  if (
    !draft.settings.team.some((agent) => agent.name === link.source) ||
    !draft.settings.team.some((agent) => agent.name === link.target)
  )
    return draft;
  return {
    ...draft,
    settings: { ...draft.settings, communication_links: [...links, link] },
  };
}

export function configurationErrors(
  draft: TeamConfiguration,
  tools: string[] = [],
  requireGoal = false,
): string[] {
  const errors: string[] = [];
  if (!draft.name.trim()) errors.push("Give the configuration a name.");
  if (requireGoal && !draft.goal.trim()) errors.push("Enter a mission goal.");
  const { team, coordinator, reviewer, communication_links } = draft.settings;
  if (team.length < 3)
    errors.push(
      "Use at least three agents: coordinator, worker, and reviewer.",
    );
  const names = team.map((agent) => agent.name);
  if (new Set(names).size !== names.length)
    errors.push("Agent names must be unique.");
  team.forEach((agent) => {
    if (!/^[a-z][a-z0-9_-]{0,39}$/.test(agent.name))
      errors.push(
        `Invalid agent name: ${agent.name || "(empty)"}. Use lowercase letters, numbers, underscores or hyphens.`,
      );
    if (!agent.role.trim() || !agent.persona.trim())
      errors.push(
        `${agent.name || "Each agent"} needs a role and instructions.`,
      );
    if (tools.length && agent.tools.some((tool) => !tools.includes(tool)))
      errors.push(`${agent.name} has an unavailable tool.`);
    if (
      agent.tools.includes("hire") &&
      (!agent.tools.includes("monitor_agents") ||
        !agent.tools.includes("instruct_agent"))
    )
      errors.push(
        `${agent.name}: hire requires monitor_agents and instruct_agent.`,
      );
  });
  if (!names.includes(coordinator) || !names.includes(reviewer))
    errors.push("Choose a coordinator and reviewer from the team.");
  if (coordinator === reviewer)
    errors.push("Coordinator and reviewer must be different agents.");
  if (
    !team.some(
      (agent) => agent.name !== reviewer && agent.tools.includes("write_file"),
    )
  )
    errors.push(
      "At least one agent other than the reviewer needs write_file access.",
    );
  if (
    communication_links?.some(
      (link) =>
        link.source === link.target ||
        !names.includes(link.source) ||
        !names.includes(link.target),
    )
  )
    errors.push(
      "Communication paths must connect two different agents in the team.",
    );
  return errors;
}

export function setToolPermission(
  tools: string[],
  tool: string,
  enabled: boolean,
) {
  const selected = new Set(tools);
  if (enabled) {
    selected.add(tool);
    if (tool === "hire") {
      selected.add("monitor_agents");
      selected.add("instruct_agent");
    }
  } else {
    selected.delete(tool);
    if (tool === "monitor_agents" || tool === "instruct_agent")
      selected.delete("hire");
  }
  return [...selected];
}

export function parseConfiguration(content: string): TeamConfiguration {
  const value = JSON.parse(content);
  if (
    !value ||
    value.version !== 1 ||
    typeof value.name !== "string" ||
    typeof value.goal !== "string" ||
    !value.settings ||
    !Array.isArray(value.settings.team)
  )
    throw new Error("Choose a version 1 collective configuration JSON file.");
  const settings = value.settings;
  if (
    typeof settings.model !== "string" ||
    typeof settings.base_url !== "string" ||
    typeof settings.coordinator !== "string" ||
    typeof settings.reviewer !== "string" ||
    typeof settings.max_concurrency !== "number" ||
    typeof settings.web_enabled !== "boolean"
  )
    throw new Error("The configuration has incomplete runtime settings.");
  if (
    settings.team.some(
      (agent: Agent) =>
        !agent ||
        typeof agent.name !== "string" ||
        typeof agent.role !== "string" ||
        typeof agent.persona !== "string" ||
        !Array.isArray(agent.skills) ||
        !agent.skills.every((skill) => typeof skill === "string") ||
        !Array.isArray(agent.tools) ||
        !agent.tools.every((tool) => typeof tool === "string"),
    )
  )
    throw new Error("The configuration has invalid agents.");
  if (
    settings.communication_links != null &&
    (!Array.isArray(settings.communication_links) ||
      settings.communication_links.some(
        (link: CommunicationLink) =>
          !link ||
          typeof link.source !== "string" ||
          typeof link.target !== "string",
      ))
  )
    throw new Error("The configuration has invalid communication paths.");
  const positions = Object.fromEntries(
    Object.entries(value.positions || {}).filter(
      ([name, point]) =>
        settings.team.some((agent: Agent) => agent.name === name) &&
        point &&
        typeof point === "object" &&
        Number.isFinite((point as { x: number }).x) &&
        Number.isFinite((point as { y: number }).y),
    ),
  ) as Record<string, GraphPosition>;
  return {
    version: 1,
    name: value.name,
    goal: value.goal,
    settings,
    positions: { ...layoutTeam(settings), ...positions },
  };
}
