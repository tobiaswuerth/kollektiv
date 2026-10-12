export type Agent = {
  name: string;
  role: string;
  persona: string;
  skills: string[];
  tools: string[];
  model?: string;
};
export type Settings = {
  model: string;
  base_url: string;
  team: Agent[];
  coordinator: string;
  reviewer: string;
  max_concurrency: number;
  web_enabled: boolean;
  [key: string]: unknown;
};
export type Task = {
  id: string;
  title: string;
  phase: string;
  description: string;
  owner: string;
  output: string;
  status: string;
  attempts: number;
  dependencies: string[];
  acceptance_criteria: string[];
  feedback: string;
  artifact?: { words: number; bytes: number; sha256: string };
};
export type RuntimeEvent = {
  id: string;
  at: string;
  kind: string;
  agent?: string;
  sender?: string;
  recipient?: string;
  content?: string;
  node?: string;
  task?: string;
  purpose?: string;
  tool?: string;
  status?: string;
  feedback?: string;
  error?: string;
  path?: string;
  tool_calls?: { name: string; args: unknown; id: string }[];
  usage?: { output_tokens: number };
  [key: string]: unknown;
};
export type Control = {
  mode: string;
  paused: boolean;
  waiting: boolean;
  running: boolean;
  stopped: boolean;
  steps_pending: number;
  active_nodes: { node: string; agent?: string; task?: string }[];
};
export type Checkpoint = {
  status?: string;
  plan?: { summary: string };
  tasks?: Record<string, Task>;
  messages?: Record<string, RuntimeEvent>;
  notes?: Record<string, RuntimeEvent>;
  goal_revision?: number;
  final_summary?: string;
  [key: string]: unknown;
};
export type Run = {
  run_id: string;
  goal: string;
  status: string;
  settings: Settings;
  running?: boolean;
};
export type Snapshot = {
  manifest: Run;
  state: Checkpoint;
  running: boolean;
  control: Control | null;
  error: string | null;
  events: RuntimeEvent[];
  cursor: number;
  artifacts: { path: string; bytes: number }[];
};
export type EventBatch = Pick<
  Snapshot,
  "events" | "cursor" | "running" | "control" | "error"
>;
export type Selection = {
  type: "agent" | "task" | "event" | "artifact" | "config" | "channel";
  id: string;
} | null;
export type Pulse = { event: RuntimeEvent; start: number; end: number };
export type Tab = "messages" | "memory" | "debug" | "state" | "artifacts";
