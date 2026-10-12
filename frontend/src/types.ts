export type Agent = {
  name: string;
  role: string;
  persona: string;
  skills: string[];
  tools: string[];
  model?: string | null;
};
export type SpecialistStatus =
  | "pending"
  | "running"
  | "awaiting_children"
  | "reviewing"
  | "completed"
  | "failed"
  | "interrupted"
  | "retired";
export type Specialist = Agent & {
  id: string;
  parent: string;
  parent_id: string | null;
  level: 2 | 3;
  status: SpecialistStatus;
  assignment: string;
  output?: string | null;
  input_files: string[];
  acceptance_criteria: string[];
  task_id?: string;
  parent_task_id?: string;
  current_node?: string;
  last_tool?: string;
  summary?: string;
  error?: string;
  artifact?: { words: number; bytes: number; sha256: string };
  created_at: string;
  updated_at: string;
  history?: RuntimeEvent[];
};
export type Delegation = {
  agents: Record<string, Specialist>;
  max_depth: number;
};
export type CommunicationLink = { source: string; target: string };
export type GraphPosition = { x: number; y: number };
export type Settings = {
  model: string;
  base_url: string;
  team: Agent[];
  coordinator: string;
  reviewer: string;
  max_concurrency: number;
  web_enabled: boolean;
  communication_links?: CommunicationLink[] | null;
  [key: string]: unknown;
};
export type TeamConfiguration = {
  version: 1;
  id?: string;
  name: string;
  goal: string;
  settings: Settings;
  positions: Record<string, GraphPosition>;
  updated_at?: string;
};
export type MissionSample = {
  id: string;
  title: string;
  difficulty: "easy" | "medium" | "elaborate";
  description: string;
  goal: string;
  settings?: Partial<Settings>;
  positions?: Record<string, GraphPosition>;
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
  positions?: Record<string, GraphPosition>;
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
  delegation?: Delegation;
};
export type EventBatch = Pick<
  Snapshot,
  "events" | "cursor" | "running" | "control" | "error" | "delegation"
>;
export type Selection = {
  type:
    | "agent"
    | "task"
    | "event"
    | "artifact"
    | "config"
    | "channel"
    | "delegation";
  id: string;
} | null;
export type Pulse = { event: RuntimeEvent; start: number; end: number };
export type Tab = "messages" | "memory" | "debug" | "state" | "artifacts";
