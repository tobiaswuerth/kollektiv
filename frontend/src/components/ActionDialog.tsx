import { useState } from "react";
import { useRoom } from "../store";
import { Modal } from "./ui";
import { applyMissionSample } from "../missionSamples";
import { MissionSamplePicker } from "./MissionSamplePicker";
import type { GraphPosition, Settings } from "../types";
import { taskOwners } from "../selectors";
export type Action = { kind: string; fields?: Record<string, string> };
export function ActionDialog({
  action,
  close,
}: {
  action: Action;
  close: () => void;
}) {
  const { defaults, snapshot, missionSamples, createRun, intervene } =
    useRoom();
  const config = snapshot?.manifest.settings || defaults!;
  const [goal, setGoal] = useState(
    action.kind === "revise_goal" ? snapshot?.manifest.goal || "" : "",
  );
  const [model, setModel] = useState(config.model),
    [concurrency, setConcurrency] = useState(config.max_concurrency);
  const [settings, setSettings] = useState(JSON.stringify(config, null, 2));
  const [sampleId, setSampleId] = useState("");
  const [samplePositions, setSamplePositions] =
    useState<Record<string, GraphPosition>>();
  const [content, setContent] = useState(""),
    [recipient, setRecipient] = useState(action.fields?.recipient || "all");
  const [owner, setOwner] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const title =
    {
      create: "Create a collective",
      inject_message: "Send an instruction",
      instruct_agent: "Guide a specialist",
      share_note: "Share knowledge",
      retry_task: "Reopen task",
      reassign_task: "Reassign task",
      revise_goal: "Edit the shared goal",
    }[action.kind] || "Intervene";
  const loadSample = (id: string) => {
    if (!id) {
      setSampleId("");
      return;
    }
    const selected = missionSamples.find((item) => item.id === id);
    if (!selected) return;
    try {
      const loaded = applyMissionSample(selected, JSON.parse(settings));
      setGoal(loaded.goal);
      setSettings(JSON.stringify(loaded.settings, null, 2));
      setSamplePositions(loaded.positions);
      setSampleId(id);
      setError("");
    } catch {
      setError(
        "Fix the advanced settings JSON before loading a mission sample.",
      );
    }
  };
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (action.kind === "create") {
        const runtimeSettings = {
          ...JSON.parse(settings),
          model,
          max_concurrency: concurrency,
        } as Settings;
        const names = new Set(runtimeSettings.team.map((agent) => agent.name));
        const positions =
          samplePositions &&
          Object.fromEntries(
            Object.entries(samplePositions).filter(([name]) => names.has(name)),
          );
        await createRun(goal, runtimeSettings, positions);
      } else if (action.kind === "revise_goal")
        await intervene(action.kind, { goal, settings: JSON.parse(settings) });
      else if (action.kind === "inject_message")
        await intervene(action.kind, { recipient, content });
      else if (action.kind === "instruct_agent")
        await intervene(action.kind, {
          agent_id: action.fields?.agent_id,
          content,
        });
      else if (action.kind === "share_note")
        await intervene(action.kind, { content });
      else
        await intervene(action.kind, {
          ...action.fields,
          ...(owner ? { owner } : {}),
        });
      close();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title={title} close={close}>
      <form onSubmit={submit}>
        {["create", "revise_goal"].includes(action.kind) && (
          <>
            {action.kind === "create" && (
              <MissionSamplePicker
                samples={missionSamples}
                value={sampleId}
                onChange={loadSample}
              />
            )}
            <label htmlFor="goal-input">Shared goal</label>
            <textarea
              id="goal-input"
              autoFocus
              required
              rows={4}
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="What should the collective achieve? Include the required deliverables."
            />
            {action.kind === "create" && (
              <div className="form-grid">
                <label>
                  Ollama model
                  <input
                    required
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                  />
                </label>
                <label>
                  Concurrent agents
                  <input
                    required
                    type="number"
                    min={1}
                    max={16}
                    value={concurrency}
                    onChange={(e) => setConcurrency(Number(e.target.value))}
                  />
                </label>
              </div>
            )}
            <details className="advanced">
              <summary>Team & advanced settings · JSON</summary>
              <textarea
                aria-label="Team and advanced settings"
                value={settings}
                onChange={(e) => setSettings(e.target.value)}
                rows={12}
                spellCheck={false}
              />
            </details>
          </>
        )}
        {action.kind === "inject_message" && (
          <>
            <label>
              Recipient
              <select
                value={recipient}
                onChange={(e) => setRecipient(e.target.value)}
              >
                <option value="all">Everyone</option>
                {config.team.map((a) => (
                  <option key={a.name}>{a.name}</option>
                ))}
              </select>
            </label>
            <label>
              Instruction
              <textarea
                aria-label="Instruction"
                required
                maxLength={6000}
                autoFocus
                rows={5}
                value={content}
                onChange={(e) => setContent(e.target.value)}
              />
            </label>
          </>
        )}
        {action.kind === "instruct_agent" && (
          <>
            <p className="help">
              Send a correction or a new constraint to{" "}
              {action.fields?.name || "this specialist"}. They receive it at
              their next work step.
            </p>
            <label>
              Instruction
              <textarea
                aria-label="Specialist instruction"
                required
                maxLength={6000}
                autoFocus
                rows={5}
                value={content}
                onChange={(e) => setContent(e.target.value)}
              />
            </label>
          </>
        )}
        {action.kind === "share_note" && (
          <label>
            Shared note
            <textarea
              aria-label="Shared note"
              required
              maxLength={6000}
              autoFocus
              rows={5}
              value={content}
              onChange={(e) => setContent(e.target.value)}
            />
          </label>
        )}
        {action.kind === "reassign_task" && (
          <label>
            New owner
            <select
              required
              value={owner}
              onChange={(e) => setOwner(e.target.value)}
            >
              <option value="">Select a worker</option>
              {taskOwners(config).map((a) => (
                <option key={a.name}>{a.name}</option>
              ))}
            </select>
          </label>
        )}
        {["retry_task", "reassign_task"].includes(action.kind) && (
          <p className="help">
            This task and its dependents are reopened. Accepted artifacts will
            be reviewed again.
          </p>
        )}
        {error && (
          <p role="alert" className="error-text">
            {error}
          </p>
        )}
        <footer>
          <button type="button" className="button" onClick={close}>
            Cancel
          </button>
          <button className="button primary" disabled={busy}>
            {busy
              ? "Saving…"
              : action.kind === "create"
                ? "Create collective"
                : "Apply intervention"}
          </button>
        </footer>
      </form>
    </Modal>
  );
}
