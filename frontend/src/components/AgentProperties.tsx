import { useEffect, useState } from "react";
import { Trash2 } from "lucide-react";
import type { Agent, TeamConfiguration } from "../types";
import { setToolPermission } from "../graph/editorModel";

export function AgentProperties({
  agent,
  draft,
  tools,
  update,
  remove,
}: {
  agent: Agent | undefined;
  draft: TeamConfiguration;
  tools: string[];
  update: (fields: Partial<Agent>) => boolean;
  remove: () => void;
}) {
  const skills = agent?.skills.join(", ") || "";
  const [skillsText, setSkillsText] = useState(skills);
  const [nameText, setNameText] = useState(agent?.name || "");
  useEffect(() => setNameText(agent?.name || ""), [agent?.name]);
  useEffect(() => setSkillsText(skills), [agent?.name, skills]);
  if (!agent)
    return (
      <div className="editor-properties empty">
        <strong>Select an agent</strong>
        <p>
          Click a node to edit its role, instructions, model, skills, and actual
          tool permissions.
        </p>
      </div>
    );
  const available = [...new Set([...tools, ...agent.tools])];
  return (
    <section className="editor-properties" aria-label="Agent properties">
      <header>
        <div>
          <small>AGENT PROPERTIES</small>
          <h3>{agent.name || "Unnamed agent"}</h3>
        </div>
        <button
          type="button"
          className="icon-button danger"
          onClick={remove}
          disabled={draft.settings.team.length === 1}
          title={
            draft.settings.team.length === 1
              ? "Keep at least one agent in the collective"
              : "Remove this agent"
          }
          aria-label="Remove selected agent"
        >
          <Trash2 size={15} />
        </button>
      </header>
      <label>
        Agent name
        <input
          aria-label="Agent name"
          value={nameText}
          onChange={(event) => setNameText(event.target.value)}
          onBlur={() => {
            if (nameText !== agent.name && !update({ name: nameText }))
              setNameText(agent.name);
          }}
          maxLength={40}
          spellCheck={false}
        />
        <small>Lowercase identifier; renaming also updates paths.</small>
      </label>
      <label>
        Role
        <input
          value={agent.role}
          onChange={(event) => update({ role: event.target.value })}
        />
      </label>
      <label>
        Instructions
        <textarea
          rows={7}
          value={agent.persona}
          onChange={(event) => update({ persona: event.target.value })}
        />
      </label>
      <label>
        Skills
        <input
          aria-label="Skills"
          value={skillsText}
          onChange={(event) => setSkillsText(event.target.value)}
          onBlur={() =>
            update({
              skills: skillsText
                .split(",")
                .map((skill) => skill.trim())
                .filter(Boolean),
            })
          }
          placeholder="research, implementation"
        />
        <small>Comma-separated capabilities used for task planning.</small>
      </label>
      <label>
        Model override
        <input
          value={agent.model || ""}
          onChange={(event) =>
            update({ model: event.target.value || undefined })
          }
          placeholder={`Default: ${draft.settings.model}`}
        />
      </label>
      <fieldset className="editor-tools">
        <legend>Tool availability</legend>
        {available.map((tool) => (
          <label key={tool}>
            <input
              type="checkbox"
              checked={agent.tools.includes(tool)}
              onChange={(event) =>
                update({
                  tools: setToolPermission(
                    agent.tools,
                    tool,
                    event.target.checked,
                  ),
                })
              }
            />
            <span>{tool}</span>
          </label>
        ))}
      </fieldset>
      <p className="editor-permission-note">
        These switches control which tools the agent receives. Web tools
        additionally require web access for the collective.
      </p>
      {available.includes("hire") && (
        <p className="editor-permission-note">
          Hiring also enables monitoring and instructions. Turning either off
          disables hiring.
        </p>
      )}
    </section>
  );
}
