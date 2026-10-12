"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ],
  );
const e = escapeHTML;
const nice = (value) => String(value || "").replace(/_/g, " ");
const initials = (name) =>
  name === "admin"
    ? "✦"
    : String(name || "?")
        .slice(0, 2)
        .toUpperCase();
const timeLabel = (at) =>
  at
    ? new Date(at).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : "";
const colors = [
  "#d6d99c",
  "#b1cdbb",
  "#ddb595",
  "#c4b5d4",
  "#9ebecb",
  "#d8be88",
  "#a7c59b",
  "#d2afa7",
];
const app = {
  defaults: null,
  runId: null,
  snapshot: null,
  runs: [],
  events: [],
  seen: new Set(),
  cursor: 0,
  tab: "messages",
  selected: null,
  busy: false,
  polling: false,
  nodeCount: 0,
  metrics: {},
  activeModels: {},
  admin: null,
  debugFilter: "",
  debugKind: "",
  rendering: {},
  artifact: null,
};

async function api(path, data) {
  const response = await fetch(
    path,
    data
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        }
      : {},
  );
  const result = await response.json();
  if (!response.ok)
    throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}
let toastTimer;
function toast(text, error = false) {
  const node = $("#toast");
  node.textContent = text;
  node.classList.toggle("error", error);
  node.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(
    () => node.classList.remove("visible"),
    error ? 6500 : 3300,
  );
}
function settings() {
  return app.snapshot?.manifest.settings || app.defaults || { team: [] };
}
function state() {
  return app.snapshot?.state || {};
}
function team() {
  return settings().team || [];
}
function color(name) {
  return name === "admin"
    ? "#d5e7bc"
    : colors[
        Math.max(
          0,
          team().findIndex((agent) => agent.name === name),
        ) % colors.length
      ];
}
function avatar(name, status = "") {
  return `<span class="agent-avatar" style="background:${color(name)}">${e(initials(name))}${status ? `<i class="${e(status)}"></i>` : ""}</span>`;
}
function tasks() {
  return Object.values(state().tasks || {});
}
function records(key, kind) {
  const records = { ...state()[key] };
  app.events
    .filter((event) => event.kind === kind)
    .forEach((event) => {
      records[event.id] = event;
    });
  return Object.values(records).sort((a, b) =>
    (a.at || "").localeCompare(b.at || ""),
  );
}
function messages() {
  return records("messages", "message");
}
function notes() {
  return records("notes", "note");
}
function canAdmin() {
  const snapshot = app.snapshot;
  return Boolean(
    snapshot &&
    (!snapshot.running ||
      (snapshot.control?.paused &&
        !snapshot.control?.active_nodes?.length &&
        snapshot.control?.waiting)),
  );
}
function agentActivity(name) {
  const control = app.snapshot?.control;
  if (app.snapshot?.running) {
    const node = control?.active_nodes?.find(
      (node) =>
        node.agent === name || state().tasks?.[node.task]?.owner === name,
    );
    if (node || app.activeModels[name])
      return {
        mode: "working",
        text: nice(app.activeModels[name] || node.node || "working"),
      };
    if (control?.mode === "paused")
      return {
        mode: "paused",
        text: control.active_nodes?.length
          ? "Pause requested"
          : "Paused at boundary",
      };
    return { mode: "waiting", text: "Listening to the collective" };
  }
  const assigned = tasks().filter((task) => task.owner === name);
  if (assigned.some((task) => task.status === "failed"))
    return { mode: "waiting", text: "Awaiting a retry" };
  if (state().status === "completed")
    return { mode: "waiting", text: "Mission complete" };
  if (control?.paused && !control.stopped)
    return { mode: "paused", text: "Ready for next step" };
  return {
    mode: "waiting",
    text: app.runId ? "Ready to collaborate" : "Waiting for a mission",
  };
}
function eventDetail(event) {
  const fields = [
    event.agent || event.sender,
    event.task,
    event.node,
    event.purpose,
    event.tool,
    event.path,
    event.status,
  ].filter(Boolean);
  return (
    fields.join(" · ") ||
    (event.content
      ? event.content.slice(0, 100)
      : event.feedback ||
        event.error ||
        event.action ||
        (event.tasks != null ? `${event.tasks} tasks` : "Collective"))
  );
}
function empty(icon, title, description) {
  return `<div class="empty-state"><span class="empty-icon">${icon}</span><strong>${e(title)}</strong><p>${e(description)}</p></div>`;
}

function acceptSnapshot(snapshot) {
  app.snapshot = snapshot;
  app.cursor = Math.max(app.cursor, snapshot.cursor || 0);
  for (const event of snapshot.events || []) {
    if (app.seen.has(event.id)) continue;
    app.seen.add(event.id);
    app.events.push(event);
    if (event.kind === "node_finished") app.nodeCount++;
    if (event.kind === "model_started")
      app.activeModels[event.agent] = event.purpose;
    if (event.kind === "model") {
      delete app.activeModels[event.agent];
      const metric = (app.metrics[event.agent] ||= { calls: 0, tokens: 0 });
      metric.calls++;
      metric.tokens += event.usage?.output_tokens || 0;
    }
    if (
      event.kind === "message" &&
      app.runId &&
      Date.now() - new Date(event.at).getTime() < 5000
    )
      world.addMessage(event);
  }
  if (app.events.length > 2000) app.events = app.events.slice(-2000);
  if (!snapshot.running) app.activeModels = {};
  renderOverview();
  renderActivity();
  renderInspector();
}
function renderOverview() {
  const snapshot = app.snapshot;
  const current = state();
  const control = snapshot?.control;
  $("#mission-title").textContent =
    snapshot?.manifest.goal || "Small team. Shared ambition.";
  $("#mission-title").classList.toggle("has-goal", Boolean(snapshot));
  if (!snapshot)
    $("#mission-title").innerHTML = "Small team.<br>Shared ambition.";
  $("#mission-detail").textContent = snapshot
    ? current.plan?.summary ||
      "Your collective will discuss a plan and work toward this shared goal."
    : "Give your agents a goal. Watch them collaborate, build, and review their work.";
  let status = current.status || "created";
  if (snapshot?.running && control?.mode === "paused")
    status = control.paused && control.waiting ? "paused" : "stepping";
  else if (snapshot?.running) status = "working";
  $("#run-status").textContent = snapshot ? nice(status) : "Ready to begin";
  $("#run-status").className = `status-chip status-${status}`;
  $("#revision").textContent = current.goal_revision
    ? `Revision ${current.goal_revision}`
    : "";
  $("#world-run-id").textContent = app.runId?.slice(-8) || "";
  $("#world-empty").hidden = Boolean(app.runId);
  $("#world-phase").textContent = snapshot
    ? snapshot.running
      ? control?.active_nodes?.length
        ? "Agents at work"
        : control?.paused && control?.waiting
          ? "World paused"
          : control?.steps_pending
            ? "Advancing a node"
            : "Collective running"
      : nice(current.status || "Ready")
    : "Awaiting a mission";
  const completed = tasks().filter(
    (task) => task.status === "completed",
  ).length;
  const total = tasks().length;
  $("#task-progress-value").textContent = total
    ? `${completed} / ${total}`
    : "—";
  $("#task-progress").style.width = total
    ? `${(completed / total) * 100}%`
    : "0%";
  $("#task-progress-label").textContent = total
    ? "Accepted tasks"
    : "Collective progress";
  $("#roster-count").textContent = team().length;
  $("#model-label").textContent =
    `Ollama · ${settings().model || "gemma4:e2b"}`;
  $("#roster").innerHTML = team()
    .map((agent) => {
      const activity = agentActivity(agent.name);
      return `<button class="roster-agent ${app.selected?.type === "agent" && app.selected.id === agent.name ? "selected" : ""}" data-agent="${e(agent.name)}">${avatar(agent.name, activity.mode)}<span class="roster-info"><span class="roster-name">${e(agent.name)}</span><span class="roster-role">${e(agent.role)}</span><span class="roster-status">${e(activity.text)}</span></span></button>`;
    })
    .join("");
  const running = snapshot?.running;
  const paused = control?.mode === "paused";
  $("#run-control-label").textContent = running
    ? paused
      ? "Resume"
      : "Pause"
    : paused && !control.stopped
      ? "Resume"
      : current.status === "completed"
        ? "Complete"
        : "Run collective";
  $("#run-control-icon").textContent = running && !paused ? "Ⅱ" : "▶";
  $("#run-control").disabled =
    app.busy || !snapshot || (current.status === "completed" && !running);
  $("#step-control").disabled =
    app.busy ||
    !snapshot ||
    current.status === "completed" ||
    (running && !(control?.paused && control?.waiting));
  $("#stop-control").disabled = app.busy || !running || control?.stopped;
  $("#retry-control").hidden =
    !snapshot ||
    running ||
    !["incomplete", "interrupted", "stopped"].includes(current.status);
  $("#control-note").textContent = snapshot?.error
    ? snapshot.error
    : running
      ? paused
        ? control.paused && control.waiting
          ? "Paused. Inspect or intervene."
          : control.steps_pending
            ? "Advancing one node…"
            : "Waiting for the current node…"
        : "Progress saved at every node."
      : current.status === "completed"
        ? "All deliverables independently reviewed."
        : "One graph node per step.";
  $("#control-note").title = snapshot?.error || $("#control-note").textContent;
  $("#node-count").textContent =
    `${app.nodeCount} node${app.nodeCount === 1 ? "" : "s"}`;
  $("#message-count").textContent = messages().length;
  $("#task-count").textContent = total;
  $("#artifact-count").textContent = snapshot?.artifacts.length || 0;
  $("#edit-goal").hidden =
    !snapshot || current.status !== "created" || Boolean(current.run_id);
  $("#edit-goal").disabled = !canAdmin();
  $("#activity-action").disabled = !snapshot || !canAdmin();
  $("#activity-action").hidden = !["messages", "memory"].includes(app.tab);
  $("#activity-action").title =
    app.tab === "memory" ? "Share a note" : "Send a message";
  $("#activity-action").setAttribute("aria-label", $("#activity-action").title);
  $("#inspector-footer-text").textContent = canAdmin()
    ? "Interventions available at this checkpoint"
    : snapshot?.running
      ? "Pause at a boundary to intervene"
      : "Live checkpoint inspection";
}
function renderActivity(force = false) {
  const panel = $("#activity-content");
  const scroll = panel.scrollTop;
  const atBottom = panel.scrollHeight - panel.clientHeight - scroll < 30;
  const key = app.tab;
  const contentKey = JSON.stringify(
    key === "messages"
      ? [messages(), state().final_summary]
      : key === "tasks"
        ? tasks()
        : key === "memory"
          ? notes()
          : key === "artifacts"
            ? app.snapshot?.artifacts
            : key === "debug"
              ? [
                  app.events.length,
                  app.events.at(-1)?.id,
                  app.debugFilter,
                  app.debugKind,
                ]
              : state(),
  );
  if (!force && app.rendering[key] === contentKey && panel.dataset.tab === key)
    return;
  const expanded = new Set(
    $$("details[open]", panel).map((node) => node.dataset.path),
  );
  const focus = document.activeElement;
  const searchFocused = panel.contains(focus) && focus.id === "debug-search";
  const selectionStart = searchFocused ? focus.selectionStart : null;
  const selectionEnd = searchFocused ? focus.selectionEnd : null;
  if (!force && panel.contains(focus) && focus.matches("select,textarea"))
    return;
  app.rendering[key] = contentKey;
  panel.dataset.tab = key;
  if (key === "messages") {
    const rows = messages();
    panel.innerHTML =
      (state().final_summary
        ? `<div class="feed-summary"><span class="summary-label">${state().status === "completed" ? "MISSION COMPLETE" : "COLLECTIVE RESULT"}</span>${e(state().final_summary)}</div>`
        : "") +
      (rows.length
        ? rows
            .map(
              (message) =>
                `<div class="message-row">${avatar(message.sender)}<div class="message-body"><div class="message-meta"><strong>${e(message.sender)}</strong><span>→ ${e(message.recipient === "all" ? "everyone" : message.recipient)}</span><time>${timeLabel(message.at)}</time></div><p class="message-text">${e(message.content)}</p></div></div>`,
            )
            .join("")
        : empty(
            "↔",
            "A conversation is about to begin",
            "Your agents’ messages and decisions appear here as they work.",
          ));
  } else if (key === "tasks") {
    panel.innerHTML = tasks().length
      ? tasks()
          .map(
            (task) =>
              `<button class="task-row status-${e(task.status)}" data-task="${e(task.id)}"><span class="task-icon">${task.status === "completed" ? "✓" : task.status === "failed" ? "!" : "◌"}</span><span class="task-title"><strong>${e(task.title)}</strong><small>${e(task.output)}</small></span><span class="task-owner">${e(task.owner)}</span><span class="status-chip">${e(task.status)}</span></button>`,
          )
          .join("")
      : empty(
          "◇",
          "A plan comes first",
          "After the team’s discussion, the coordinator creates tasks with dependencies and acceptance criteria.",
        );
  } else if (key === "artifacts") {
    const artifacts = app.snapshot?.artifacts || [];
    panel.innerHTML = artifacts.length
      ? artifacts
          .map(
            (file) =>
              `<button class="artifact-row" data-artifact="${e(file.path)}"><span class="task-icon">▤</span><span class="task-title"><strong>${e(file.path)}</strong><small>${(file.bytes / 1024).toFixed(1)} KB · text artifact</small></span><span class="muted">↗</span></button>`,
          )
          .join("")
      : empty(
          "▤",
          "Ideas will become artifacts",
          "Open a file to read it, download it, or edit it at a paused checkpoint.",
        );
  } else if (key === "memory") {
    const memory = notes();
    panel.innerHTML = memory.length
      ? memory
          .map(
            (note) =>
              `<div class="memory-row"><strong>${e(note.sender)} · ${timeLabel(note.at)}</strong><p>${e(note.content)}</p></div>`,
          )
          .join("")
      : empty(
          "✧",
          "A shared memory",
          "Reusable lessons, conventions, and decisions stay available to the whole collective.",
        );
  } else if (key === "debug") {
    const kinds = [...new Set(app.events.map((event) => event.kind))].sort();
    const filtered = app.events.filter(
      (event) =>
        (!app.debugKind || event.kind === app.debugKind) &&
        (!app.debugFilter ||
          JSON.stringify(event)
            .toLowerCase()
            .includes(app.debugFilter.toLowerCase())),
    );
    panel.innerHTML =
      `<div class="debug-toolbar"><input id="debug-search" aria-label="Filter debug events" placeholder="Search events, agents, tools…" value="${e(app.debugFilter)}"><select id="debug-kind" aria-label="Filter event kind"><option value="">All events</option>${kinds.map((kind) => `<option ${kind === app.debugKind ? "selected" : ""} value="${e(kind)}">${e(nice(kind))}</option>`).join("")}</select></div>` +
      (filtered.length
        ? filtered
            .slice(-300)
            .reverse()
            .map(
              (event) =>
                `<button class="event-row ${/error|interrupt|retry/.test(event.kind) ? "error" : ""} ${app.selected?.id === event.id ? "selected" : ""}" data-event="${e(event.id)}"><time>${timeLabel(event.at)}</time><span class="event-kind">${e(event.kind)}</span><span class="event-detail">${e(eventDetail(event))}</span><span>↗</span></button>`,
            )
            .join("")
        : empty(
            "⌁",
            "Watch the graph think",
            "Actual node boundaries, model calls, tool results, and reviews will appear here.",
          ));
  } else if (key === "state") {
    panel.innerHTML = `<div class="state-toolbar"><p>Actual LangGraph checkpoint · updates live</p><button id="copy-state" class="text-button">Copy JSON</button></div><div class="json-tree">${jsonTree(state(), "$", 0)}</div>`;
    $$("details", panel).forEach((node) => {
      if (expanded.has(node.dataset.path)) node.open = true;
    });
  }
  panel.scrollTop =
    key === "messages" && atBottom ? panel.scrollHeight : scroll;
  if (searchFocused && key === "debug") {
    $("#debug-search").focus();
    $("#debug-search").setSelectionRange(selectionStart, selectionEnd);
  }
}
function jsonTree(value, path, depth) {
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value);
    const isArray = Array.isArray(value);
    return `<details data-path="${e(path)}" ${depth === 0 ? "open" : ""}><summary><span class="json-key">${e(depth === 0 ? "checkpoint" : path.split(".").at(-1))}</span> <span class="muted">${isArray ? "[" : "{"}${entries.length}${isArray ? "]" : "}"}</span></summary>${entries.map(([key, item]) => jsonTree(item, `${path}.${key}`, depth + 1)).join("")}</details>`;
  }
  return `<div class="json-leaf"><strong>${e(path.split(".").at(-1))}</strong>: ${e(JSON.stringify(value))}</div>`;
}
function select(type, id) {
  app.selected = { type, id };
  app.artifact = null;
  app.inspectorKey = null;
  renderOverview();
  renderInspector(true);
  $(".inspector").classList.add("open");
  if (type === "artifact") loadArtifact(id);
}
function renderInspector(force = false) {
  const panel = $("#inspector-content");
  const selected = app.selected;
  const scroll = panel.scrollTop;
  const key = JSON.stringify([
    selected,
    selected?.type === "agent"
      ? [
          agentActivity(selected.id),
          tasks().filter((task) => task.owner === selected.id),
          app.metrics[selected.id],
          canAdmin(),
        ]
      : selected?.type === "task"
        ? [state().tasks?.[selected.id], canAdmin()]
        : selected?.type === "artifact"
          ? [app.artifact, canAdmin()]
          : selected?.type === "config"
            ? settings()
            : null,
  ]);
  if (
    !force &&
    (app.inspectorKey === key ||
      (panel.contains(document.activeElement) &&
        document.activeElement.matches("input,textarea,select")))
  )
    return;
  app.inspectorKey = key;
  if (!selected) {
    panel.innerHTML = `<div class="inspector-empty"><div class="eye-symbol"></div><h3>A wider perspective.</h3><p>Select an agent, task, or event to see the details. You have a window into every decision.</p></div><div class="inspector-tips"><h4>You’re in control</h4><p>Run or pause the collective <kbd>Space</kbd></p><p>Advance one graph node <kbd>.</kbd></p><p>Drag the world to explore</p><p>Scroll to zoom in or out</p></div><div class="inspector-section"><h4>Intervene with intention</h4><p>Pause at a node boundary to send instructions, share knowledge, reassign work, or edit artifacts. Every intervention is recorded.</p></div>`;
  } else if (selected.type === "agent") {
    const agent = team().find((agent) => agent.name === selected.id);
    if (!agent) return;
    const activity = agentActivity(agent.name);
    const metric = app.metrics[agent.name] || { calls: 0, tokens: 0 };
    const assigned = tasks().filter((task) => task.owner === agent.name);
    const trace = app.events
      .filter(
        (event) => event.agent === agent.name || event.sender === agent.name,
      )
      .slice(-5)
      .reverse();
    panel.innerHTML = `<div class="inspector-profile">${avatar(agent.name, activity.mode)}<div><h3>${e(agent.name)}</h3><p>${e(agent.role)}</p></div></div><span class="status-chip status-${activity.mode}">${e(activity.text)}</span><div class="metric-grid"><div class="metric"><strong>${assigned.length}</strong><span>Assigned tasks</span></div><div class="metric"><strong>${metric.calls}</strong><span>Observed model calls</span></div></div><div class="inspector-actions"><button class="button button-light" data-admin="inject_message" data-recipient="${e(agent.name)}" ${canAdmin() ? "" : "disabled"}>↗ Send instruction</button></div>${!canAdmin() && app.runId ? '<p class="boundary-note">Pause and wait for the current node to finish to intervene.</p>' : ""}<div class="inspector-section"><h4>Persona</h4><p>${e(agent.persona)}</p></div><div class="inspector-section"><h4>Capabilities</h4><div class="tag-list">${agent.skills.map((skill) => `<span class="tag">${e(skill)}</span>`).join("")}</div><div class="tag-list" style="margin-top:9px">${agent.tools.map((tool) => `<span class="tool-tag">${e(tool)}</span>`).join("")}</div></div><div class="inspector-section"><h4>Assigned work</h4>${assigned.length ? assigned.map((task) => `<button class="task-row" data-task="${e(task.id)}"><span class="task-title"><strong>${e(task.title)}</strong><small>${e(task.status)}</small></span><span>↗</span></button>`).join("") : "<p>No tasks assigned yet.</p>"}</div><div class="inspector-section"><h4>Recent activity</h4>${trace.length ? trace.map((event) => `<button class="event-row" data-event="${e(event.id)}"><span class="event-kind">${e(nice(event.kind))}</span><span class="event-detail">${e(eventDetail(event))}</span></button>`).join("") : "<p>Ready to begin.</p>"}</div>`;
  } else if (selected.type === "task") {
    const task = state().tasks?.[selected.id];
    if (!task) return;
    panel.innerHTML = `<h3 class="detail-heading">${e(task.title)}</h3><p class="detail-subtitle">${e(task.id)} · ${e(task.phase)}</p><div style="margin-top:12px"><span class="status-chip status-${e(task.status)}">${e(task.status)}</span></div><div class="inspector-section"><h4>Assignment</h4><p>${e(task.description)}</p><p style="margin-top:10px">Owner: <strong>${e(task.owner)}</strong><br>Attempts: ${task.attempts || 0}<br>Output: ${e(task.output)}</p></div><div class="inspector-section"><h4>Acceptance criteria</h4><ul class="detail-list">${task.acceptance_criteria.map((criterion) => `<li>${e(criterion)}</li>`).join("")}</ul></div><div class="inspector-section"><h4>Dependencies</h4><p>${e(task.dependencies.join(", ") || "No prerequisites")}</p></div>${task.feedback ? `<div class="inspector-section"><h4>Independent review</h4><p>${e(task.feedback)}</p></div>` : ""}<div class="inspector-actions"><button class="button button-ghost" data-admin="reassign_task" data-task-id="${e(task.id)}" ${canAdmin() ? "" : "disabled"}>Reassign</button><button class="button button-light" data-admin="retry_task" data-task-id="${e(task.id)}" ${canAdmin() ? "" : "disabled"}>Reopen task</button></div><p class="boundary-note">Reopening also rebuilds dependent work. The resulting artifacts are reviewed again.</p><button class="button button-ghost" data-artifact="${e(task.output)}" style="width:100%;margin-top:15px">Open artifact ↗</button>`;
  } else if (selected.type === "event") {
    const event = app.events.find((event) => event.id === selected.id);
    if (!event) return;
    panel.innerHTML = `<h3 class="detail-heading">${e(nice(event.kind))}</h3><p class="detail-subtitle">${timeLabel(event.at)} · ${e(event.agent || event.sender || "collective")}</p><div class="inspector-section"><h4>Event payload</h4><pre class="json-view">${e(JSON.stringify(event, null, 2))}</pre></div>`;
  } else if (selected.type === "artifact") {
    const file = app.artifact;
    panel.innerHTML = `<h3 class="detail-heading">${e(selected.id.split("/").at(-1))}</h3><p class="detail-subtitle">${e(selected.id)}</p>${file ? `<div class="artifact-meta"><span>${new TextEncoder().encode(file.content).length} bytes</span><span>${file.content.split(/\s+/).filter(Boolean).length} words</span></div><textarea id="artifact-editor" class="artifact-editor" aria-label="Artifact contents" spellcheck="false" ${canAdmin() ? "" : "readonly"}>${e(file.content)}</textarea><div class="inspector-actions"><button id="save-artifact" class="button button-light" ${canAdmin() ? "" : "disabled"}>Save & reopen</button><button id="download-artifact" class="button button-ghost">Download</button></div><p class="boundary-note">Saving a changed artifact reopens its task and dependent work for review.</p>` : '<p class="boundary-note" style="margin-top:20px">Loading actual artifact…</p>'}`;
  } else if (selected.type === "config") {
    panel.innerHTML = `<h3 class="detail-heading">World configuration</h3><p class="detail-subtitle">${e(app.runId || "Defaults for a new world")}</p><div class="inspector-section"><pre class="json-view">${e(JSON.stringify(settings(), null, 2))}</pre></div>`;
  }
  panel.scrollTop = force ? 0 : scroll;
}
async function loadArtifact(path) {
  const runId = app.runId;
  try {
    const file = await api(
      `/api/runs/${runId}/artifact?path=${encodeURIComponent(path)}`,
    );
    if (
      app.runId === runId &&
      app.selected?.type === "artifact" &&
      app.selected.id === path
    ) {
      app.artifact = file;
      renderInspector(true);
    }
  } catch (error) {
    toast(error.message, true);
  }
}
async function refreshRuns() {
  try {
    const result = await api("/api/runs");
    app.runs = result.runs;
    $("#run-select").innerHTML =
      `<option value="">${app.runs.length ? "Choose a world" : "Create your first world"}</option>` +
      app.runs
        .map(
          (run) =>
            `<option value="${e(run.run_id)}">${e(run.goal.slice(0, 47))}${run.goal.length > 47 ? "…" : ""}</option>`,
        )
        .join("");
    $("#run-select").value = app.runId || "";
  } catch (error) {
    toast(error.message, true);
  }
}
async function selectRun(runId) {
  app.runId = runId || null;
  app.snapshot = null;
  app.selected = null;
  app.artifact = null;
  app.events = [];
  app.cursor = 0;
  app.seen.clear();
  app.metrics = {};
  app.activeModels = {};
  app.nodeCount = 0;
  app.rendering = {};
  app.inspectorKey = null;
  world.messages = [];
  if (!app.runId) {
    renderOverview();
    renderActivity(true);
    renderInspector(true);
    return;
  }
  try {
    const result = await api(`/api/runs/${runId}`);
    if (app.runId === runId) {
      acceptSnapshot(result);
      refreshHealth();
    }
  } catch (error) {
    toast(error.message, true);
  }
}
async function poll() {
  if (app.polling || !app.runId) return;
  const runId = app.runId;
  app.polling = true;
  try {
    const result = await api(`/api/runs/${runId}?cursor=${app.cursor}`);
    if (runId === app.runId) acceptSnapshot(result);
  } catch (error) {
    $("#inspector-footer-text").textContent =
      `Connection interrupted: ${error.message}`;
  } finally {
    app.polling = false;
  }
}
async function sendControl(action, retry = false) {
  if (!app.runId || app.busy) return;
  const runId = app.runId;
  app.busy = true;
  renderOverview();
  try {
    const result = await api(`/api/runs/${runId}/control`, { action, retry });
    if (runId === app.runId) acceptSnapshot(result);
  } catch (error) {
    toast(error.message, true);
  } finally {
    app.busy = false;
    renderOverview();
  }
}
function showCreate() {
  if (!app.defaults) {
    toast("The control room is still connecting.");
    return;
  }
  $("#goal-input").value = "";
  $("#model-input").value = app.defaults.model;
  $("#concurrency-input").value = app.defaults.max_concurrency;
  $("#web-input").checked = app.defaults.web_enabled;
  $("#settings-input").value = JSON.stringify(app.defaults, null, 2);
  $("#create-error").textContent = "";
  $("#create-dialog").showModal();
  $("#goal-input").focus();
}
function optionList(values, current) {
  return values
    .map(
      (value) =>
        `<option ${value === current ? "selected" : ""} value="${e(value)}">${e(value === "all" ? "Everyone" : value)}</option>`,
    )
    .join("");
}
function showAdmin(action, detail = {}) {
  if (!app.runId || !canAdmin()) {
    toast("Pause and wait for the current node to finish to intervene.");
    return;
  }
  app.admin = { action, ...detail };
  $("#admin-error").textContent = "";
  $("#admin-submit").textContent = "Apply intervention";
  let title, description, fields;
  if (action === "inject_message") {
    title = "Send an instruction";
    description =
      "Your message enters the selected agent’s inbox and is included in their next turn. The intervention is recorded.";
    fields = `<label for="admin-recipient">Recipient</label><select id="admin-recipient">${optionList(["all", ...team().map((agent) => agent.name)], detail.recipient || "all")}</select><label for="admin-content">Instruction</label><textarea id="admin-content" rows="5" required maxlength="6000" placeholder="Give a concrete instruction or helpful context."></textarea>`;
  } else if (action === "share_note") {
    title = "Share knowledge";
    description =
      "Add a reusable fact, decision, or convention to the collective’s shared memory.";
    fields = `<label for="admin-content">Shared note</label><textarea id="admin-content" rows="5" required maxlength="6000" placeholder="What should every agent remember?"></textarea>`;
  } else if (action === "reassign_task") {
    title = "Reassign this task";
    description =
      "Transfer ownership to an agent with file-writing access. This task and dependent work will be reopened.";
    fields = `<label for="admin-owner">New owner</label><select id="admin-owner">${optionList(
      team()
        .filter((agent) => agent.tools.includes("write_file"))
        .map((agent) => agent.name),
      state().tasks[detail.task_id].owner,
    )}</select>`;
  } else if (action === "retry_task") {
    title = "Reopen this task";
    description =
      "Give this task a fresh attempt budget. Its dependent work will also be reopened so the final result stays consistent.";
    fields = `<p class="boundary-note">${e(state().tasks[detail.task_id].title)}</p>`;
    $("#admin-submit").textContent = "Reopen task";
  } else if (action === "revise_goal") {
    title = "Revise your mission";
    description =
      "Update the goal and configuration before the first graph node runs. After work starts, use messages and shared notes to steer the agents.";
    fields = `<label for="admin-goal">Shared goal</label><textarea id="admin-goal" rows="4" required>${e(app.snapshot.manifest.goal)}</textarea><label for="admin-settings">Team & configuration (JSON)</label><textarea id="admin-settings" class="mono" rows="12" spellcheck="false">${e(JSON.stringify(settings(), null, 2))}</textarea>`;
  }
  $("#admin-title").textContent = title;
  $("#admin-description").textContent = description;
  $("#admin-fields").innerHTML = fields;
  $("#admin-dialog").showModal();
  $("#admin-content, #admin-owner, #admin-goal")?.focus();
}
function download(name, content, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

document.addEventListener("click", (event) => {
  const node = event.target.closest("button");
  if (!node || node.disabled) return;
  if (node.dataset.agent) select("agent", node.dataset.agent);
  if (node.dataset.task) select("task", node.dataset.task);
  if (node.dataset.event) select("event", node.dataset.event);
  if (node.dataset.artifact) select("artifact", node.dataset.artifact);
  if (node.dataset.admin)
    showAdmin(node.dataset.admin, {
      recipient: node.dataset.recipient,
      task_id: node.dataset.taskId,
    });
  if (node.dataset.tab) {
    app.tab = node.dataset.tab;
    $$(".tab").forEach((tab) => tab.classList.toggle("active", tab === node));
    renderOverview();
    renderActivity(true);
    $("#activity-content").scrollTop =
      app.tab === "messages" ? $("#activity-content").scrollHeight : 0;
  }
  if (node.classList.contains("dialog-close")) node.closest("dialog").close();
  if (node.id === "copy-state")
    navigator.clipboard
      .writeText(JSON.stringify(state(), null, 2))
      .then(() => toast("Checkpoint JSON copied."))
      .catch(() =>
        download(
          "state.json",
          JSON.stringify(state(), null, 2),
          "application/json",
        ),
      );
  if (node.id === "save-artifact") saveArtifact();
  if (node.id === "download-artifact" && app.artifact)
    download(app.artifact.path.split("/").at(-1), $("#artifact-editor").value);
});
$("#new-run").addEventListener("click", showCreate);
$("#world-create").addEventListener("click", showCreate);
$("#refresh-runs").addEventListener("click", refreshRuns);
$("#run-select").addEventListener("change", (event) =>
  selectRun(event.target.value),
);
$("#show-config").addEventListener("click", () => select("config", "settings"));
$("#run-control").addEventListener("click", () =>
  sendControl(
    app.snapshot?.running
      ? app.snapshot.control?.mode === "paused"
        ? "resume"
        : "pause"
      : "resume",
  ),
);
$("#step-control").addEventListener("click", () => sendControl("step"));
$("#stop-control").addEventListener("click", () => sendControl("stop"));
$("#retry-control").addEventListener("click", () =>
  sendControl("resume", true),
);
$("#edit-goal").addEventListener("click", () => showAdmin("revise_goal"));
$("#activity-action").addEventListener("click", () =>
  showAdmin(app.tab === "memory" ? "share_note" : "inject_message"),
);
$("#inspector-close").addEventListener("click", () =>
  $(".inspector").classList.remove("open"),
);
$("#activity-content").addEventListener("input", (event) => {
  if (event.target.id === "debug-search") {
    app.debugFilter = event.target.value;
    const pos = event.target.selectionStart;
    renderActivity(true);
    $("#debug-search").focus();
    $("#debug-search").setSelectionRange(pos, pos);
  }
});
$("#activity-content").addEventListener("change", (event) => {
  if (event.target.id === "debug-kind") {
    app.debugKind = event.target.value;
    renderActivity(true);
  }
});
$("#create-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("#create-error").textContent = "";
  $("#create-submit").disabled = true;
  try {
    const config = JSON.parse($("#settings-input").value);
    config.model = $("#model-input").value;
    config.max_concurrency = Number($("#concurrency-input").value);
    config.web_enabled = $("#web-input").checked;
    const result = await api("/api/runs", {
      goal: $("#goal-input").value,
      settings: config,
    });
    $("#create-dialog").close();
    await selectRun(result.manifest.run_id);
    await refreshRuns();
    toast("Your world is ready. Run it or advance one node at a time.");
  } catch (error) {
    $("#create-error").textContent = error.message;
  } finally {
    $("#create-submit").disabled = false;
  }
});
$("#admin-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("#admin-error").textContent = "";
  $("#admin-submit").disabled = true;
  try {
    const fields = { action: app.admin.action };
    if (app.admin.action === "inject_message") {
      fields.recipient = $("#admin-recipient").value;
      fields.content = $("#admin-content").value;
    }
    if (app.admin.action === "share_note")
      fields.content = $("#admin-content").value;
    if (["retry_task", "reassign_task"].includes(app.admin.action))
      fields.task_id = app.admin.task_id;
    if (app.admin.action === "reassign_task")
      fields.owner = $("#admin-owner").value;
    if (app.admin.action === "revise_goal") {
      fields.goal = $("#admin-goal").value;
      fields.settings = JSON.parse($("#admin-settings").value);
    }
    const result = await api(`/api/runs/${app.runId}/admin`, fields);
    acceptSnapshot(result);
    $("#admin-dialog").close();
    await refreshRuns();
    toast("Intervention saved. Resume or step to continue.");
  } catch (error) {
    $("#admin-error").textContent = error.message;
  } finally {
    $("#admin-submit").disabled = false;
  }
});
async function saveArtifact() {
  $("#save-artifact").disabled = true;
  try {
    const content = $("#artifact-editor").value;
    const result = await api(`/api/runs/${app.runId}/admin`, {
      action: "replace_artifact",
      path: app.artifact.path,
      content,
    });
    app.artifact.content = content;
    acceptSnapshot(result);
    renderInspector(true);
    toast("Artifact saved. Task and dependent work reopened.");
  } catch (error) {
    toast(error.message, true);
    $("#save-artifact").disabled = !canAdmin();
  }
}
document.addEventListener("keydown", (event) => {
  if (
    event.target.matches("input,textarea,select") ||
    $$("dialog[open]").length ||
    event.ctrlKey ||
    event.altKey ||
    event.metaKey
  )
    return;
  if (event.code === "Space") {
    event.preventDefault();
    if (!$("#run-control").disabled) $("#run-control").click();
  }
  if (event.key === "." && !$("#step-control").disabled) {
    event.preventDefault();
    $("#step-control").click();
  }
  if (event.key === "Escape") $(".inspector").classList.remove("open");
});

// The world is an illustration of the configured collective. Activity, messages,
// task ownership and pauses always come from the server; the camera is local.
const world = {
  canvas: $("#world-canvas"),
  ctx: $("#world-canvas").getContext("2d"),
  width: 0,
  height: 0,
  camera: { zoom: 1, x: 0, y: 0 },
  hits: [],
  messages: [],
  drag: null,
  moved: false,
  reducedMotion: matchMedia("(prefers-reduced-motion: reduce)").matches,
  points: [
    { x: -94, y: -18 },
    { x: 73, y: -48 },
    { x: 140, y: 65 },
    { x: -88, y: 108 },
  ],
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    this.canvas.width = rect.width * ratio;
    this.canvas.height = rect.height * ratio;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  },
  fit() {
    this.camera = { zoom: 1, x: 0, y: 0 };
  },
  addMessage(event) {
    this.messages.push({ ...event, started: performance.now() });
    if (this.messages.length > 15) this.messages.shift();
  },
  scale() {
    return (
      Math.min(this.width / 610, this.height / 320, 1.7) * this.camera.zoom
    );
  },
  project(x, y) {
    const scale = this.scale();
    return {
      x: this.width / 2 + this.camera.x + x * scale,
      y: this.height / 2 + this.camera.y + (y + 3) * scale,
    };
  },
  position(index) {
    return this.points[index % 4]
      ? {
          x:
            this.points[index % 4].x +
            (index >= 4 ? 36 * Math.floor(index / 4) : 0),
          y:
            this.points[index % 4].y +
            (index >= 4 ? 30 * Math.floor(index / 4) : 0),
        }
      : { x: 0, y: 0 };
  },
  poly(points, fill, stroke) {
    const ctx = this.ctx;
    ctx.beginPath();
    points.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
    ctx.closePath();
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.stroke();
    }
  },
  ellipse(x, y, rx, ry, fill) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
  },
  isoBox(x, y, w, d, h, top, left, right) {
    this.poly(
      [
        [x, y],
        [x + w, y + w * 0.45],
        [x + w - d, y + (w + d) * 0.45],
        [x - d, y + d * 0.45],
      ],
      top,
    );
    this.poly(
      [
        [x - d, y + d * 0.45],
        [x + w - d, y + (w + d) * 0.45],
        [x + w - d, y + (w + d) * 0.45 + h],
        [x - d, y + d * 0.45 + h],
      ],
      left,
    );
    this.poly(
      [
        [x + w - d, y + (w + d) * 0.45],
        [x + w, y + w * 0.45],
        [x + w, y + w * 0.45 + h],
        [x + w - d, y + (w + d) * 0.45 + h],
      ],
      right,
    );
  },
  tree(x, y, size = 1) {
    const ctx = this.ctx;
    this.ellipse(x + 4, y + 4, 14 * size, 6 * size, "#5a72551b");
    ctx.fillStyle = "#7b8051";
    ctx.fillRect(x - 2 * size, y - 12 * size, 4 * size, 16 * size);
    this.ellipse(x, y - 18 * size, 11 * size, 15 * size, "#78905e");
    this.ellipse(x - 4 * size, y - 22 * size, 10 * size, 12 * size, "#94a974");
    this.ellipse(x + 4 * size, y - 15 * size, 8 * size, 10 * size, "#7f985f");
  },
  building(x, y, w, d, h, style) {
    const ctx = this.ctx;
    this.ellipse(x + 7, y + 31, w * 0.85, 15, "#56634619");
    this.isoBox(x, y - h, w, d, h, "#e1d9b9", "#c7c4a3", "#aea98b");
    if (style === "library") {
      this.isoBox(
        x - 6,
        y - h - 8,
        w + 12,
        d + 7,
        8,
        "#7c9476",
        "#63795c",
        "#546b50",
      );
      for (let i = 0; i < 3; i++) {
        ctx.fillStyle = "#738968";
        ctx.fillRect(x - d + 9 + i * 13, y + d * 0.45 - h + 14, 6, 13);
        ctx.fillStyle = "#a9ba97";
        ctx.fillRect(x - d + 10 + i * 13, y + d * 0.45 - h + 15, 4, 3);
      }
      ctx.strokeStyle = "#9f9e7e";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x - d, y + d * 0.45 - 3);
      ctx.lineTo(x + w - d, y + (w + d) * 0.45 - 3);
      ctx.stroke();
    } else if (style === "atelier") {
      this.poly(
        [
          [x - 5, y - h - 5],
          [x + w + 5, y + w * 0.45 - h - 5],
          [x + w - d + 3, y + (w + d) * 0.45 - h - 4],
          [x - d - 3, y + d * 0.45 - h - 4],
        ],
        "#c78f72",
      );
      this.poly(
        [
          [x - 5, y - h - 5],
          [x + 15, y - h - 26],
          [x + w + 5, y + w * 0.45 - h - 5],
        ],
        "#d7a085",
      );
      this.poly(
        [
          [x + 15, y - h - 26],
          [x + 15 - d, y + d * 0.45 - h - 25],
          [x - d - 3, y + d * 0.45 - h - 4],
          [x + w - d + 3, y + (w + d) * 0.45 - h - 4],
          [x + w + 5, y + w * 0.45 - h - 5],
        ],
        "#b88268",
      );
      this.isoBox(
        x + 18,
        y - h - 17,
        7,
        7,
        12,
        "#baa488",
        "#aa927d",
        "#917e67",
      );
      ctx.fillStyle = "#7f8d76";
      ctx.fillRect(x - d + 10, y + d * 0.45 - h + 11, 12, 16);
      ctx.fillStyle = "#c0c4a4";
      ctx.fillRect(x - d + 12, y + d * 0.45 - h + 13, 8, 12);
    } else if (style === "observatory") {
      this.ellipse(x + 8, y - h + 15, 30, 18, "#99aa91");
      ctx.beginPath();
      ctx.ellipse(x + 8, y - h + 12, 30, 26, 0, Math.PI, 0);
      ctx.fillStyle = "#adbda0";
      ctx.fill();
      ctx.strokeStyle = "#859b7c";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.ellipse(x + 8, y - h + 12, 15, 26, 0, Math.PI, 0);
      ctx.stroke();
      this.poly(
        [
          [x + 1, y - h - 6],
          [x + 15, y - h - 18],
          [x + 36, y - h - 2],
          [x + 21, y - h + 10],
        ],
        "#728c78",
      );
      this.poly(
        [
          [x + 21, y - h + 10],
          [x + 36, y - h - 2],
          [x + 39, y - h + 3],
          [x + 25, y - h + 15],
        ],
        "#5d7863",
      );
      ctx.fillStyle = "#87937c";
      ctx.fillRect(x - d + 10, y + d * 0.45 - h + 21, 8, 11);
    } else {
      this.isoBox(
        x - 4,
        y - h - 4,
        w + 7,
        d + 7,
        5,
        "#cebb95",
        "#baa982",
        "#a79577",
      );
      for (let i = 0; i < 2; i++) {
        ctx.fillStyle = "#8e9b83";
        ctx.fillRect(x - d + 10 + i * 13, y + d * 0.45 - h + 13, 7, 12);
      }
    }
  },
  draw(now) {
    const ctx = this.ctx;
    if (!this.width || !this.height) return;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.save();
    ctx.translate(
      this.width / 2 + this.camera.x,
      this.height / 2 + this.camera.y,
    );
    const scale = this.scale();
    ctx.scale(scale, scale);
    ctx.translate(0, 3);
    // Quiet cartographic marks and an elevated, isometric commons.
    ctx.strokeStyle = "#a8b79c23";
    ctx.lineWidth = 0.7;
    for (let i = -480; i < 500; i += 34) {
      ctx.beginPath();
      ctx.moveTo(i - 250, -260);
      ctx.lineTo(i + 400, 280);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(i + 250, -260);
      ctx.lineTo(i - 400, 280);
      ctx.stroke();
    }
    this.ellipse(0, 80, 227, 83, "#7b8c6313");
    this.ellipse(0, 76, 195, 69, "#6d7f5716");
    const edge = [
      [-254, 17],
      [-57, -123],
      [239, -1],
      [76, 154],
    ];
    this.poly(
      [
        [-254, 17],
        [76, 154],
        [76, 173],
        [-254, 35],
      ],
      "#b0bda0",
    );
    this.poly(
      [
        [76, 154],
        [239, -1],
        [239, 17],
        [76, 173],
      ],
      "#a1b295",
    );
    this.poly(edge, "#c3cdb1", "#b6c3a8");
    this.poly(
      [
        [-237, 18],
        [-53, -112],
        [221, 0],
        [75, 141],
      ],
      "#b7c49f",
    );
    this.poly(
      [
        [-224, 12],
        [-150, -35],
        [-111, -21],
        [-170, 27],
      ],
      "#aabd91",
    );
    this.poly(
      [
        [107, 44],
        [169, 16],
        [206, 31],
        [148, 67],
      ],
      "#afc099",
    );
    this.poly(
      [
        [-71, 114],
        [-15, 81],
        [34, 103],
        [-16, 135],
      ],
      "#adbF95",
    );
    // Paved paths connect the agent stations around a common fountain.
    ctx.lineCap = "round";
    ctx.lineWidth = 16;
    ctx.strokeStyle = "#dce0c2";
    for (const point of this.points) {
      ctx.beginPath();
      ctx.moveTo(0, 36);
      ctx.lineTo(point.x, point.y + 18);
      ctx.stroke();
    }
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#c4c9a8";
    for (const point of this.points) {
      ctx.beginPath();
      ctx.moveTo(0, 36);
      ctx.lineTo(point.x, point.y + 18);
      ctx.stroke();
    }
    this.ellipse(0, 36, 45, 23, "#dddfc1");
    this.ellipse(0, 36, 30, 16, "#c4ceb6");
    this.ellipse(0, 34, 24, 12, "#9fbdb3");
    this.ellipse(0, 34, 18, 8, "#b0cec1");
    ctx.fillStyle = "#d6dfbd";
    ctx.fillRect(-2, 19, 4, 16);
    this.ellipse(0, 19, 8, 4, "#e6e6c8");
    const plants = [
      [-187, 7, 1.1],
      [-164, -25, 0.9],
      [-144, -53, 0.8],
      [-30, -89, 0.95],
      [131, -7, 0.9],
      [164, 17, 1.1],
      [122, 114, 0.9],
      [-95, 105, 0.95],
      [-139, 70, 0.9],
      [44, 123, 0.6],
    ];
    plants.forEach((p) => this.tree(...p));
    // Shared gathering pavilion, research library, workshop, and observatory.
    this.building(-117, -37, 53, 44, 33, "pavilion");
    this.building(32, -68, 50, 38, 39, "library");
    this.building(120, 22, 46, 36, 34, "atelier");
    this.building(-104, 61, 43, 35, 31, "observatory");
    this.isoBox(-42, 75, 17, 11, 5, "#cbb28b", "#b09876", "#957f60");
    this.isoBox(93, -27, 15, 9, 5, "#cbb28b", "#b09876", "#957f60");
    ctx.fillStyle = "#73865c";
    for (const [x, y] of [
      [-204, 31],
      [187, 8],
      [-100, -73],
      [165, 45],
      [-136, 96],
    ]) {
      ctx.fillRect(x, y, 2, 4);
      ctx.fillRect(x + 4, y - 2, 2, 5);
    }
    ctx.font = "600 6px " + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = "center";
    ctx.fillStyle = "#6e8061";
    ctx.letterSpacing = "1px";
    ctx.fillText("COMMONS", -98, -68);
    ctx.fillText("RESEARCH", 54, -94);
    ctx.fillText("ATELIER", 142, 2);
    ctx.fillText("OBSERVATORY", -85, 43);
    // Broadcast links fade after actual message events arrive.
    if (!this.reducedMotion) {
      this.messages = this.messages.filter(
        (message) => now - message.started < 4000,
      );
      for (const message of this.messages) {
        const sender = team().findIndex((a) => a.name === message.sender);
        if (sender < 0) continue;
        const from = this.position(sender);
        const recipients =
          message.recipient === "all"
            ? team()
                .map((_, i) => i)
                .filter((i) => i !== sender)
            : [team().findIndex((a) => a.name === message.recipient)];
        for (const index of recipients) {
          if (index < 0) continue;
          const to = this.position(index);
          const progress = Math.min(1, (now - message.started) / 1800);
          ctx.globalAlpha = Math.max(0, 1 - (now - message.started) / 4000);
          ctx.strokeStyle = "#f7f6da";
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 4]);
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.quadraticCurveTo(
            (from.x + to.x) / 2,
            (from.y + to.y) / 2 - 40,
            to.x,
            to.y,
          );
          ctx.stroke();
          ctx.setLineDash([]);
          const t = progress;
          const x =
            (1 - t) ** 2 * from.x +
            (2 * (1 - t) * t * (from.x + to.x)) / 2 +
            t * t * to.x;
          const y =
            (1 - t) ** 2 * from.y +
            2 * (1 - t) * t * ((from.y + to.y) / 2 - 40) +
            t * t * to.y;
          this.ellipse(x, y, 3, 3, "#f8f6d5");
        }
        ctx.globalAlpha = 1;
      }
    }
    this.hits = [];
    team().forEach((agent, index) => {
      const p = this.position(index);
      const activity = agentActivity(agent.name);
      const selected =
        app.selected?.type === "agent" && app.selected.id === agent.name;
      const bounce =
        activity.mode === "working" && !this.reducedMotion
          ? Math.sin(now / 450 + index) * 1.3
          : 0;
      this.ellipse(p.x, p.y + 16, 13, 5, "#59644525");
      if (activity.mode === "working") {
        const pulse = this.reducedMotion ? 1 : (Math.sin(now / 420) + 1) / 2;
        ctx.strokeStyle = "#79a16a";
        ctx.globalAlpha = 0.35 + pulse * 0.25;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.ellipse(
          p.x,
          p.y + 13,
          17 + pulse * 3,
          8 + pulse * 2,
          0,
          0,
          Math.PI * 2,
        );
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      if (selected) {
        ctx.strokeStyle = "#345b3d";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.ellipse(p.x, p.y + 13, 21, 10, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.fillStyle = "#596954";
      ctx.fillRect(p.x - 5, p.y + 6 + bounce, 4, 10);
      ctx.fillRect(p.x + 1, p.y + 6 + bounce, 4, 10);
      ctx.beginPath();
      ctx.roundRect(p.x - 9, p.y - 4 + bounce, 18, 16, 5);
      ctx.fillStyle = color(agent.name);
      ctx.fill();
      this.ellipse(p.x, p.y - 9 + bounce, 7, 8, "#e7c7a1");
      this.ellipse(
        p.x,
        p.y - 13 + bounce,
        7,
        4,
        index % 2 ? "#6c6652" : "#746e50",
      );
      ctx.fillStyle = "#596248";
      ctx.fillRect(p.x - 3, p.y - 10 + bounce, 1, 1);
      ctx.fillRect(p.x + 2, p.y - 10 + bounce, 1, 1);
      const label = agent.name;
      ctx.font = "500 7px " + getComputedStyle(document.body).fontFamily;
      const textWidth = ctx.measureText(label).width;
      ctx.beginPath();
      ctx.roundRect(p.x - textWidth / 2 - 8, p.y + 23, textWidth + 16, 15, 4);
      ctx.fillStyle = selected ? "#344d36" : "#eff0dadd";
      ctx.fill();
      ctx.fillStyle = selected ? "#e3edcf" : "#607350";
      ctx.fillText(label, p.x, p.y + 33);
      if (activity.mode === "working") {
        ctx.beginPath();
        ctx.roundRect(p.x + 10, p.y - 23 + bounce, 23, 14, 5);
        ctx.fillStyle = "#eff1db";
        ctx.fill();
        ctx.fillStyle = "#6f9763";
        for (let dot = 0; dot < 3; dot++) {
          const dotBounce = this.reducedMotion
            ? 0
            : Math.sin(now / 230 - dot * 0.9) * 1.2;
          this.ellipse(
            p.x + 15 + dot * 6,
            p.y - 16 + bounce + dotBounce,
            1.4,
            1.4,
            "#6f9763",
          );
        }
      } else if (activity.mode === "paused") {
        ctx.fillStyle = "#c4a669";
        ctx.fillRect(p.x + 12, p.y - 17, 2, 6);
        ctx.fillRect(p.x + 16, p.y - 17, 2, 6);
      }
      const screen = this.project(p.x, p.y);
      this.hits.push({
        x: screen.x,
        y: screen.y,
        r: 24 * scale,
        agent: agent.name,
      });
    });
    ctx.restore();
  },
};
new ResizeObserver(() => world.resize()).observe(world.canvas);
let lastFrame = 0;
function animate(now) {
  if (now - lastFrame > 33 || (world.reducedMotion && now - lastFrame > 500)) {
    world.draw(now);
    lastFrame = now;
  }
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
world.canvas.addEventListener("pointerdown", (event) => {
  world.drag = {
    x: event.clientX,
    y: event.clientY,
    cx: world.camera.x,
    cy: world.camera.y,
  };
  world.moved = false;
  world.canvas.setPointerCapture(event.pointerId);
});
world.canvas.addEventListener("pointermove", (event) => {
  if (!world.drag) return;
  const dx = event.clientX - world.drag.x,
    dy = event.clientY - world.drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 4) world.moved = true;
  world.camera.x = world.drag.cx + dx;
  world.camera.y = world.drag.cy + dy;
});
world.canvas.addEventListener("pointerup", (event) => {
  if (!world.moved) {
    const rect = world.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left,
      y = event.clientY - rect.top;
    const hit = world.hits.find(
      (hit) => Math.hypot(x - hit.x, y - hit.y) < hit.r,
    );
    if (hit) select("agent", hit.agent);
  }
  world.drag = null;
});
world.canvas.addEventListener("pointercancel", () => {
  world.drag = null;
});
world.canvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    world.camera.zoom = Math.max(
      0.5,
      Math.min(3, world.camera.zoom * Math.exp(-event.deltaY * 0.001)),
    );
  },
  { passive: false },
);
$("#camera-fit").addEventListener("click", () => world.fit());
$("#zoom-in").addEventListener("click", () => {
  world.camera.zoom = Math.min(3, world.camera.zoom * 1.2);
});
$("#zoom-out").addEventListener("click", () => {
  world.camera.zoom = Math.max(0.5, world.camera.zoom / 1.2);
});
$("#world-fullscreen").addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else
    $("#world-viewport")
      .requestFullscreen()
      .catch((error) => toast(error.message, true));
});

async function bootstrap() {
  try {
    const data = await api("/api/bootstrap");
    app.defaults = data.settings;
    await refreshRuns();
    renderOverview();
    renderActivity(true);
    renderInspector(true);
    if (app.runs.length) await selectRun(app.runs[0].run_id);
  } catch (error) {
    toast(`Unable to connect: ${error.message}`, true);
    $("#connection span").textContent = "Server unavailable";
    $("#connection").classList.add("offline");
  }
  await refreshHealth();
  setInterval(poll, 800);
  setInterval(refreshRuns, 15000);
  setInterval(refreshHealth, 30000);
}
async function refreshHealth() {
  const runId = app.runId;
  try {
    const health = await api(
      "/api/health" + (runId ? `?run=${encodeURIComponent(runId)}` : ""),
    );
    if (runId !== app.runId) return;
    const ready = health.connected && health.models.includes(health.model);
    $("#connection span").textContent = ready
      ? "Ollama connected"
      : health.connected
        ? "Model unavailable"
        : "Ollama offline";
    $("#connection").classList.toggle("offline", !ready);
    $("#connection").title =
      health.error || `Available models: ${(health.models || []).join(", ")}`;
  } catch {
    $("#connection span").textContent = "Connection unavailable";
    $("#connection").classList.add("offline");
  }
}
bootstrap();
