import { expect, test } from "@playwright/test";
import type { Snapshot, Specialist } from "../src/types";

test("an interrupted specialist exposes its effective model and can inherit the parent model only while idle", async ({
  page,
}) => {
  const parent = {
    name: "orchestrator",
    role: "Orchestrator",
    persona: "Delegate and integrate complete work",
    skills: ["planning"],
    tools: ["assemble_files", "hire", "monitor_agents", "instruct_agent"],
  };
  const child: Specialist = {
    name: "outliner",
    id: "outliner-id",
    role: "Paper outliner",
    persona: "Produce a precise outline",
    skills: ["writing"],
    tools: ["read_file"],
    parent: parent.name,
    parent_id: null,
    level: 2,
    status: "interrupted",
    assignment: "Return a complete paper outline",
    output: null,
    input_files: [],
    acceptance_criteria: ["All sections present"],
    model: "GPT-4o",
    error: "Model not found",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
  const snapshot: Snapshot = {
    manifest: {
      run_id: "model-recovery",
      goal: "Write a sourced paper",
      status: "working",
      settings: {
        model: "gemma4:e4b",
        base_url: "http://localhost:11434",
        team: [parent],
        coordinator: parent.name,
        reviewer: parent.name,
        max_concurrency: 1,
        web_enabled: false,
      },
    },
    state: { status: "working", tasks: {} },
    running: true,
    control: null,
    error: null,
    events: [],
    cursor: 10,
    artifacts: [],
    delegation: { agents: { outliner: child }, max_depth: 3 },
  };
  await page.route("**/api/bootstrap", (route) =>
    route.fulfill({ json: { settings: snapshot.manifest.settings } }),
  );
  await page.route("**/api/health*", (route) =>
    route.fulfill({
      json: { connected: true, model: "gemma4:e4b", models: ["gemma4:e4b"] },
    }),
  );
  await page.route("**/api/runs", (route) =>
    route.fulfill({ json: { runs: [snapshot.manifest] } }),
  );
  await page.route(/\/api\/runs\/model-recovery(?:\?.*)?$/, (route) =>
    route.fulfill({ json: snapshot }),
  );
  await page.route("**/api/runs/model-recovery/events?*", (route) =>
    route.fulfill({
      json: {
        events: [],
        cursor: snapshot.cursor,
        running: snapshot.running,
        control: snapshot.control,
        error: snapshot.error,
        delegation: snapshot.delegation,
      },
    }),
  );
  let resetRequests = 0;
  await page.route("**/api/runs/model-recovery/admin", async (route) => {
    expect(route.request().postDataJSON()).toEqual({
      action: "reset_agent_model",
      agent_id: child.id,
    });
    resetRequests++;
    child.model = snapshot.manifest.settings.model;
    snapshot.cursor++;
    await route.fulfill({ json: snapshot });
  });
  await page.goto("/");
  await page.locator('[data-agent="outliner"]').click();
  await expect(page.locator(".specialist-assignment dl")).toContainText(
    "GPT-4o",
  );
  await expect(
    page.getByRole("button", { name: "Use parent model", exact: true }),
  ).toHaveCount(0);
  snapshot.running = false;
  snapshot.cursor++;
  await expect(
    page.getByRole("button", { name: "Use parent model", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Use parent model", exact: true })
    .click();
  await expect(page.locator(".specialist-assignment dl")).toContainText(
    "gemma4:e4b",
  );
  expect(resetRequests).toBe(1);
  await expect(page.locator(".specialist-assignment .error-text")).toHaveText(
    "Model not found",
  );
  child.status = "completed";
  child.error = "";
  snapshot.cursor++;
  await expect(
    page.getByRole("button", { name: "Use parent model", exact: true }),
  ).toHaveCount(0);
});
