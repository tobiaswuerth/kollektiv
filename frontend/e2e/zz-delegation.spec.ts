import { expect, test } from "@playwright/test";
import type { Settings, Snapshot } from "../src/types";

test("real specialist hires, grandchildren, live guidance, parent reports and retirement history", async ({
  page,
  request,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const { settings } = (await (await request.get("/api/bootstrap")).json()) as {
    settings: Settings;
  };
  const builder = settings.team.find((agent) => agent.name === "builder")!;
  builder.tools = [
    ...new Set([
      ...builder.tools,
      "hire",
      "fire",
      "monitor_agents",
      "instruct_agent",
    ]),
  ];
  builder.skills.push("delegation", "supervision");
  settings.max_concurrency = 3;
  settings.discussion_rounds = 1;
  settings.max_steps = 100;
  const response = await request.post("/api/runs", {
    data: {
      goal: "DELEGATION_BROWSER Supervise a real specialist family, then create summary.md.",
      settings,
    },
  });
  expect(response.ok()).toBe(true);
  const created = (await response.json()) as Snapshot;
  await page.goto("/");
  await page.locator("#run-select").selectOption(created.manifest.run_id);
  await expect(page.locator(".person-card")).toHaveCount(4);
  await page.getByRole("button", { name: "Edit team", exact: true }).click();
  const editor = page.getByRole("dialog", {
    name: "Edit collective",
    exact: true,
  });
  await editor.getByRole("checkbox", { name: "hire", exact: true }).check();
  await expect(
    editor.getByRole("checkbox", { name: "monitor_agents", exact: true }),
  ).toBeChecked();
  await expect(
    editor.getByRole("checkbox", { name: "instruct_agent", exact: true }),
  ).toBeChecked();
  await editor
    .getByRole("checkbox", { name: "instruct_agent", exact: true })
    .uncheck();
  await expect(
    editor.getByRole("checkbox", { name: "hire", exact: true }),
  ).not.toBeChecked();
  await editor
    .getByRole("button", { name: "Close editor", exact: true })
    .click();
  await page.locator("#run-control").click();
  const scout = page.locator('[data-agent="browser_scout"]');
  const leaf = page.locator('[data-agent="browser_leaf"]');
  await expect(scout).toHaveAttribute("data-level", "2");
  await expect(scout).toHaveAttribute("data-parent", "builder");
  await expect(leaf).toHaveAttribute("data-level", "3");
  await expect(leaf).toHaveAttribute("data-parent", "browser_scout");
  await expect(page.locator(".person-card")).toHaveCount(6);
  await expect(
    page.locator('[data-flow-path="builder→browser_scout"]'),
  ).toBeAttached();
  await expect(
    page.locator('[data-flow-path="browser_scout→browser_leaf"]'),
  ).toBeAttached();
  await leaf.click();
  await expect(page.locator(".inspector h3")).toHaveText("browser_leaf");
  await expect(page.locator(".specialist-assignment h4")).toHaveText(
    "LEVEL 3 OF 3",
  );
  await expect(page.locator(".inspector .tool-tags")).not.toContainText("hire");
  await expect(page.locator(".inspector .tool-tags")).not.toContainText("fire");
  await expect(
    page.getByRole("button", { name: "Open specialist artifact", exact: true }),
  ).toBeVisible();
  await page.locator("#run-control").click();
  await expect(page.locator("#step-control")).toBeEnabled();
  await page
    .getByRole("button", { name: "Send instruction", exact: true })
    .click();
  await page
    .getByRole("textbox", { name: "Specialist instruction", exact: true })
    .fill(
      "BROWSER_OPERATOR: keep the evidence explicit and report this correction to your parent.",
    );
  await page
    .getByRole("button", { name: "Apply intervention", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.locator("#run-control")).toContainText("Resume");
  await page.locator("#run-control").click();
  await expect(
    page.locator('[data-pane="primary"] .message').filter({
      hasText: "BROWSER_OPERATOR instruction applied at the next work step.",
    }),
  ).toBeVisible();
  await page.locator('[data-agent="builder"]').click();
  await expect(
    page.locator('.inspector [data-specialist="browser_scout"]'),
  ).toBeVisible();
  await expect(
    page.locator('[data-pane="primary"] .message').filter({
      hasText: "Subagent browser_scout changed from pending to running.",
    }),
  ).toBeVisible();
  await scout.click();
  await expect(
    page.locator('.inspector [data-specialist="browser_leaf"]'),
  ).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("specialist-family.png") });
  await page
    .getByRole("button", { name: "Fire specialist", exact: true })
    .click();
  await expect(scout).toHaveCount(0);
  await expect(leaf).toHaveCount(0);
  await expect(page.locator(".person-card")).toHaveCount(4);
  await expect(page.locator(".inspector .detail-heading .badge")).toHaveText(
    "retired",
  );
  await expect(page.locator("#run-control")).toContainText("Complete");
  const finished = (await (
    await request.get(`/api/runs/${created.manifest.run_id}`)
  ).json()) as Snapshot;
  expect(finished.manifest.settings.team).toEqual(settings.team);
  expect(finished.delegation?.agents.browser_scout.status).toBe("retired");
  expect(finished.delegation?.agents.browser_leaf.status).toBe("retired");
  expect(finished.control?.active_nodes).toHaveLength(0);
  await page.reload();
  await expect(page.locator(".person-card")).toHaveCount(4);
  await page
    .getByRole("button", { name: "Inspect specialist history", exact: true })
    .click();
  await page.locator('.inspector [data-specialist="browser_leaf"]').click();
  await expect(page.locator(".inspector .detail-heading .badge")).toHaveText(
    "retired",
  );
  await expect(
    page.getByRole("button", { name: "Send instruction", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Fire specialist", exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Open specialist artifact", exact: true })
    .click();
  await expect(
    page.getByRole("textbox", { name: "Artifact contents", exact: true }),
  ).toHaveValue(/COMPLETE evidence from browser_leaf/);
  await expect(
    page.getByRole("textbox", { name: "Artifact contents", exact: true }),
  ).toHaveAttribute("readonly", "");
  await expect(
    page.getByRole("button", { name: "Save & reopen", exact: true }),
  ).toHaveCount(0);
  expect(errors).toEqual([]);
});
