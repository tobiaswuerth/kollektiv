import { expect, test } from "@playwright/test";
import type { Settings, Snapshot, TeamConfiguration } from "../src/types";

test("two-agent drafts can save, load, shrink to one shared orchestrator and create a stable graph", async ({
  page,
  request,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const { settings } = (await (await request.get("/api/bootstrap")).json()) as {
    settings: Settings;
  };
  const builder = structuredClone(
    settings.team.find((agent) => agent.name === "builder")!,
  );
  const reviewer = structuredClone(
    settings.team.find((agent) => agent.name === settings.reviewer)!,
  );
  builder.name = "mini_builder";
  reviewer.name = "mini_review";
  settings.team = [builder, reviewer];
  settings.coordinator = builder.name;
  settings.reviewer = reviewer.name;
  settings.communication_links = [
    { source: builder.name, target: reviewer.name },
  ];
  const response = await request.post("/api/runs", {
    data: { goal: "Create one concise useful guide in guide.md.", settings },
  });
  expect(response.ok()).toBe(true);
  const created = (await response.json()) as Snapshot;
  await page.goto("/");
  await page.locator("#run-select").selectOption(created.manifest.run_id);
  await expect(page.locator(".person-card")).toHaveCount(2);
  await page.getByRole("button", { name: "Edit team", exact: true }).click();
  const editor = page.getByRole("dialog", {
    name: "Edit collective",
    exact: true,
  });
  const pairName = `Two-agent regression ${Date.now()}`;
  await editor
    .getByRole("textbox", { name: "Configuration name", exact: true })
    .fill(pairName);
  await editor.getByRole("button", { name: "Save", exact: true }).click();
  await expect(editor.getByRole("status")).toHaveText(`Saved ${pairName}.`);
  const pair = (
    await (await request.get("/api/configurations")).json()
  ).configurations.find(
    (entry: TeamConfiguration) => entry.name === pairName,
  ) as TeamConfiguration;
  await editor.locator('[data-draft-agent="mini_review"]').click();
  await editor
    .getByRole("button", { name: "Remove selected agent", exact: true })
    .click();
  await expect(editor.locator(".draft-agent")).toHaveCount(1);
  await expect(editor.locator(".draft-agent-details")).toContainText(
    "Coordinator & reviewer",
  );
  await expect(
    editor.getByRole("button", { name: "Remove selected agent", exact: true }),
  ).toBeDisabled();
  await expect(editor.locator(".react-flow__edge")).toHaveCount(0);
  await expect(
    editor.getByRole("button", { name: "Create collective", exact: true }),
  ).toBeEnabled();
  await editor.getByRole("tab", { name: "Settings", exact: true }).click();
  await expect(
    editor.getByRole("textbox", { name: "Planning & review", exact: true }),
  ).toHaveValue("mini_builder");
  const soloName = `One-agent regression ${Date.now()}`;
  await editor
    .getByRole("textbox", { name: "Configuration name", exact: true })
    .fill(soloName);
  await editor
    .getByRole("button", { name: "Save configuration as copy", exact: true })
    .click();
  await expect(editor.getByRole("status")).toHaveText(`Saved ${soloName}.`);
  const solo = (
    await (await request.get("/api/configurations")).json()
  ).configurations.find(
    (entry: TeamConfiguration) => entry.name === soloName,
  ) as TeamConfiguration;
  expect(solo.settings.team).toHaveLength(1);
  expect(solo.settings.coordinator).toBe("mini_builder");
  expect(solo.settings.reviewer).toBe("mini_builder");
  await editor
    .getByRole("combobox", { name: "Load saved configuration", exact: true })
    .selectOption(pair.id!);
  await expect(editor.locator(".draft-agent")).toHaveCount(2);
  await editor
    .getByRole("combobox", { name: "Load saved configuration", exact: true })
    .selectOption(solo.id!);
  await expect(editor.locator(".draft-agent")).toHaveCount(1);
  await editor
    .getByRole("button", { name: "Create collective", exact: true })
    .click();
  await expect(editor).toHaveCount(0);
  await expect(page.locator(".person-card")).toHaveCount(1);
  await expect(page.locator(".person-card")).toContainText(
    "Coordinator & reviewer",
  );
  await expect(page.locator(".react-flow__edge")).toHaveCount(0);
  await page.reload();
  await expect(page.locator(".person-card")).toHaveCount(1);
  await expect(page.locator(".person-card")).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("single-agent.png") });
  expect(errors).toEqual([]);
});

test("one assembly-only orchestrator hires its author, publishes accepted work and retires the specialist", async ({
  page,
  request,
}, testInfo) => {
  const { settings } = (await (await request.get("/api/bootstrap")).json()) as {
    settings: Settings;
  };
  settings.team = [
    {
      name: "orchestrator",
      role: "Collective orchestrator",
      persona:
        "Hire authors, monitor their progress and publish complete accepted contributions.",
      skills: ["planning", "delegation", "integration"],
      tools: [
        "list_files",
        "read_file",
        "assemble_files",
        "check_file",
        "hire",
        "fire",
        "monitor_agents",
        "instruct_agent",
      ],
    },
  ];
  settings.coordinator = "orchestrator";
  settings.reviewer = "orchestrator";
  settings.communication_links = [];
  settings.max_concurrency = 2;
  settings.discussion_rounds = 1;
  settings.max_steps = 24;
  const response = await request.post("/api/runs", {
    data: {
      goal: "SOLO_BROWSER Publish the complete hired author's contribution in solo_result.md.",
      settings,
    },
  });
  expect(response.ok()).toBe(true);
  const created = (await response.json()) as Snapshot;
  await page.goto("/");
  await page.locator("#run-select").selectOption(created.manifest.run_id);
  await expect(page.locator(".person-card")).toHaveCount(1);
  await page.locator("#run-control").click();
  await expect(page.locator('[data-agent="solo_writer"]')).toHaveAttribute(
    "data-parent",
    "orchestrator",
  );
  await expect(page.locator("#run-control")).toContainText("Complete");
  await expect(page.locator(".person-card")).toHaveCount(2);
  const snapshot = (await (
    await request.get(`/api/runs/${created.manifest.run_id}`)
  ).json()) as Snapshot;
  expect(snapshot.manifest.settings.team).toHaveLength(1);
  expect(snapshot.manifest.settings.team[0].tools).not.toContain("write_file");
  expect(snapshot.delegation?.agents.solo_writer.status).toBe("completed");
  const contribution = await (
    await request.get(
      `/api/runs/${created.manifest.run_id}/artifact?path=delegates%2Fsolo_writer%2Fnote.md`,
    )
  ).json();
  const publication = await (
    await request.get(
      `/api/runs/${created.manifest.run_id}/artifact?path=solo_result.md`,
    )
  ).json();
  expect(publication.content).toBe(contribution.content);
  expect(publication.content).toContain("COMPLETE");
  await page.locator('[data-task="solo"]').click();
  await page.getByRole("button", { name: "Reassign", exact: true }).click();
  await expect(
    page.getByRole("option", { name: "orchestrator", exact: true }),
  ).toBeAttached();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await page.locator('[data-agent="solo_writer"]').click();
  await page
    .getByRole("button", { name: "Fire specialist", exact: true })
    .click();
  await expect(page.locator(".person-card")).toHaveCount(1);
  await expect(page.locator("#run-control")).toContainText("Complete");
  await page.screenshot({ path: testInfo.outputPath("solo-orchestrator.png") });
});

test("the solo orchestrator sample loads an editable task placeholder and creates one manager", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "New collective", exact: true })
    .click();
  const dialog = page.getByRole("dialog", {
    name: "Create a collective",
    exact: true,
  });
  await dialog
    .getByRole("combobox", { name: "Mission sample", exact: true })
    .selectOption("solo-orchestrator");
  await expect(dialog.locator("#goal-input")).toHaveValue(/\[YOUR TASK HERE\]/);
  const settings = JSON.parse(
    await dialog
      .locator('textarea[aria-label="Team and advanced settings"]')
      .inputValue(),
  ) as Settings;
  expect(settings.team).toHaveLength(1);
  expect(settings.coordinator).toBe("orchestrator");
  expect(settings.reviewer).toBe("orchestrator");
  expect(settings.team[0].tools).toEqual(
    expect.arrayContaining([
      "hire",
      "fire",
      "monitor_agents",
      "instruct_agent",
      "assemble_files",
    ]),
  );
  expect(settings.team[0].tools).not.toContain("write_file");
  await dialog
    .locator("#goal-input")
    .fill(
      "Use focused specialists to write a concise field guide in field_guide.md.",
    );
  await dialog
    .getByRole("button", { name: "Create collective", exact: true })
    .click();
  await expect(dialog).toHaveCount(0);
  await expect(page.locator(".person-card")).toHaveCount(1);
  await expect(page.locator('[data-agent="orchestrator"]')).toContainText(
    "Coordinator & reviewer",
  );
  await expect(page.locator(".react-flow__edge")).toHaveCount(0);
});
