import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
import type { TeamConfiguration } from "../src/types";

test("edit agents and directed paths, save/load, JSON round trip and persist run layout", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.locator(".person-card")).toHaveCount(4);
  await page.getByRole("button", { name: "Edit team", exact: true }).click();
  const editor = page.getByRole("dialog", {
    name: "Edit collective",
    exact: true,
  });
  await expect(editor).toBeVisible();
  await editor
    .getByRole("textbox", { name: "Mission goal", exact: true })
    .fill("Write a short guide for the editor browser test.");
  const configurationName = `Editor regression ${Date.now()}`;
  await editor
    .getByRole("textbox", { name: "Configuration name", exact: true })
    .fill(configurationName);
  await editor.getByRole("button", { name: "Add agent", exact: true }).click();
  await expect(editor.locator(".draft-agent")).toHaveCount(5);
  await editor
    .getByRole("textbox", { name: "Agent name", exact: true })
    .fill("guide");
  await editor
    .getByRole("textbox", { name: "Role", exact: true })
    .fill("Documentation specialist");
  await editor
    .getByRole("textbox", { name: "Instructions", exact: true })
    .fill(
      "Produce clear, short documentation and inspect shared files before writing.",
    );
  await editor
    .getByRole("textbox", { name: "Skills", exact: true })
    .fill("writing, documentation");
  await editor.getByRole("textbox", { name: "Role", exact: true }).focus();
  await editor
    .getByRole("checkbox", { name: "check_file", exact: true })
    .uncheck();
  await editor
    .getByRole("checkbox", { name: "send_message", exact: true })
    .uncheck();
  await editor
    .getByRole("button", { name: "Auto layout", exact: true })
    .click();
  await editor
    .getByRole("button", { name: "Reset graph view", exact: true })
    .click();
  const source = await editor
    .locator('[data-draft-agent="coordinator"] .react-flow__handle.source')
    .boundingBox();
  const target = await editor
    .locator('[data-draft-agent="guide"] .react-flow__handle.target')
    .boundingBox();
  await page.mouse.move(
    source!.x + source!.width / 2,
    source!.y + source!.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(
    target!.x + target!.width / 2,
    target!.y + target!.height / 2,
    { steps: 12 },
  );
  await page.mouse.up();
  await expect(
    editor.getByRole("combobox", { name: "Communication mode" }),
  ).toHaveValue("custom");
  await expect(
    editor.getByRole("button", {
      name: "Remove path coordinator to guide",
      exact: true,
    }),
  ).toBeVisible();
  await editor
    .getByRole("button", {
      name: "Remove path coordinator to guide",
      exact: true,
    })
    .click();
  await expect(editor.locator(".react-flow__edge")).toHaveCount(0);
  // Choose a valid starter topology and ensure a new node can be removed too.
  await editor
    .getByRole("combobox", { name: "Communication mode" })
    .selectOption("unrestricted");
  await editor
    .getByRole("combobox", { name: "Communication mode" })
    .selectOption("custom");
  await editor.getByRole("button", { name: "Add agent", exact: true }).click();
  await expect(editor.locator(".draft-agent")).toHaveCount(6);
  await editor
    .getByRole("button", { name: "Remove selected agent", exact: true })
    .click();
  await expect(editor.locator(".draft-agent")).toHaveCount(5);
  const saveResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/configurations") &&
      response.request().method() === "POST",
  );
  await editor.getByRole("button", { name: "Save", exact: true }).click();
  const saved = (await (await saveResponse).json()) as TeamConfiguration;
  await expect(editor.getByRole("status")).toContainText(
    `Saved ${configurationName}`,
  );
  expect(
    saved.settings.team.find((agent) => agent.name === "guide"),
  ).toMatchObject({
    role: "Documentation specialist",
    persona:
      "Produce clear, short documentation and inspect shared files before writing.",
    skills: ["writing", "documentation"],
  });
  expect(
    saved.settings.team.find((agent) => agent.name === "guide")!.tools,
  ).not.toContain("send_message");
  expect(saved.settings.communication_links).toContainEqual({
    source: "coordinator",
    target: "guide",
  });
  await editor
    .getByRole("textbox", { name: "Configuration name", exact: true })
    .fill("Unsaved change");
  await editor
    .getByRole("combobox", { name: "Load saved configuration", exact: true })
    .selectOption("");
  await editor
    .getByRole("combobox", { name: "Load saved configuration", exact: true })
    .selectOption(saved.id!);
  await expect(
    editor.getByRole("textbox", { name: "Configuration name", exact: true }),
  ).toHaveValue(configurationName);
  const downloadEvent = page.waitForEvent("download");
  await editor
    .getByRole("button", { name: "Export configuration JSON", exact: true })
    .click();
  const file = await downloadEvent;
  const exported = JSON.parse(
    await readFile((await file.path())!, "utf8"),
  ) as TeamConfiguration;
  expect(exported.settings).toEqual(saved.settings);
  expect(exported.positions).toEqual(saved.positions);
  const imported = {
    ...exported,
    name: `${configurationName} imported`,
    positions: { ...exported.positions, guide: { x: 444, y: 222 } },
  };
  await editor
    .getByLabel("Import configuration JSON", { exact: true })
    .setInputFiles({
      name: "collective.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(imported)),
    });
  await expect(
    editor.getByRole("textbox", { name: "Configuration name", exact: true }),
  ).toHaveValue(imported.name);
  const createResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/runs") &&
      response.request().method() === "POST",
  );
  await editor
    .getByRole("button", { name: "Create collective", exact: true })
    .click();
  const created = await (await createResponse).json();
  await expect(editor).toHaveCount(0);
  await expect(page.locator(".person-card")).toHaveCount(5);
  expect(created.manifest.positions.guide).toEqual({ x: 444, y: 222 });
  await page.reload();
  await expect(page.locator('[data-agent="guide"]')).toBeVisible();
  const manifest = await (
    await request.get(`/api/runs/${created.manifest.run_id}`)
  ).json();
  expect(manifest.manifest.positions.guide).toEqual({ x: 444, y: 222 });
  await page.getByRole("button", { name: "Edit team", exact: true }).click();
  await editor
    .getByRole("combobox", { name: "Load saved configuration", exact: true })
    .selectOption(saved.id!);
  await editor
    .getByRole("button", { name: "Delete saved configuration", exact: true })
    .click();
  await expect(editor.getByRole("status")).toContainText(
    "Saved configuration removed",
  );
  await expect(
    editor
      .getByRole("combobox", { name: "Load saved configuration", exact: true })
      .locator(`option[value="${saved.id}"]`),
  ).toHaveCount(0);
  await editor
    .getByRole("button", { name: "Close editor", exact: true })
    .click();
  expect(errors).toEqual([]);
});

test("invalid imports stay reviewable and mobile editor remains usable", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Edit team", exact: true }).click();
  const editor = page.getByRole("dialog", {
    name: "Edit collective",
    exact: true,
  });
  await editor
    .getByLabel("Import configuration JSON", { exact: true })
    .setInputFiles({
      name: "invalid.json",
      mimeType: "application/json",
      buffer: Buffer.from('{"version":2}'),
    });
  await expect(editor.getByRole("alert")).toContainText("version 1");
  await expect(editor.locator(".draft-agent").first()).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await editor.getByRole("button", { name: "Add agent", exact: true }).click();
  await editor
    .getByRole("textbox", { name: "Role", exact: true })
    .fill("Mobile specialist");
  await expect(
    editor.getByRole("textbox", { name: "Role", exact: true }),
  ).toHaveValue("Mobile specialist");
  expect(
    await editor.evaluate((element) => element.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await editor
    .getByRole("button", { name: "Remove selected agent", exact: true })
    .click();
  await editor
    .getByRole("button", { name: "Close editor", exact: true })
    .click();
  await expect(editor).toHaveCount(0);
});
