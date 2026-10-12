import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import type { MissionSample, Settings, Snapshot } from "../src/types";

const samples = JSON.parse(
  readFileSync(
    new URL("../../src/kollektiv/mission_samples.json", import.meta.url),
    "utf8",
  ),
) as MissionSample[];

for (const id of ["scientific-paper-lab", "fifty-page-novel-studio"]) {
  test(`${id} creates its complete studio from New collective and persists the authored graph`, async ({
    page,
    request,
  }) => {
    const sample = samples.find((entry) => entry.id === id)!;
    const { settings } = (await (
      await request.get("/api/bootstrap")
    ).json()) as { settings: Settings };
    // Keep a default baseline independent of the previous test's active team.
    const baseline = (await (
      await request.post("/api/runs", {
        data: { goal: "Studio preset browser baseline", settings },
      })
    ).json()) as Snapshot;
    // A running server may still publish the old six-entry catalogue.
    await page.route("**/api/bootstrap", async (route) => {
      const response = await route.fetch();
      const data = await response.json();
      await route.fulfill({
        response,
        json: { ...data, mission_samples: samples.slice(0, 6) },
      });
    });
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    await page.locator("#run-select").selectOption(baseline.manifest.run_id);
    await expect(page.locator(".person-card")).toHaveCount(4);
    await page
      .getByRole("button", { name: "New collective", exact: true })
      .click();
    const dialog = page.getByRole("dialog", {
      name: "Create a collective",
      exact: true,
    });
    await dialog
      .getByLabel("Ollama model", { exact: true })
      .fill("studio-test-model");
    await dialog.getByLabel("Concurrent agents", { exact: true }).fill("2");
    await dialog.getByLabel("Mission sample", { exact: true }).selectOption(id);
    await expect(dialog.getByLabel("Shared goal", { exact: true })).toHaveValue(
      sample.goal,
    );
    await expect(
      dialog.getByLabel("Ollama model", { exact: true }),
    ).toHaveValue("studio-test-model");
    await expect(
      dialog.getByLabel("Concurrent agents", { exact: true }),
    ).toHaveValue("2");
    const createdResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/runs") &&
        response.request().method() === "POST",
    );
    await dialog
      .getByRole("button", { name: "Create collective", exact: true })
      .click();
    const created = (await (await createdResponse).json()) as Snapshot;
    expect(created.manifest.goal).toBe(sample.goal);
    expect(created.manifest.settings).toMatchObject({
      ...sample.settings,
      model: "studio-test-model",
      max_concurrency: 2,
      base_url: settings.base_url,
    });
    expect(created.manifest.positions).toEqual(sample.positions);
    await expect(dialog).toHaveCount(0);
    await expect(page.locator(".person-card")).toHaveCount(
      sample.settings!.team!.length,
    );
    await expect(page.locator(".graph-canvas .react-flow__edge")).toHaveCount(
      sample.settings!.communication_links!.length,
    );
    await page.reload();
    await page.locator("#run-select").selectOption(created.manifest.run_id);
    await expect(page.locator(".person-card")).toHaveCount(
      sample.settings!.team!.length,
    );
    const persisted = (await (
      await request.get(`/api/runs/${created.manifest.run_id}`)
    ).json()) as Snapshot;
    expect(persisted.manifest.positions).toEqual(sample.positions);
    await page.getByRole("button", { name: "Edit team", exact: true }).click();
    const editor = page.getByRole("dialog", {
      name: "Edit collective",
      exact: true,
    });
    expect(await editor.boundingBox()).toMatchObject({
      x: 0,
      y: 0,
      width: 1440,
      height: 930,
    });
    await expect(editor.locator(".draft-agent")).toHaveCount(
      sample.settings!.team!.length,
    );
    await editor.getByRole("tab", { name: "Agents", exact: true }).click();
    const publisher = sample.settings!.team!.find((agent) =>
      agent.tools.includes("assemble_files"),
    )!;
    await editor
      .getByRole("searchbox", { name: "Find an agent", exact: true })
      .fill(publisher.name);
    await editor
      .getByRole("button", {
        name: `Select agent ${publisher.name}`,
        exact: true,
      })
      .click();
    await expect(
      editor.getByRole("checkbox", { name: "assemble_files", exact: true }),
    ).toBeChecked();
    await editor
      .getByRole("button", { name: "Close editor", exact: true })
      .click();
    expect(errors).toEqual([]);
  });
}

test("easy samples replace a studio with a compact graph through both creation flows", async ({
  page,
  request,
}) => {
  const { settings } = await (await request.get("/api/bootstrap")).json();
  const studio = samples.find(
    (sample) => sample.id === "fifty-page-novel-studio",
  )!;
  const baseline = await (
    await request.post("/api/runs", {
      data: {
        goal: studio.goal,
        settings: { ...settings, ...studio.settings },
        positions: studio.positions,
      },
    })
  ).json();
  await page.goto("/");
  await page.locator("#run-select").selectOption(baseline.manifest.run_id);
  await expect(page.locator(".person-card")).toHaveCount(14);
  await page
    .getByRole("button", { name: "New collective", exact: true })
    .click();
  const dialog = page.getByRole("dialog", {
    name: "Create a collective",
    exact: true,
  });
  await dialog
    .getByLabel("Mission sample", { exact: true })
    .selectOption("pip-field-card");
  const createdResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/runs") &&
      response.request().method() === "POST",
  );
  await dialog
    .getByRole("button", { name: "Create collective", exact: true })
    .click();
  const created = (await (await createdResponse).json()) as Snapshot;
  expect(created.manifest.settings.team).toHaveLength(3);
  expect(Object.keys(created.manifest.positions!)).toHaveLength(3);
  await expect(page.locator(".person-card")).toHaveCount(3);
  await page.getByRole("button", { name: "Edit team", exact: true }).click();
  const editor = page.getByRole("dialog", {
    name: "Edit collective",
    exact: true,
  });
  for (const id of [
    "fifty-page-novel-studio",
    "picnic-plan",
    "hire-a-garden-specialist",
  ]) {
    await editor
      .getByRole("button", { name: "Mission & samples", exact: true })
      .click();
    await editor.getByLabel("Mission sample", { exact: true }).selectOption(id);
    await expect(editor.locator(".draft-agent")).toHaveCount(
      id === "fifty-page-novel-studio" ? 14 : 3,
    );
  }
  const gardenResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/runs") &&
      response.request().method() === "POST",
  );
  await editor
    .getByRole("button", { name: "Create collective", exact: true })
    .click();
  const garden = (await (await gardenResponse).json()) as Snapshot;
  expect(garden.manifest.settings.team).toHaveLength(3);
  expect(
    garden.manifest.settings.team.find((agent) => agent.name === "builder")!
      .tools,
  ).toEqual(
    expect.arrayContaining([
      "hire",
      "fire",
      "monitor_agents",
      "instruct_agent",
    ]),
  );
  expect(garden.manifest.settings.max_dynamic_agents).toBe(1);
  await expect(page.locator(".person-card")).toHaveCount(3);
});
