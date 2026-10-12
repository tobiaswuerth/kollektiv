import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { MissionSample, TeamConfiguration } from "../src/types";

const bundledMissionSamples = JSON.parse(
  readFileSync(
    new URL("../../src/kollektiv/mission_samples.json", import.meta.url),
    "utf8",
  ),
) as MissionSample[];

test("large mission teams get a full-screen canvas, searchable roster, readable focus and saved edits", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.getByRole("button", { name: "Edit team", exact: true }).click();
  const editor = page.getByRole("dialog", {
    name: "Edit collective",
    exact: true,
  });
  const bounds = await editor.boundingBox();
  expect(bounds).toMatchObject({ x: 0, y: 0, width: 1440, height: 930 });

  for (const id of ["scientific-paper-lab", "fifty-page-novel-studio"]) {
    const sample = bundledMissionSamples.find((entry) => entry.id === id)!;
    await editor
      .getByRole("button", { name: "Mission & samples", exact: true })
      .click();
    await editor.getByLabel("Mission sample", { exact: true }).selectOption(id);
    await expect(editor.locator(".draft-agent")).toHaveCount(
      sample.settings!.team!.length,
    );
    const canvas = await editor.locator(".editor-canvas").boundingBox();
    expect(canvas!.width).toBeGreaterThan(1440 * 0.6);
    expect(canvas!.height).toBeGreaterThan(930 * 0.8);
    await expect(
      editor.getByRole("textbox", { name: "Mission goal", exact: true }),
    ).toHaveValue(sample.goal);
    if (id === "fifty-page-novel-studio")
      await page.screenshot({
        path: join(tmpdir(), "kollektiv-editor-1440-overview.png"),
      });

    await editor.getByRole("tab", { name: "Agents", exact: true }).click();
    const specialist = sample.settings!.team![4];
    await editor
      .getByRole("searchbox", { name: "Find an agent", exact: true })
      .fill(specialist.name);
    await expect(editor.locator(".editor-roster-list > button")).toHaveCount(1);
    await editor
      .getByRole("button", {
        name: `Select agent ${specialist.name}`,
        exact: true,
      })
      .click();
    await expect(
      editor.getByRole("textbox", { name: "Role", exact: true }),
    ).toHaveValue(specialist.role);
    const card = editor.locator(`[data-draft-agent="${specialist.name}"]`);
    await expect
      .poll(async () => (await card.boundingBox())?.width || 0)
      .toBeGreaterThan(190);
    if (id === "fifty-page-novel-studio")
      await page.screenshot({
        path: join(tmpdir(), "kollektiv-editor-1440-focus.png"),
      });
    await editor
      .getByRole("button", {
        name: "Highlight selected agent paths",
        exact: true,
      })
      .click();
    await expect(
      editor.getByRole("button", {
        name: "Highlight selected agent paths",
        exact: true,
      }),
    ).toHaveAttribute("aria-pressed", "true");
    if (id === "fifty-page-novel-studio") {
      const otherCard = editor.locator('[data-draft-agent="world_builder"]');
      const viewport = editor.locator(".react-flow__viewport");
      const previousViewport = await viewport.getAttribute("style");
      await otherCard.locator(".draft-agent-heading").click();
      await expect(
        editor.getByRole("textbox", { name: "Agent name", exact: true }),
      ).toHaveValue("world_builder");
      await expect(viewport).toHaveAttribute("style", previousViewport!);
      await editor
        .getByRole("button", { name: "Focus selected agent", exact: true })
        .click();
      const centered = async () => {
        const nodeBounds = await otherCard.boundingBox();
        const canvasBounds = await editor
          .locator(".editor-canvas")
          .boundingBox();
        return Math.hypot(
          nodeBounds!.x +
            nodeBounds!.width / 2 -
            canvasBounds!.x -
            canvasBounds!.width / 2,
          nodeBounds!.y +
            nodeBounds!.height / 2 -
            canvasBounds!.y -
            canvasBounds!.height / 2,
        );
      };
      await expect.poll(centered).toBeLessThan(3);
      await editor
        .getByRole("button", { name: "Hide agent inspector", exact: true })
        .click();
      await expect.poll(centered).toBeLessThan(3);
      await expect
        .poll(async () => (await otherCard.boundingBox())?.width || 0)
        .toBeGreaterThan(190);
      await editor
        .getByRole("button", { name: "Show agent inspector", exact: true })
        .click();
      await expect.poll(centered).toBeLessThan(3);
      await editor
        .getByRole("button", {
          name: `Select agent ${specialist.name}`,
          exact: true,
        })
        .click();
    }
    await editor
      .getByRole("textbox", { name: "Instructions", exact: true })
      .fill(`${specialist.persona}\nAdditional editorial test instruction.`);

    const saveResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/configurations") &&
        response.request().method() === "POST",
    );
    await editor.getByRole("button", { name: "Save", exact: true }).click();
    const saved = (await (await saveResponse).json()) as TeamConfiguration;
    expect(
      saved.settings.team.find((entry) => entry.name === specialist.name)!
        .persona,
    ).toContain("Additional editorial test instruction.");
    expect(saved.positions).toEqual(sample.positions);
    expect(saved.settings.communication_links).toEqual(
      sample.settings!.communication_links,
    );
    await editor
      .getByRole("textbox", { name: "Instructions", exact: true })
      .fill("Unsaved scratch changes");
    await editor
      .getByRole("combobox", { name: "Load saved configuration", exact: true })
      .selectOption("");
    await editor
      .getByRole("combobox", { name: "Load saved configuration", exact: true })
      .selectOption(saved.id!);
    await editor.getByRole("tab", { name: "Agents", exact: true }).click();
    await editor
      .getByRole("searchbox", { name: "Find an agent", exact: true })
      .fill(specialist.name);
    await editor
      .getByRole("button", {
        name: `Select agent ${specialist.name}`,
        exact: true,
      })
      .click();
    await expect(
      editor.getByRole("textbox", { name: "Instructions", exact: true }),
    ).toHaveValue(
      `${specialist.persona}\nAdditional editorial test instruction.`,
    );
    await editor
      .getByRole("button", { name: "Delete saved configuration", exact: true })
      .click();
    await expect(editor.getByRole("status")).toContainText(
      "Saved configuration removed",
    );
  }

  await page.setViewportSize({ width: 1280, height: 800 });
  await editor
    .getByRole("button", { name: "Hide agent inspector", exact: true })
    .click();
  await editor
    .getByRole("button", { name: "Reset graph view", exact: true })
    .click();
  await expect
    .poll(
      async () =>
        (
          await editor
            .locator('[data-draft-agent="character_designer"]')
            .boundingBox()
        )?.width || 0,
    )
    .toBeLessThan(190);
  await page.screenshot({
    path: join(tmpdir(), "kollektiv-editor-1280-overview.png"),
  });

  await editor
    .getByRole("button", { name: "Hide design panel", exact: true })
    .click();
  expect((await editor.locator(".editor-canvas").boundingBox())!.width).toBe(
    1280,
  );
  await page.keyboard.press("Escape");
  await expect(editor).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("phone editor keeps the graph tall while mission, library and inspector use dismissible drawers", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: "Edit team", exact: true }).click();
  const editor = page.getByRole("dialog", {
    name: "Edit collective",
    exact: true,
  });
  expect(await editor.boundingBox()).toMatchObject({
    x: 0,
    y: 0,
    width: 390,
    height: 844,
  });
  const graph = editor.locator(".editor-canvas");
  expect((await graph.boundingBox())!.height).toBeGreaterThan(600);
  expect((await graph.boundingBox())!.width).toBe(390);
  await editor
    .getByRole("button", { name: "Mission & samples", exact: true })
    .click();
  await editor
    .getByLabel("Mission sample", { exact: true })
    .selectOption("fifty-page-novel-studio");
  await expect(editor.locator(".draft-agent")).toHaveCount(14);
  await editor.getByRole("tab", { name: "Library", exact: true }).click();
  await editor
    .getByRole("textbox", { name: "Configuration name", exact: true })
    .fill("Phone novel studio");
  await expect(
    editor.getByRole("combobox", {
      name: "Load saved configuration",
      exact: true,
    }),
  ).toBeVisible();
  await editor
    .getByRole("button", { name: "Close design panel", exact: true })
    .click();
  await editor.getByRole("button", { name: "Add agent", exact: true }).click();
  await expect(
    editor.getByRole("textbox", { name: "Role", exact: true }),
  ).toBeVisible();
  await editor
    .getByRole("textbox", { name: "Role", exact: true })
    .fill("Additional specialist");
  await editor
    .getByRole("button", { name: "Remove selected agent", exact: true })
    .click();
  await editor
    .getByRole("button", { name: "Close agent inspector", exact: true })
    .click();
  expect((await graph.boundingBox())!.width).toBe(390);
  expect(await editor.evaluate((element) => element.scrollWidth)).toBe(390);
  await editor
    .getByRole("button", { name: "Close editor", exact: true })
    .click();
  await expect(editor).toHaveCount(0);
});
