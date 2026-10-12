import { expect, test } from "@playwright/test";

test("graph, independent dock panes, live flow, controls, admin, inspector and resizing", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.locator(".person-card")).toHaveCount(4);
  await page.locator("#new-run").click();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.locator("#new-run").click();
  await page
    .locator("#goal-input")
    .fill("Create design.md and a consistent story.md for the browser test.");
  await page
    .getByRole("button", { name: "Create collective", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.locator("#step-control").click();
  await expect(page.locator("#run-control")).toContainText("Resume");
  await expect(page.locator("#step-control")).toBeEnabled();
  await page.locator('[data-agent="builder"]').click();
  await expect(page.locator(".inspector h3")).toHaveText("builder");
  await page
    .getByRole("button", { name: "Send instruction", exact: true })
    .click();
  await page
    .getByRole("textbox", { name: "Instruction" })
    .fill("BROWSER OPERATOR: keep the shared design consistent.");
  await page.getByRole("button", { name: "Apply intervention" }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.locator("#run-control").click();
  await expect(page.locator("[data-flow-event]").first()).toBeAttached();
  await page.screenshot({ path: testInfo.outputPath("live.png") });
  await expect(page.locator("#run-control")).toContainText("Complete");
  const left = page.locator('[data-pane="primary"]'),
    right = page.locator('[data-pane="secondary"]');
  await expect(left.locator(".message")).not.toHaveCount(0);
  await expect(right.locator(".debug-event")).not.toHaveCount(0);
  await left.locator('[data-tab="memory"]').click();
  await expect(left.locator(".memory-card")).not.toHaveCount(0);
  await expect(right.locator('[data-tab="debug"]')).toHaveClass("active");
  await right.locator('[data-tab="state"]').click();
  await expect(right.locator(".json-branch").first()).toContainText(
    "worker_states",
  );
  await right.locator('[data-tab="artifacts"]').click();
  await right.locator('[data-artifact="story.md"]').click();
  await expect(
    page.getByRole("textbox", { name: "Artifact contents" }),
  ).toHaveValue(/COMPLETE/);
  await page
    .getByRole("textbox", { name: "Artifact contents" })
    .fill("# story\nCOMPLETE\nOperator edit.");
  await page.getByRole("button", { name: "Save & reopen" }).click();
  await expect(page.locator("#step-control")).toBeEnabled();
  const dock = await page.locator(".data-dock").boundingBox();
  const overview = await page.locator(".overview").boundingBox();
  expect(dock!.x).toBe(0);
  expect(dock!.width).toBe(1440);
  expect(overview!.y + overview!.height).toBeCloseTo(dock!.y, 0);
  const split = page.getByRole("separator", { name: "Resize panel split" });
  await split.focus();
  await page.keyboard.press("ArrowRight");
  expect((await left.boundingBox())!.width).toBeGreaterThan(720);
  const dockHeight = page.getByRole("separator", {
    name: "Resize bottom dock",
  });
  await dockHeight.focus();
  await page.keyboard.press("ArrowUp");
  await expect(dockHeight).toHaveAttribute("aria-valuenow", "34");
  await page.locator('[data-agent="builder"]').click();
  const person = page.locator('[data-agent="builder"]');
  const before = await person.boundingBox();
  await page.mouse.move(before!.x + 40, before!.y + 15);
  await page.mouse.down();
  await page.mouse.move(before!.x + 70, before!.y + 30, { steps: 5 });
  await page.mouse.up();
  expect((await person.boundingBox())!.x).toBeGreaterThan(before!.x + 15);
  await page.screenshot({ path: testInfo.outputPath("desktop.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Close inspector" }).click();
  await expect(page.locator(".inspector")).toBeHidden();
  await expect(async () => {
    const viewport = await page.locator(".graph-viewport").boundingBox();
    for (const node of await page.locator(".person-card").all()) {
      const box = await node.boundingBox();
      expect(box!.x).toBeGreaterThanOrEqual(viewport!.x);
      expect(box!.x + box!.width).toBeLessThanOrEqual(
        viewport!.x + viewport!.width,
      );
      expect(box!.y).toBeGreaterThanOrEqual(viewport!.y);
      expect(box!.y + box!.height).toBeLessThanOrEqual(
        viewport!.y + viewport!.height,
      );
    }
  }).toPass({ timeout: 5000 });
  expect(await page.evaluate(() => document.body.scrollWidth)).toBe(390);
  await expect(left.locator('[data-tab="memory"]')).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("mobile.png"),
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("full snapshot polling supports a server without the event endpoint", async ({
  page,
}) => {
  let eventRequests = 0;
  await page.route("**/api/runs/*/events?*", (route) => {
    eventRequests++;
    return route.fulfill({
      status: 404,
      contentType: "application/json",
      body: '{"error":"Not found"}',
    });
  });
  await page.goto("/");
  await page.locator("#new-run").click();
  await page
    .locator("#goal-input")
    .fill("Create design.md and a consistent story.md with legacy polling.");
  await page
    .getByRole("button", { name: "Create collective", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.locator("#run-control").click();
  await expect(page.locator("#run-control")).toContainText("Complete");
  await expect(
    page.locator('[data-pane="secondary"] .debug-event').first(),
  ).toBeVisible();
  expect(eventRequests).toBeGreaterThan(0);
});
