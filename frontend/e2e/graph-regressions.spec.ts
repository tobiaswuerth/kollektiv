import { expect, test, type Page } from "@playwright/test";

async function expectPeopleInView(page: Page, count: number) {
  await expect(page.locator(".person-card")).toHaveCount(count);
  await expect(async () => {
    const viewport = (await page.locator(".graph-viewport").boundingBox())!;
    const transform = await page
      .locator(".graph-canvas .react-flow__viewport")
      .evaluate((el) => getComputedStyle(el).transform);
    expect(transform).not.toMatch(/NaN|Infinity/);
    for (const person of await page.locator(".person-card").all()) {
      const box = (await person.boundingBox())!;
      expect(box.x).toBeGreaterThanOrEqual(viewport.x - 1);
      expect(box.x + box.width).toBeLessThanOrEqual(
        viewport.x + viewport.width + 1,
      );
      expect(box.y).toBeGreaterThanOrEqual(viewport.y - 1);
      expect(box.y + box.height).toBeLessThanOrEqual(
        viewport.y + viewport.height + 1,
      );
    }
  }).toPass({ timeout: 6000 });
}

test("graph remains measured and visible through run switches, delete, resizing and reload", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const { settings } = await (await request.get("/api/bootstrap")).json();
  const first = await (
    await request.post("/api/runs", {
      data: { goal: "Graph camera regression A", settings },
    })
  ).json();
  const custom = structuredClone(settings);
  custom.team.push({ ...custom.team[2], name: "editor", role: "Copy editor" });
  const second = await (
    await request.post("/api/runs", {
      data: { goal: "Graph camera regression B", settings: custom },
    })
  ).json();
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");
  await expectPeopleInView(page, 5);
  for (const run of [first, second, first]) {
    await page.locator("#run-select").selectOption(run.manifest.run_id);
    await expectPeopleInView(page, run.manifest.settings.team.length);
  }
  await page.locator('[data-agent="builder"]').click();
  await page.keyboard.press("Delete");
  await expectPeopleInView(page, 4);
  const handle = page.getByRole("separator", { name: "Resize bottom dock" });
  await handle.focus();
  for (let i = 0; i < 6; i++) await page.keyboard.press("ArrowUp");
  await expectPeopleInView(page, 4);
  for (const viewport of [
    { width: 850, height: 700 },
    { width: 390, height: 844 },
    { width: 1440, height: 930 },
  ]) {
    await page.setViewportSize(viewport);
    await page
      .locator(".graph-card")
      .getByRole("button", { name: "Reset graph view" })
      .click();
    await expectPeopleInView(page, 4);
  }
  await page.reload();
  await page.locator("#run-select").selectOption(first.manifest.run_id);
  await expectPeopleInView(page, 4);
  expect(errors).toEqual([]);
});

test("moving packets remain on the displayed communication path while dragging and zooming", async ({
  page,
  request,
}) => {
  const { settings } = await (await request.get("/api/bootstrap")).json();
  const created = await (
    await request.post("/api/runs", {
      data: { goal: "Create design.md and a consistent story.md", settings },
    })
  ).json();
  await page.goto("/");
  await page.locator("#run-select").selectOption(created.manifest.run_id);
  await expectPeopleInView(page, 4);
  await page.locator("#run-control").click();
  await expect(page.locator("circle[data-flow-event]").first()).toBeAttached();
  const assertOnPath = async () => {
    const distances = await page
      .locator("circle[data-flow-event]")
      .evaluateAll((circles) =>
        circles.slice(0, 20).map((element) => {
          const circle = element as SVGCircleElement;
          const geometry = document.querySelector(
            `[data-flow-path="${CSS.escape(circle.dataset.flowEdge!)}"]`,
          ) as SVGPathElement;
          const displayed = geometry.parentElement!.querySelector(
            ".react-flow__edge-path",
          ) as SVGPathElement;
          if (
            !displayed ||
            !geometry ||
            displayed.getAttribute("d") !== geometry.getAttribute("d")
          )
            return Infinity;
          const point = displayed.getPointAtLength(
            displayed.getTotalLength() * Number(circle.dataset.flowProgress),
          );
          const center = new DOMPoint(
            circle.cx.baseVal.value,
            circle.cy.baseVal.value,
          ).matrixTransform(circle.getScreenCTM()!);
          const expected = new DOMPoint(point.x, point.y).matrixTransform(
            displayed.getScreenCTM()!,
          );
          return Math.hypot(center.x - expected.x, center.y - expected.y);
        }),
      );
    expect(distances.length).toBeGreaterThan(0);
    expect(Math.max(...distances)).toBeLessThan(1);
  };
  await assertOnPath();
  const person = (await page.locator('[data-agent="builder"]').boundingBox())!;
  await page.mouse.move(person.x + 30, person.y + 20);
  await page.mouse.down();
  await page.mouse.move(person.x + 70, person.y + 35, { steps: 6 });
  await page.mouse.up();
  await assertOnPath();
  const graph = (await page.locator(".graph-viewport").boundingBox())!;
  await page.mouse.move(graph.x + graph.width / 2, graph.y + graph.height / 2);
  await page.mouse.wheel(0, -100);
  await assertOnPath();
  await expect(page.locator("#run-control")).toContainText("Complete");
});

test("mission dropdown loads each difficulty and retains editable text", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator("#new-run").click();
  const picker = page.getByLabel("Mission sample", { exact: true });
  await expect(picker.locator("optgroup")).toHaveCount(3);
  for (const value of [
    "pip-field-card",
    "pip-silent-adventure",
    "keyboard-pong",
  ]) {
    await picker.selectOption(value);
    await expect(page.locator("#goal-input")).not.toHaveValue("");
  }
  await page.locator("#goal-input").fill("My customized mission");
  await expect(page.locator("#goal-input")).toHaveValue(
    "My customized mission",
  );
});

test("older static routes and missing catalogue still show a working graph and mission picker", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/assets/**", (route) =>
    route.fulfill({ status: 404, body: "Not found" }),
  );
  await page.route("**/api/bootstrap", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    delete body.mission_samples;
    delete body.settings.communication_links;
    await route.fulfill({ response, json: body });
  });
  await page.goto("/");
  await expect(page.locator(".person-card").first()).toBeVisible();
  await page.locator("#new-run").click();
  await page
    .getByLabel("Mission sample", { exact: true })
    .selectOption("pip-silent-adventure");
  await expect(page.locator("#goal-input")).toHaveValue(/design.md.*story.md/s);
  await page
    .getByRole("button", { name: "Create collective", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.locator(".mission h1")).toContainText("Pip");
  expect(errors).toEqual([]);
});
