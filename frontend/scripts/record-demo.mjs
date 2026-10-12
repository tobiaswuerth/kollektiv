/**
 * Record the real UI against the isolated, scripted model demo server:
 *   uv run --extra dev python tests/ui_server.py --port 8768 --delay 0.12
 *   cd frontend && npm run demo:record
 * Requires ffmpeg on PATH (or FFMPEG_PATH), and Edge on Windows or BROWSER_PATH.
 * The run directory is temporary; this script deliberately refuses production ports.
 */
import { chromium, expect } from "@playwright/test";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const exec = promisify(execFile);
const pause = (milliseconds) =>
  new Promise((done) => setTimeout(done, milliseconds));
const repository = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const output = join(repository, "docs", "control-room.gif");
const demoUrl = new URL(process.env.DEMO_URL || "http://127.0.0.1:8768");
if (
  !["localhost", "127.0.0.1"].includes(demoUrl.hostname) ||
  demoUrl.port !== "8768"
) {
  throw new Error(
    "Use the isolated demo server on localhost:8768 for recording.",
  );
}
const fps = 8;
const viewport = { width: 1440, height: 1000 };
const temporaryRoot = resolve(tmpdir());
const frames = await mkdtemp(join(temporaryRoot, "kollektiv-demo-"));
const browser = await chromium.launch({
  executablePath:
    process.env.BROWSER_PATH ||
    (process.platform === "win32"
      ? "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
      : undefined),
  headless: true,
});
const page = await browser.newPage({
  viewport,
  reducedMotion: "no-preference",
});
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
let recording = false;
let frameCount = 0;
let capture;

try {
  await page.goto(demoUrl.origin);
  await expect(page.locator(".person-card")).toHaveCount(4);
  await expect(page.locator("#new-run")).toBeEnabled();
  recording = true;
  const started = performance.now();
  capture = (async () => {
    while (recording) {
      const frameStarted = performance.now();
      await page.screenshot({
        path: join(
          frames,
          `frame-${String(frameCount++).padStart(5, "0")}.png`,
        ),
      });
      await pause(Math.max(1, 1000 / fps - (performance.now() - frameStarted)));
    }
  })();

  await pause(600);
  await demonstrateEditor(page);
  await page.locator("#new-run").click();
  await page
    .getByRole("combobox", { name: "Mission sample" })
    .selectOption("pip-silent-adventure");
  await pause(700);
  await page
    .getByRole("button", { name: "Create collective", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await pause(450);
  await page.locator("#step-control").click();
  await expect(page.locator("#run-control")).toContainText("Resume");
  await pause(500);
  await page.locator("#run-control").click();
  await page.locator('[data-agent="builder"]').click();
  await expect(page.locator("[data-flow-event]").first()).toBeAttached();
  await pause(2600);
  await page.locator('[data-pane="primary"] [data-tab="memory"]').click();
  await pause(700);
  await page.locator('[data-pane="primary"] [data-tab="messages"]').click();
  await expect(page.locator("#run-control")).toContainText("Complete", {
    timeout: 20_000,
  });
  await pause(Math.max(1000, 14_000 - (performance.now() - started)));
  recording = false;
  await capture;
  if (errors.length) throw new Error(`Browser errors: ${errors.join("; ")}`);
  await mkdir(dirname(output), { recursive: true });
  await exec(process.env.FFMPEG_PATH || "ffmpeg", [
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-framerate",
    String(fps),
    "-i",
    join(frames, "frame-%05d.png"),
    "-filter_complex",
    "scale=1000:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
    "-loop",
    "0",
    output,
  ]);
  const file = await stat(output);
  console.log(
    JSON.stringify(
      {
        output,
        width: 1000,
        height: Math.round((viewport.height * 1000) / viewport.width),
        frames: frameCount,
        duration: frameCount / fps,
        bytes: file.size,
        browserErrors: errors,
      },
      null,
      2,
    ),
  );
} finally {
  recording = false;
  if (capture) await capture.catch(() => {});
  await browser.close();
  if (
    dirname(resolve(frames)) !== temporaryRoot ||
    !basename(frames).startsWith("kollektiv-demo-")
  ) {
    throw new Error("Refusing to clean up an unexpected recording directory.");
  }
  await rm(frames, { recursive: true, force: true });
}

async function demonstrateEditor(demoPage) {
  // These controls edit a draft, keeping the subsequent mission on the original four-agent team.
  await demoPage
    .getByRole("button", { name: "Edit team", exact: true })
    .click();
  await demoPage
    .getByRole("combobox", { name: "Mission sample" })
    .selectOption("pip-silent-adventure");
  await demoPage
    .getByRole("button", { name: "Add agent", exact: true })
    .click();
  await demoPage
    .getByRole("textbox", { name: "Role", exact: true })
    .fill("Continuity specialist");
  await demoPage
    .getByRole("button", { name: "Auto layout", exact: true })
    .click();
  await pause(600);
  await demoPage
    .getByRole("textbox", { name: "Configuration name", exact: true })
    .fill("Story workshop");
  await demoPage.getByRole("button", { name: "Save", exact: true }).click();
  await expect(demoPage.getByRole("status")).toContainText(
    "Saved Story workshop.",
  );
  await pause(500);
  await demoPage
    .getByRole("button", { name: "Close editor", exact: true })
    .click();
  await pause(250);
}
