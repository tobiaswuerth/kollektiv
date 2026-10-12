import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  timeout: 30_000,
  use: {
    baseURL: "http://127.0.0.1:8767",
    viewport: { width: 1440, height: 930 },
    launchOptions: {
      executablePath:
        process.env.BROWSER_PATH ||
        (process.platform === "win32"
          ? "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
          : undefined),
    },
  },
  webServer: {
    command:
      "uv run --project .. --extra dev python ../tests/ui_server.py --delay 0.02",
    url: "http://127.0.0.1:8767/api/bootstrap",
    reuseExistingServer: false,
  },
});
