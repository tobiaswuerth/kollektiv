import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../src/kollektiv/web",
    emptyOutDir: true,
    cssCodeSplit: false,
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        entryFileNames: "app.js",
        // Existing control-room processes serve these fixed filenames only.
        inlineDynamicImports: true,
        assetFileNames: (asset) =>
          asset.names?.some((name) => name.endsWith(".css"))
            ? "style.css"
            : "assets/[name]-[hash][extname]",
      },
    },
  },
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq, request) => {
            const host = request.headers.host;
            if (!host || request.headers.origin !== `http://${host}`) return;
            try {
              const dev = new URL(`http://${host}`);
              if (
                dev.host.toLowerCase() === host.toLowerCase() &&
                ["127.0.0.1", "localhost", "[::1]"].includes(dev.hostname)
              ) {
                // Preserve backend origin checks while allowing the local dev UI.
                proxyReq.setHeader("Origin", "http://127.0.0.1:8765");
              }
            } catch {
              // Invalid or foreign origins reach the backend unchanged and are refused.
            }
          });
        },
      },
    },
  },
});
