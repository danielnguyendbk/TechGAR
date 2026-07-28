import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 4173,
  },
  preview: {
    port: 4173,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/tests/setup.ts",
    css: true,
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
