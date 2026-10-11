import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  // The API's CORS_ORIGINS allows exactly http://localhost:5173, so fail instead of moving to 5174.
  server: { port: 5173, strictPort: true },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.js"],
    restoreMocks: true,
    unstubGlobals: true,
    unstubEnvs: true,
  },
});
