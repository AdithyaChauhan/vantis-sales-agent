import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Each test starts with an empty page at "/", no saved chat, no sales key and no saved preferences.
// (Tests that run without a page, like contrast.test.js, have nothing to reset.)
afterEach(() => {
  if (typeof window === "undefined") return;
  cleanup();
  sessionStorage.clear();
  localStorage.clear();
  window.history.replaceState(null, "", "/");
});
