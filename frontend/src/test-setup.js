import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Each test starts with an empty page, no saved chat and no saved preferences.
afterEach(() => {
  cleanup();
  sessionStorage.clear();
  localStorage.clear();
});
