import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Each test starts with an empty page and no saved chat.
afterEach(() => {
  cleanup();
  sessionStorage.clear();
});
