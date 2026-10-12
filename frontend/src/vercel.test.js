import { describe, expect, it } from "vitest";

import vercel from "../vercel.json";
import { route } from "./App.jsx";

// T13a: Vercel serves real files (/assets/...) first; every other path gets index.html, so a refresh or a shared
// link to a page the app routes in the browser (src/router.jsx) works instead of a 404.
describe("vercel.json", () => {
  const [rule] = vercel.rewrites;
  const matches = (path) => new RegExp(`^${rule.source}$`).test(path);

  it("sends every page path to /index.html", () => {
    expect(vercel.rewrites).toHaveLength(1);
    expect(rule.destination).toBe("/index.html");
    for (const path of ["/", "/leads", "/leads/12", "/support", "/support/", "/nowhere"]) {
      expect(matches(path), path).toBe(true);
    }
  });

  it("the app knows each of those pages after the rewrite", () => {
    expect(route("/leads/12")).toEqual({ page: "leads", leadId: 12 });
    expect(route("/support")).toEqual({ page: "support" });
  });
});
