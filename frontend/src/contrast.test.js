// @vitest-environment node
// (no page needed: it only reads styles.css)
import fs from "node:fs";
import { describe, expect, it } from "vitest";

/**
 * WCAG 2.x contrast for every text, icon and control colour pair the pages use, in light and dark:
 * text at least 4.5:1 (AA), icons and controls at least 3:1. The colours are read from styles.css, so
 * a token change that breaks a pair fails here. T10c checked its 68 pairs with a script; T11 moved the
 * check into the tests and added the dashboard's pairs.
 * CONTRAST_TABLE=1 npx vitest run src/contrast.test.js prints the table for docs/PLAN.md.
 */

const TEXT = 4.5;
const ICON = 3; // icons, control boundaries and states (WCAG 1.4.11)

const css = fs.readFileSync(new URL("./styles.css", import.meta.url), "utf8");

/** {"text": "#101b3f", ...} from one :root block (hex values only; gradients are given by hand below). */
function hexTokens(block) {
  return Object.fromEntries(
    [...block.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})\b/g)].map(([, name, hex]) => [name, hex.toLowerCase()]),
  );
}

function rootBlock(from) {
  const start = css.indexOf(":root {", from);
  return css.slice(start, css.indexOf("}", start));
}

const LIGHT = hexTokens(rootBlock(0));
const DARK = { ...LIGHT, ...hexTokens(rootBlock(css.indexOf("@media (prefers-color-scheme: dark)"))) };

function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const linear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
}

/** The contrast ratio of two colours, e.g. ratio("#101b3f", "#fdf9f4") = 16.02. */
export function ratio(a, b) {
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (high + 0.05) / (low + 0.05);
}

/** [label, foreground, background, minimum] for one theme's tokens (t). */
function pairs(t, theme) {
  const scoreBlock = theme === "light" ? "#fef2e4" : t["score-bg"]; // light: the darker end of a gradient
  return [
    // Chat page (T10c)
    ["Body text on page", t.text, t.page, TEXT],
    ["Text on chat card", t.text, t.card, TEXT],
    ["Secondary text on card", t["text-2"], t.card, TEXT],
    ["Secondary text on panel", t["text-2"], t.panel, TEXT],
    ["Muted text on card (times, notes, table headers)", t.muted, t.card, TEXT],
    ["Muted placeholder on the message box", t.muted, t["field-bg"], TEXT],
    ["Muted citation tag in a reply bubble", t.muted, t.bubble, TEXT],
    ["Reply text on assistant bubble", t["bubble-text"], t.bubble, TEXT],
    ["How I answered on its footer", t["steps-text"], t["steps-bg"], TEXT],
    ["Step label on the footer", t.text, t["steps-bg"], TEXT],
    ["White text on customer bubble", t["user-text"], t["user-bubble"], TEXT],
    ["Source pill text", t["pill-text"], t["pill-bg"], TEXT],
    ["Navy on orange button, light end", t["on-orange"], t["orange-1"], TEXT],
    ["Navy on orange button, dark end", t["on-orange"], t["orange-2"], TEXT],
    ["Nav label on current item", t["nav-text"], t["nav-bg"], TEXT],
    ["Score number on score block", t["score-text"], scoreBlock, TEXT],
    ["'/ 100' on score block", t["text-2"], scoreBlock, TEXT],
    ["HOT badge", t["hot-text"], t["hot-bg"], TEXT],
    ["WARM badge", t["warm-text"], t["warm-bg"], TEXT],
    ["COLD badge", t["cold-text"], t["cold-bg"], TEXT],
    ["'Saved as lead' chip", t["saved-text"], t["saved-bg"], TEXT],
    ["'Not saved yet' chip, Demo tag", t["unsaved-text"], t["unsaved-bg"], TEXT],
    ["Starter question, filter chip, Lock button", t.text, t.panel, TEXT],
    ["Notice text, cap note, unlock note", t["notice-text"], t["notice-bg"], TEXT],
    ["Error text", t["danger-text"], t["danger-bg"], TEXT],
    ["Nav icon on current item", t["nav-icon"], t["nav-bg"], ICON],
    ["Check (collected)", t.check, t.panel, ICON],
    ["Empty circle (not yet)", t.empty, t.panel, ICON],
    ["Row icons", t["text-2"], t.panel, ICON],
    ["Switch track on, against page", t.accent, t.page, ICON],
    ["Switch thumb on, against track", t["switch-thumb-on"], t.accent, ICON],
    ["Switch track off, against page", t.empty, t.page, ICON],
    ["Switch thumb off, against track", "#ffffff", t.empty, ICON],
    ["Focus ring on card", t.accent, t.card, ICON],
    // T11: chat page scrollbars and staff links, the dashboard
    ["Scrollbar thumb on card", t["scroll-thumb"], t.card, ICON],
    ["Scrollbar thumb on page", t["scroll-thumb"], t.page, ICON],
    ["Scrollbar thumb on panel", t["scroll-thumb"], t.panel, ICON],
    ["Section link label on page", t.text, t.page, TEXT],
    ["Section link label on hover", t.text, t["nav-hover"], TEXT],
    ["Section link icon on page", t["text-2"], t.page, ICON],
    ["Dashboard subtitle, brand line on page", t["text-2"], t.page, TEXT],
    ["Search placeholder on its box", t.muted, t.panel, TEXT],
    ["Search icon on its box", t.muted, t.panel, ICON],
    ["Active filter chip, current section", t["nav-text"], t["chip-on"], TEXT],
    ["Current section icon (dashboard)", t["nav-icon"], t["chip-on"], ICON],
    ["Text on a selected row", t.text, t["row-selected"], TEXT],
    ["Secondary text on a selected row", t["text-2"], t["row-selected"], TEXT],
    ["Muted text on a selected row", t.muted, t["row-selected"], TEXT],
    ["Text on a hovered row", t.text, t["row-hover"], TEXT],
    ["Muted text on a hovered row", t.muted, t["row-hover"], TEXT],
    ["Selected row outline against its fill", t["select-border"], t["row-selected"], ICON],
    ["Selected row outline against the card", t["select-border"], t.card, ICON],
    ["Next action text", t.text, t["next-bg"], TEXT],
    ["Next action icon", t.accent, t["next-bg"], ICON],
    ["'Score before the cap' on the cap note", t["text-2"], t["notice-bg"], TEXT],
    ["Cap icon on the cap note", t.accent, t["notice-bg"], ICON],
    ["Contact icons on card", t.accent, t.card, ICON],
    ["Unlock key icon on its circle", t.accent, t["sun-avatar-bg"], ICON],
    ["Text on a phone card (dashboard)", t.text, t.card, TEXT],
    ["Waking note text on its card (T13a)", t.text, t.card, TEXT],
  ];
}

describe.each([
  ["light", LIGHT],
  ["dark", DARK],
])("contrast, %s", (theme, tokens) => {
  const list = pairs(tokens, theme);

  it("reads every colour it needs from styles.css", () => {
    const missing = list.filter(([, fg, bg]) => !/^#[0-9a-f]{6}$/.test(fg ?? "") || !/^#[0-9a-f]{6}$/.test(bg ?? ""));
    expect(missing.map(([label]) => label)).toEqual([]);
  });

  it.each(list)("%s", (label, fg, bg, minimum) => {
    expect(ratio(fg, bg)).toBeGreaterThanOrEqual(minimum);
  });

  if (process.env.CONTRAST_TABLE) {
    it("prints the table", () => {
      const rows = list.map(([label, fg, bg, minimum]) => `| ${label} | \`${fg}\` on \`${bg}\` | ${ratio(fg, bg).toFixed(2)} | ${minimum} |`);
      console.log(`\n${theme} (${list.length} pairs)\n| Pair | Colours | Ratio | Target |\n|---|---|---|---|\n${rows.join("\n")}`);
    });
  }
});
