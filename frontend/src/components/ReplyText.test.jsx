import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ReplyText from "./ReplyText.jsx";

describe("ReplyText", () => {
  it("shows a <script> reply as plain text and never as HTML", () => {
    const { container } = render(<ReplyText text={'<script>alert(1)</script> <img src=x onerror="alert(2)">'} />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<script>alert(1)</script>");
  });

  it("turns line breaks into paragraphs", () => {
    const { container } = render(<ReplyText text={"First line.\nSecond line.\n\nThird."} />);
    expect([...container.querySelectorAll("p")].map((p) => p.textContent)).toEqual([
      "First line.",
      "Second line.",
      "Third.",
    ]);
  });

  it("turns '- ' lines into a list and **x** into bold", () => {
    const { container } = render(<ReplyText text={"Options:\n- **HomeStart 3**: 3 kW\n- HomePro 5"} />);
    expect([...container.querySelectorAll("li")].map((li) => li.textContent)).toEqual([
      "HomeStart 3: 3 kW",
      "HomePro 5",
    ]);
    expect(container.querySelector("strong").textContent).toBe("HomeStart 3");
  });

  it("shows a KB citation as a short tag with the full title on hover", () => {
    render(<ReplyText text="We offer three packages [KB Section 8: Product & Service Catalog]." />);
    const tag = screen.getByTitle("KB Section 8: Product & Service Catalog");
    expect(tag.firstChild.textContent).toBe("KB 8"); // small and quiet
    expect(tag.querySelector(".visually-hidden").textContent).toBe("KB Section 8: Product & Service Catalog");
  });

  it("leaves text that isn't a full citation alone", () => {
    const text = "See [KB Section x] and [KB Section 8] and [Section 8: Catalog].";
    const { container } = render(<ReplyText text={text} />);
    expect(container.querySelector(".cite")).toBeNull();
    expect(container.textContent).toBe(text);
  });
  it("keeps a numbered list's own starting number", () => {
    const { container } = render(<ReplyText text={"Next:\n3. Site visit\n4. Installation"} />);
    const list = container.querySelector("ol");
    expect(list.getAttribute("start")).toBe("3");
    expect([...list.querySelectorAll("li")].map((li) => li.textContent)).toEqual(["Site visit", "Installation"]);
    const fromOne = render(<ReplyText text={"1. Enquiry\n2. Assessment"} />).container.querySelector("ol");
    expect(fromOne.hasAttribute("start")).toBe(false);
  });

  it("turns '* ' lines into bullets, *x* into italics, and never shows a raw asterisk", () => {
    const text = "* First point\n* Second point\n**Bold** and *italic*, a stray * star and **unclosed";
    const { container } = render(<ReplyText text={text} />);
    expect([...container.querySelectorAll("ul li")].map((li) => li.textContent)).toEqual(["First point", "Second point"]);
    expect(container.querySelector("strong").textContent).toBe("Bold");
    expect(container.querySelector("em").textContent).toBe("italic");
    expect(container.textContent).not.toContain("*");
  });

  it("leaves underscores alone, so email addresses stay intact", () => {
    const text = "Write to rahul_k_sharma@gmail.com or _the team_ today.";
    const { container } = render(<ReplyText text={text} />);
    expect(container.textContent).toBe(text);
    expect(container.querySelector("em")).toBeNull();
  });

  it("treats '# ' as a heading only with a space after the hash", () => {
    const { container } = render(<ReplyText text={"# Steps\n#1 choice for homes"} />);
    const [heading, line] = container.querySelectorAll("p");
    expect(heading.className).toBe("reply-heading");
    expect(heading.textContent).toBe("Steps");
    expect(line.className).toBe("");
    expect(line.textContent).toBe("#1 choice for homes");
  });
});
