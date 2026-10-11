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
    expect(tag.firstChild.textContent).toBe("KB Section 8");
    expect(tag.textContent).toBe("KB Section 8: Product & Service Catalog"); // the title is there for screen readers
  });

  it("leaves text that isn't a full citation alone", () => {
    const text = "See [KB Section x] and [KB Section 8] and [Section 8: Catalog].";
    const { container } = render(<ReplyText text={text} />);
    expect(container.querySelector(".cite")).toBeNull();
    expect(container.textContent).toBe(text);
  });
});
