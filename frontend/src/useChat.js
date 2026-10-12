import { useEffect, useRef, useState } from "react";

import { ApiError, GENERIC_ERROR, postChat } from "./api.js";
import { clearChat, loadChat, saveChat } from "./storage.js";

// After this long without a reply, say it is still working (T13a: neutral words). A tab left idle for 15+ minutes
// finds the server asleep again, and no new health ping is sent then.
export const SLOW_AFTER_MS = 8000;
export const STILL_WORKING = "Still working on it. This can take up to a minute.";

const EMPTY_CHAT = { sessionId: null, messages: [], lead: null };
let lastId = 0;
const newId = () => `${Date.now()}-${++lastId}`;

/**
 * The conversation: messages, the lead summary and the session id (kept in sessionStorage), plus
 * the sending state. One message at a time. On any failure the customer's text goes back into the
 * draft, so nothing they typed is lost.
 */
export function useChat() {
  const [chat, setChat] = useState(() => loadChat() ?? EMPTY_CHAT);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [slow, setSlow] = useState(false);
  const [notice, setNotice] = useState(null); // {kind: "error" | "rate_limit", text}
  const [waitLeft, setWaitLeft] = useState(0); // seconds until sending is allowed again (429)
  const generation = useRef(0); // bumped by newChat, so a late reply for the old chat is dropped

  useEffect(() => saveChat(chat), [chat]);

  useEffect(() => {
    if (waitLeft <= 0) return undefined;
    const timer = setTimeout(() => setWaitLeft((seconds) => seconds - 1), 1000);
    return () => clearTimeout(timer);
  }, [waitLeft]);

  /** Send one message (typed or a starter question). Ignored while waiting or when empty. */
  async function send(text) {
    const message = text.trim();
    if (!message || pending || waitLeft > 0) return;
    const gen = generation.current;
    const userMessage = { id: newId(), role: "user", text: message, at: new Date().toISOString() };
    setChat((c) => ({ ...c, messages: [...c.messages, userMessage] }));
    setDraft((d) => (d.trim() === message ? "" : d));
    setPending(true);
    setSlow(false);
    setNotice(null);
    const slowTimer = setTimeout(() => setSlow(true), SLOW_AFTER_MS);
    const started = performance.now();
    try {
      const body = await postChat({ message, sessionId: chat.sessionId });
      if (gen !== generation.current) return;
      const reply = {
        id: newId(),
        role: "assistant",
        text: body.reply,
        sources: Array.isArray(body.sources) ? body.sources : [],
        steps: Array.isArray(body.steps) ? body.steps : [],
        degraded: Boolean(body.degraded),
        elapsedMs: Math.round(performance.now() - started),
        at: new Date().toISOString(),
      };
      setChat((c) => ({ sessionId: body.session_id, messages: [...c.messages, reply], lead: body.lead ?? null }));
    } catch (error) {
      if (gen !== generation.current) return;
      const failure = error instanceof ApiError ? error : new ApiError("server", GENERIC_ERROR);
      setChat((c) => ({
        ...c,
        messages: c.messages.filter((m) => m.id !== userMessage.id),
        // An id the server refuses can't be reused: the next message starts a new chat.
        sessionId: failure.field === "session_id" ? null : c.sessionId,
      }));
      setDraft((d) => (d.trim() ? d : message));
      setNotice({ kind: failure.kind === "rate_limit" ? "rate_limit" : "error", text: failure.message });
      if (failure.kind === "rate_limit") setWaitLeft(failure.retryAfter);
    } finally {
      clearTimeout(slowTimer);
      if (gen === generation.current) {
        setPending(false);
        setSlow(false);
      }
    }
  }

  /** Forget this conversation and start a fresh one (a 429 wait still applies: it is per network). */
  function newChat() {
    generation.current += 1;
    clearChat();
    setChat(EMPTY_CHAT);
    setPending(false);
    setSlow(false);
    if (notice?.kind !== "rate_limit") setNotice(null);
  }

  return { ...chat, draft, setDraft, pending, slow, notice, waitLeft, send, newChat };
}
