import { useEffect, useRef, useState } from "react";

import { ApiError, GENERIC_ERROR, getLead, getLeads, getSupportRequests } from "../api.js";
import { WAKE_NOTICE_AFTER_MS } from "../useWelcome.js";

/**
 * The dashboard's data from the sales API. Filtering, search, sorting and paging all happen on the
 * server, so they stay right however many leads there are. An answer that arrives after the request
 * changed (a new filter, search or Refresh) is dropped. A 401 means the key stopped working: the hook
 * calls onUnauthorized and the dashboard goes back to the unlock card.
 */

export const PAGE_SIZE = 50;

/** True once `active` has stayed true for `ms` (shows the "waking up the server" notice). */
export function useSlow(active, ms = WAKE_NOTICE_AFTER_MS) {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    setSlow(false);
    if (!active) return undefined;
    const timer = setTimeout(() => setSlow(true), ms);
    return () => clearTimeout(timer);
  }, [active, ms]);
  return slow;
}

/** `value`, but only after it has stopped changing for `ms` (search as you type, one request). */
export function useDebounced(value, ms) {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return settled;
}

const asApiError = (error) => (error instanceof ApiError ? error : new ApiError("server", GENERIC_ERROR));
const FIRST_LOAD = { status: "loading", body: null, items: [], error: null, loadingMore: false, moreError: null };

/**
 * A paged list. load(offset) returns a promise of {items, total, ...}. The first page loads when
 * `requestKey` (a string of everything the request depends on) changes; more() appends the next page.
 */
function usePagedList(load, requestKey, onUnauthorized) {
  const [state, setState] = useState(FIRST_LOAD);
  const [attempt, setAttempt] = useState(0);
  const latest = useRef(0); // the newest first-page request; answers to older ones are dropped
  const loadRef = useRef(load);
  const unauthorizedRef = useRef(onUnauthorized);
  loadRef.current = load;
  unauthorizedRef.current = onUnauthorized;

  useEffect(() => {
    const id = ++latest.current;
    setState((current) => ({ ...current, status: "loading", error: null, moreError: null }));
    loadRef.current(0).then(
      (body) => id === latest.current && setState({ ...FIRST_LOAD, status: "ready", body, items: body.items }),
      (error) => {
        if (id !== latest.current) return;
        if (error?.kind === "unauthorized") unauthorizedRef.current();
        else setState((current) => ({ ...current, status: "error", error: asApiError(error) }));
      },
    );
  }, [requestKey, attempt]);

  function more() {
    if (state.status !== "ready" || state.loadingMore || state.items.length >= state.body.total) return;
    const id = latest.current; // a new filter or search while this loads makes it stale
    setState((current) => ({ ...current, loadingMore: true, moreError: null }));
    loadRef.current(state.items.length).then(
      (body) =>
        id === latest.current &&
        setState((current) => ({ ...current, body, items: [...current.items, ...body.items], loadingMore: false })),
      (error) => {
        if (id !== latest.current) return;
        if (error?.kind === "unauthorized") unauthorizedRef.current();
        else setState((current) => ({ ...current, loadingMore: false, moreError: asApiError(error) }));
      },
    );
  }

  const total = state.body?.total ?? 0;
  return { ...state, total, hasMore: state.items.length < total, more, reload: () => setAttempt((n) => n + 1) };
}

/** The leads list for {status, q, sort}, plus `counts` (leads per status in the whole table). */
export function useLeadList(key, { status, q, sort }, refreshTick, onUnauthorized) {
  const load = (offset) => getLeads({ status, q, sort, limit: PAGE_SIZE, offset }, key);
  const list = usePagedList(load, JSON.stringify([key, status, q, sort, refreshTick]), onUnauthorized);
  return { ...list, counts: list.body?.counts ?? null };
}

/** The support requests, newest first. */
export function useSupportList(key, refreshTick, onUnauthorized) {
  const load = (offset) => getSupportRequests({ limit: PAGE_SIZE, offset }, key);
  return usePagedList(load, JSON.stringify([key, refreshTick]), onUnauthorized);
}

/** One lead with every field (GET /api/leads/{id}); status "idle" while no lead is selected. */
export function useLeadDetail(key, leadId, refreshTick, onUnauthorized) {
  const [state, setState] = useState({ status: "idle", lead: null, error: null });
  const [attempt, setAttempt] = useState(0);
  const unauthorizedRef = useRef(onUnauthorized);
  unauthorizedRef.current = onUnauthorized;

  useEffect(() => {
    if (leadId == null) {
      setState({ status: "idle", lead: null, error: null });
      return undefined;
    }
    let active = true;
    // While the same lead refreshes, keep showing it; a different lead starts empty
    setState((current) => ({ status: "loading", lead: current.lead?.id === leadId ? current.lead : null, error: null }));
    getLead(leadId, key).then(
      (lead) => active && setState({ status: "ready", lead, error: null }),
      (error) => {
        if (!active) return;
        if (error?.kind === "unauthorized") unauthorizedRef.current();
        else setState({ status: "error", lead: null, error: asApiError(error) });
      },
    );
    return () => {
      active = false;
    };
  }, [key, leadId, refreshTick, attempt]);

  return { ...state, retry: () => setAttempt((n) => n + 1) };
}
