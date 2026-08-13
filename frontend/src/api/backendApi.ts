/**
 * Backend API client for TechGAR.
 *
 * Review fixes:
 *   #19 — No 127.0.0.1 fallback in production
 *   #17 — Frontend calls API, not reading JSON files
 *   #26 — No shared filesystem dependency
 */

const getBaseUrl = (): string => {
  // Use VITE_BACKEND_URL if available, otherwise use relative path (same-origin proxy)
  const envUrl = import.meta.env.VITE_BACKEND_URL;
  if (envUrl) return envUrl;
  // In development, Vite proxy handles /api → backend
  return "";
};

const BASE = getBaseUrl();

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    throw new Error(`API ${path}: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string, body: object): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`API ${path}: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ── Session API ──

export async function getSession(sessionId: string) {
  return fetchJson<any>(`/api/session/${sessionId}`);
}

export async function getWaitingSessions(gateId?: string) {
  const query = gateId ? `?gate_id=${gateId}` : "";
  return fetchJson<any[]>(`/api/sessions/waiting${query}`);
}

export async function claimSession(sessionId: string) {
  return postJson<any>("/api/session/claim", { sessionId });
}

export async function selectSpot(sessionId: string, spotId: string | null) {
  return postJson<any>("/api/session/select", { sessionId, spotId });
}

export async function startExit(sessionId: string) {
  return postJson<any>("/api/session/exit", { sessionId });
}

// ── Parking API ──

export async function getParkingStatus() {
  return fetchJson<any>("/api/parking");
}

// ── Vehicle API ──

export async function getAllVehicles() {
  return fetchJson<any>("/api/vehicles");
}
