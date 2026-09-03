'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import { SessionEndedError, sessionClient } from '@/lib/api/session-client';
import { demoSession } from '@/lib/demo-data';
import type { DriverSession } from '@/lib/domain/types';

const claimedSessions = new Set<string>();

export interface DriverSessionState {
  readonly session: DriverSession | null;
  readonly ended: boolean;
  readonly loading: boolean;
  readonly error: string | null;
  readonly selectSpot: (spotId: string) => Promise<void>;
  readonly requestExit: () => Promise<void>;
}

export function useDriverSession(): DriverSessionState {
  const [session, setSession] = useState<DriverSession | null>(null);
  const [ended, setEnded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useRef<string>('S42');
  const demo = useRef(false);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const query = new URLSearchParams(window.location.search);
    sessionId.current = query.get('session') ?? 'S42';
    demo.current = !query.has('session') || query.has('fixture');
    const fixture = query.get('fixture');

    async function start(): Promise<void> {
      if (demo.current) {
        if (active) {
          const base = { ...demoSession, sessionId: sessionId.current };
          setSession(fixture === 'parked-long'
            ? { ...base, state: 'PARKED', targetSpotId: 'D09', parkedSpotId: 'D09' }
            : fixture === 'parked-fallback'
              ? { ...base, state: 'PARKED', targetSpotId: 'B04', parkedSpotId: 'B04' }
              : fixture === 'driver-isolation'
                ? { ...base, state: 'NAVIGATING', targetSpotId: 'D02' }
                : fixture === 'off-route'
                  ? { ...base, state: 'NAVIGATING', targetSpotId: 'D08' }
                  : base);
          setLoading(false);
        }
        return;
      }
      try {
        let next: DriverSession;
        if (!claimedSessions.has(sessionId.current)) {
          claimedSessions.add(sessionId.current);
          next = await sessionClient.claim(sessionId.current);
        } else {
          next = await sessionClient.get(sessionId.current);
        }
        if (active) setSession(next);
      } catch (caught) {
        if (caught instanceof SessionEndedError) {
          if (active) setEnded(true);
        } else if (active) {
          setError(caught instanceof Error ? caught.message : 'Không tải được phiên');
          setSession({ ...demoSession, sessionId: sessionId.current });
          demo.current = true;
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    async function refresh(): Promise<void> {
      if (!active || demo.current) return;
      try {
        setSession(await sessionClient.get(sessionId.current));
        setError(null);
      } catch (caught) {
        if (caught instanceof SessionEndedError) setEnded(true);
        else setError(caught instanceof Error ? caught.message : 'Không tải được phiên');
      }
      if (active && !ended) timer = setTimeout(refresh, 500);
    }

    void start().then(() => {
      if (active && !demo.current) timer = setTimeout(refresh, 500);
    });
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [ended]);

  const selectSpot = useCallback(async (spotId: string) => {
    if (!session) return;
    if (demo.current) {
      setSession({ ...session, state: 'NAVIGATING', targetSpotId: spotId, updatedAt: Date.now() / 1_000 });
      return;
    }
    setSession(await sessionClient.selectSpot(session.sessionId, spotId));
  }, [session]);

  const requestExit = useCallback(async () => {
    if (!session) return;
    if (demo.current) {
      setSession({ ...session, state: 'EXIT_NAVIGATION', updatedAt: Date.now() / 1_000 });
      return;
    }
    setSession(await sessionClient.exit(session.sessionId));
  }, [session]);

  return { session, ended, loading, error, selectSpot, requestExit };
}
