'use client';

import { useEffect } from 'react';

import { runtimeClient } from '@/lib/api/runtime-client';
import { validateRuntimeSnapshot } from '@/lib/domain/schema';
import { parkingStore } from '@/lib/stores/parking-store';

const FIXTURES = new Set([
  'normal', 'flicker-gap', 'ghost', 'parked-long', 'parked-fallback',
  'driver-isolation', 'off-route', 'post-reset', 'offline',
]);

export function useRuntimePolling(): void {
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;
    const fixture = new URLSearchParams(window.location.search).get('fixture');

    async function replay(name: string): Promise<void> {
      parkingStore.setSource('replay');
      const stepDelay = name === 'ghost' || name === 'off-route' ? 300 : 90;
      try {
        const response = await fetch(`/fixtures/${name}.json`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`Không tải được fixture ${name}`);
        const raw: unknown = await response.json();
        if (!Array.isArray(raw)) throw new Error(`Fixture ${name} không phải chuỗi snapshot`);
        let previous: number | null = null;
        for (const item of raw) {
          if (stopped) return;
          const snapshot = validateRuntimeSnapshot(item, previous);
          previous = snapshot.frame_index;
          parkingStore.commit(snapshot, snapshot.published_at);
          parkingStore.setSource('replay');
          await new Promise((resolve) => { timer = setTimeout(resolve, stepDelay); });
        }
        if (name === 'offline') parkingStore.fail(new Error('Mất kết nối Runtime API'));
      } catch (error) {
        parkingStore.fail(error);
      }
    }

    async function poll(): Promise<void> {
      if (stopped) return;
      try {
        const previous = parkingStore.getState().snapshot.frame_index;
        const snapshot = await runtimeClient.fetchSnapshot(previous, controller.signal);
        parkingStore.commit(snapshot);
      } catch (error) {
        if (!controller.signal.aborted) parkingStore.fail(error);
      }
      if (!stopped) timer = setTimeout(poll, runtimeClient.nextDelay());
    }

    if (fixture && FIXTURES.has(fixture)) void replay(fixture);
    else void poll();
    return () => {
      stopped = true;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, []);
}
