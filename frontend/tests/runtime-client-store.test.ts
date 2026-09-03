import { describe, expect, it, vi } from 'vitest';

import { RuntimeClient, type FetchLike } from '@/lib/api/runtime-client';
import { makeDemoSnapshot } from '@/lib/demo-data';
import { ParkingStore } from '@/lib/stores/parking-store';

describe('runtime client and store resilience', () => {
  it('coalesces overlapping snapshot fetches', async () => {
    let release: ((response: Response) => void) | undefined;
    const pending = new Promise<Response>((resolve) => { release = resolve; });
    const fetcher = vi.fn(() => pending) as unknown as FetchLike;
    const client = new RuntimeClient('', fetcher);
    const first = client.fetchSnapshot(null);
    const second = client.fetchSnapshot(null);
    expect(first).toBe(second);
    expect(fetcher).toHaveBeenCalledTimes(1);
    release?.(new Response(JSON.stringify(makeDemoSnapshot()), { status: 200 }));
    await expect(first).resolves.toMatchObject({ schema_version: '1.0' });
  });

  it('backs off 1, 2, 4, then caps at 5 seconds', async () => {
    const fetcher: FetchLike = async () => new Response('', { status: 503 });
    const client = new RuntimeClient('', fetcher);
    expect(client.nextDelay()).toBe(1_000);
    for (const expected of [1_000, 2_000, 4_000, 5_000]) {
      await expect(client.fetchSnapshot(null)).rejects.toThrow();
      expect(client.nextDelay()).toBe(expected);
    }
  });

  it('keeps the last snapshot on a network failure and clears markers locally on reset', () => {
    const initial = makeDemoSnapshot();
    const store = new ParkingStore(initial);
    store.fail(new Error('offline'));
    expect(store.getState().snapshot).toBe(initial);
    expect(store.getState().connection.state).toBe('error');
    store.resetLocal();
    expect(store.getState().snapshot.vehicles).toHaveLength(0);
  });
});
