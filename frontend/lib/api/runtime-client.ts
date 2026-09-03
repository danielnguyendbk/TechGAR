import { validateRuntimeSnapshot } from '@/lib/domain/schema';
import type { RuntimeSnapshot, WorldPoint } from '@/lib/domain/types';

export interface FetchLike {
  (input: string | URL | Request, init?: RequestInit): Promise<Response>;
}

export interface ResetResult {
  readonly reset: true;
  readonly retired_identities: number;
  readonly include_sessions: boolean;
}

export class RuntimeClient {
  private inFlight: Promise<RuntimeSnapshot> | null = null;
  private failures = 0;

  constructor(
    private readonly baseUrl = process.env.NEXT_PUBLIC_RUNTIME_URL ?? '',
    private readonly fetcher: FetchLike = (input, init) => globalThis.fetch(input, init),
  ) {}

  fetchSnapshot(previousFrameIndex: number | null, signal?: AbortSignal): Promise<RuntimeSnapshot> {
    if (this.inFlight) return this.inFlight;
    const request = this.fetcher(`${this.baseUrl}/api/runtime/snapshot`, {
      cache: 'no-store',
      signal,
      headers: { Accept: 'application/json' },
    }).then(async (response) => {
      if (!response.ok) throw new Error(`Runtime API ${response.status}`);
      return validateRuntimeSnapshot(await response.json(), previousFrameIndex);
    }).then((snapshot) => {
      this.failures = 0;
      return snapshot;
    }).catch((error: unknown) => {
      this.failures += 1;
      throw error;
    }).finally(() => {
      this.inFlight = null;
    });
    this.inFlight = request;
    return request;
  }

  nextDelay(): number {
    return Math.min(5_000, 1_000 * 2 ** Math.max(0, this.failures - 1));
  }

  async resetIdentities(includeSessions: boolean): Promise<ResetResult> {
    const response = await this.fetcher(`${this.baseUrl}/api/runtime/reset-identities`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ include_sessions: includeSessions }),
    });
    if (!response.ok) throw new Error(`Reset ID thất bại (${response.status})`);
    const data = await response.json() as Partial<ResetResult>;
    if (data.reset !== true || typeof data.retired_identities !== 'number' || !Number.isInteger(data.retired_identities)) {
      throw new Error('Reset ID trả về dữ liệu không hợp lệ');
    }
    return {
      reset: true,
      retired_identities: data.retired_identities,
      include_sessions: data.include_sessions === true,
    };
  }

  async saveGates(points: readonly WorldPoint[]): Promise<void> {
    if (points.length !== 6) throw new Error('Cấu hình cổng cần đúng 6 điểm');
    const response = await this.fetcher(`${this.baseUrl}/api/runtime/gates`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ points }),
    });
    if (!response.ok) throw new Error(`Lưu cổng thất bại (${response.status})`);
  }
}

export const runtimeClient = new RuntimeClient();
