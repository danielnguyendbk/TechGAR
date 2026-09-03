import type { DriverSession } from '@/lib/domain/types';

function isSession(value: unknown): value is DriverSession {
  if (typeof value !== 'object' || value === null) return false;
  const session = value as Record<string, unknown>;
  return typeof session.sessionId === 'string'
    && ['WAITING', 'NAVIGATING', 'PARKED', 'EXIT_NAVIGATION'].includes(String(session.state));
}

export class SessionEndedError extends Error {}

export class SessionClient {
  constructor(private readonly baseUrl = process.env.NEXT_PUBLIC_SESSION_URL ?? '') {}

  private async request(path: string, init?: RequestInit): Promise<DriverSession> {
    const headers = new Headers(init?.headers);
    headers.set('Content-Type', 'application/json');
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      cache: 'no-store',
      headers,
    });
    if (response.status === 404) throw new SessionEndedError('Phiên đã kết thúc');
    if (!response.ok) throw new Error(`Session API ${response.status}`);
    const data: unknown = await response.json();
    if (!isSession(data)) throw new Error('Session API trả về dữ liệu không hợp lệ');
    return data;
  }

  get(sessionId: string): Promise<DriverSession> {
    return this.request(`/api/sessions/${encodeURIComponent(sessionId)}`);
  }

  claim(sessionId: string): Promise<DriverSession> {
    return this.request(`/api/sessions/${encodeURIComponent(sessionId)}/claim`, {
      method: 'POST', body: '{}',
    });
  }

  selectSpot(sessionId: string, spotId: string): Promise<DriverSession> {
    return this.request(`/api/sessions/${encodeURIComponent(sessionId)}/select-spot`, {
      method: 'POST', body: JSON.stringify({ spot_id: spotId }),
    });
  }

  exit(sessionId: string): Promise<DriverSession> {
    return this.request(`/api/sessions/${encodeURIComponent(sessionId)}/exit`, {
      method: 'POST', body: '{}',
    });
  }

  async waiting(): Promise<DriverSession[]> {
    const response = await fetch(`${this.baseUrl}/api/sessions/waiting`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Session API ${response.status}`);
    const data: unknown = await response.json();
    if (!Array.isArray(data) || !data.every(isSession)) {
      throw new Error('Danh sách phiên không hợp lệ');
    }
    return data;
  }
}

export const sessionClient = new SessionClient();
