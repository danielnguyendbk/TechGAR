'use client';

import { useEffect, useState } from 'react';
import { Clock3, QrCode, ScanLine } from 'lucide-react';
import { QRCodeSVG } from 'qrcode.react';

import { AppHeader } from '@/components/app-header';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { sessionClient } from '@/lib/api/session-client';
import { demoSession } from '@/lib/demo-data';
import type { DriverSession } from '@/lib/domain/types';

export function KioskApp() {
  const [sessions, setSessions] = useState<readonly DriverSession[]>([{ ...demoSession, sessionId: 'S42' }]);
  const [origin] = useState(() => typeof window === 'undefined' ? 'http://localhost:3000' : window.location.origin);
  const [demo, setDemo] = useState(true);

  useEffect(() => {
    let active = true;
    sessionClient.waiting().then((next) => { if (active) { setSessions(next); setDemo(false); } }).catch(() => { if (active) setDemo(true); });
    return () => { active = false; };
  }, []);

  return (
    <main className="min-h-screen bg-[#101712] text-white">
      <AppHeader connection={demo ? 'stale' : 'live'} />
      <div className="mx-auto max-w-[1200px] px-5 py-10 sm:px-8">
        <div className="mb-8 text-center"><div className="mx-auto grid size-14 place-items-center rounded-2xl bg-[#d7ff3f] text-[#101712]"><ScanLine className="size-7" /></div><p className="mt-5 text-xs font-semibold uppercase tracking-[0.25em] text-[#d7ff3f]">Kiosk cổng vào</p><h1 className="mt-2 font-heading text-4xl font-semibold tracking-tight">Quét mã để bắt đầu đỗ xe</h1><p className="mx-auto mt-3 max-w-xl text-sm text-white/55">Mỗi QR là deep-link riêng cho một phiên. Ứng dụng chỉ hiển thị Global ID đã liên kết với phiên đó.</p></div>
        {demo && <output className="mx-auto mb-5 block max-w-xl rounded-xl bg-[#ffbe5c]/12 px-4 py-3 text-center text-xs text-[#ffdb9a]">Runtime chưa kết nối — đang hiển thị phiên mẫu S42.</output>}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {sessions.map((session) => {
            const url = `${origin}/?session=${encodeURIComponent(session.sessionId)}`;
            return <Card key={session.sessionId} className="border-white/10 bg-white/[.055] text-white ring-white/10"><CardHeader><div className="flex items-center justify-between"><span className="rounded-full bg-[#d7ff3f]/12 px-3 py-1 text-xs font-semibold text-[#d7ff3f]">{session.sessionId}</span><span className="flex items-center gap-1 text-xs text-white/45"><Clock3 className="size-3" /> Đang chờ</span></div><CardTitle className="mt-2 text-white">Xe GID {session.globalVehicleId ?? 'chưa gán'}</CardTitle><CardDescription className="text-white/50">Quét bằng camera điện thoại</CardDescription></CardHeader><CardContent><a href={url} aria-label={`Mở phiên ${session.sessionId}`} className="mx-auto grid w-fit min-h-44 min-w-44 place-items-center rounded-2xl bg-white p-4 text-[#101712] focus-visible:outline-4 focus-visible:outline-[#d7ff3f]"><QRCodeSVG value={url} size={160} level="M" title={`QR phiên ${session.sessionId}`} /></a><p className="mt-3 flex items-center justify-center gap-2 text-xs text-white/45"><QrCode className="size-4" /> Chỉ dùng cho một phiên</p></CardContent></Card>;
          })}
        </div>
      </div>
    </main>
  );
}
