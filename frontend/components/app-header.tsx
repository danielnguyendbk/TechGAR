'use client';

import Link from 'next/link';
import { CircleParking, Radio } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import type { ConnectionState } from '@/lib/domain/types';

const labels: Record<ConnectionState, string> = {
  connecting: 'Đang kết nối',
  live: 'Trực tuyến',
  stale: 'Dữ liệu trễ',
  error: 'Mất kết nối',
};

export function AppHeader({ connection = 'live' }: { connection?: ConnectionState }) {
  return (
    <header className="border-b border-white/8 bg-[#101712]/95 text-white">
      <div className="mx-auto flex max-w-[1500px] items-center justify-between gap-3 px-4 py-3 sm:px-7">
        <Link href="/" className="flex min-h-11 items-center gap-3 rounded-xl focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#d7ff3f]">
          <span className="grid size-10 place-items-center rounded-xl bg-[#d7ff3f] text-[#101712]">
            <CircleParking className="size-5" aria-hidden="true" />
          </span>
          <span>
            <strong className="block font-heading text-base font-semibold tracking-tight">TechGAR Control</strong>
            <span className="hidden text-xs text-white/55 sm:block">Global ID · Parking guidance</span>
          </span>
        </Link>
        <nav aria-label="Điều hướng chính" className="flex items-center gap-1 text-xs">
          <Link className="flex min-h-11 items-center rounded-lg px-3 text-white/65 transition hover:bg-white/8 hover:text-white" href="/">Tài xế</Link>
          <Link className="flex min-h-11 items-center rounded-lg px-3 text-white/65 transition hover:bg-white/8 hover:text-white" href="/monitor">Monitor</Link>
          <Link className="hidden min-h-11 items-center rounded-lg px-3 text-white/65 transition hover:bg-white/8 hover:text-white sm:flex" href="/kiosk/entry">Kiosk</Link>
          <Badge className={connection === 'live' ? 'ml-1 bg-[#d7ff3f] text-[#101712] hover:bg-[#d7ff3f]' : 'ml-1 bg-[#ffbe5c] text-[#30200a] hover:bg-[#ffbe5c]'}>
            <Radio className="size-3" aria-hidden="true" /> {labels[connection]}
          </Badge>
        </nav>
      </div>
    </header>
  );
}

