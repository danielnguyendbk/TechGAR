import { CloudOff, TriangleAlert } from 'lucide-react';

import type { ConnectionStatus } from '@/lib/domain/types';

export function ConnectionBanner({ connection }: { connection: ConnectionStatus }) {
  if (connection.state === 'live' || connection.state === 'connecting') return null;
  return (
    <output
      aria-live="polite"
      className={connection.state === 'error'
        ? 'flex items-center justify-center gap-2 bg-[#fff0e9] px-4 py-2 text-xs font-medium text-[#943d2f]'
        : 'flex items-center justify-center gap-2 bg-[#fff5d8] px-4 py-2 text-xs font-medium text-[#745413]'}
    >
      {connection.state === 'error' ? <CloudOff className="size-4" /> : <TriangleAlert className="size-4" />}
      {connection.state === 'error'
        ? 'Mất kết nối Runtime API — đang giữ dữ liệu gần nhất trên bản đồ.'
        : 'Snapshot đang trễ hơn 5 giây — trạng thái cũ vẫn được giữ nguyên.'}
    </output>
  );
}
