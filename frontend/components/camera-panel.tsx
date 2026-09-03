import { CameraOff, Radio } from 'lucide-react';

import type { CameraHealth } from '@/lib/domain/types';

export function CameraPanel({ cameraId, health }: { cameraId: string; health: CameraHealth }) {
  return (
    <section aria-label={`Video trực tuyến ${cameraId}`} className="overflow-hidden rounded-xl bg-[#101712] text-white">
      <div className="flex items-center justify-between border-b border-white/8 px-3 py-2 text-xs">
        <strong>{cameraId}</strong>
        <span className={health.online ? 'flex items-center gap-1 text-[#d7ff3f]' : 'flex items-center gap-1 text-[#ff9b8c]'}>
          <Radio className="size-3" /> {health.online ? 'Online' : 'Offline'}
        </span>
      </div>
      {health.online ? (
        <div className="relative aspect-video bg-[radial-gradient(circle_at_60%_25%,#344a3b,#131d16_62%)]">
          {/* MJPEG is a live stream and cannot be optimized as a static Next image. */}
          {/* oxlint-disable-next-line next/no-img-element */}
          <img src={`/api/runtime/cameras/${cameraId}.mjpg`} alt={`Luồng camera ${cameraId}`} className="size-full object-cover opacity-65" />
          <div className="absolute inset-x-4 bottom-4 h-16 skew-x-[-16deg] border border-[#d7ff3f]/25 bg-[#d7ff3f]/5" />
        </div>
      ) : (
        <div className="grid aspect-video place-items-center bg-[#1b241e] text-center text-white/55">
          <div><CameraOff className="mx-auto mb-2 size-6" /><p className="text-xs font-medium">Mất tín hiệu</p></div>
        </div>
      )}
    </section>
  );
}
