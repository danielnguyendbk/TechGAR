'use client';

import { useState } from 'react';
import { Activity, Camera, CarFront, CircleParking, Radar } from 'lucide-react';

import { AppHeader } from '@/components/app-header';
import { CameraPanel } from '@/components/camera-panel';
import { ConnectionBanner } from '@/components/connection-banner';
import { GateEditor } from '@/components/gate-editor';
import { MetricCard } from '@/components/metric-card';
import { ParkingMap } from '@/components/parking-map';
import { ResetIdDialog } from '@/components/reset-id-dialog';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useRuntimePolling } from '@/hooks/use-runtime-polling';
import { selectCounts, selectVisibleVehicles } from '@/lib/display/resolve';
import type { WorldPoint } from '@/lib/domain/types';
import { useParkingState } from '@/lib/stores/parking-store';

export function MonitorApp() {
  useRuntimePolling();
  const { snapshot, connection, trackingSource } = useParkingState();
  const [gateEditing, setGateEditing] = useState(false);
  const [gatePoints, setGatePoints] = useState<readonly WorldPoint[]>([]);
  const vehicles = selectVisibleVehicles(snapshot);
  const counts = selectCounts(snapshot);
  const latencyMs = Math.round((snapshot.latency.e2e ?? 0) * 1_000);

  return (
    <main className="min-h-screen bg-background text-foreground">
      <AppHeader connection={connection.state} />
      <ConnectionBanner connection={connection} />
      <div className="mx-auto grid max-w-[1500px] gap-4 px-4 py-4 lg:grid-cols-[minmax(0,1fr)_330px] lg:px-7 lg:py-6">
        <section className="space-y-4" aria-labelledby="monitor-map-title">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard icon={<CarFront />} label="Xe hiển thị" value={String(counts.vehicles).padStart(2, '0')} detail={`${vehicles.filter((vehicle) => vehicle.state === 'observed').length} đang được quan sát`} />
            <MetricCard icon={<CircleParking />} label="Ô còn trống" value={String(counts.empty).padStart(2, '0')} detail={`${Math.round((counts.empty / Math.max(counts.capacity, 1)) * 100)}% sức chứa`} />
            <MetricCard icon={<Camera />} label="Camera" value={`${Object.values(snapshot.cameras).filter((camera) => camera.online).length}/${Object.keys(snapshot.cameras).length}`} detail="Sức khỏe nguồn hình" />
            <MetricCard icon={<Activity />} label="E2E latency" value={String(latencyMs)} suffix="ms" detail={snapshot.overload ? 'Đang quá tải' : 'Trong ngưỡng vận hành'} />
          </div>

          <Card className="gap-0 border-0 bg-[#152019] py-0 text-white ring-0">
            <CardHeader className="border-b border-white/8 py-4 sm:grid-cols-[1fr_auto]">
              <div><CardTitle id="monitor-map-title" className="text-white">Bản đồ vận hành trực tiếp</CardTitle><CardDescription className="text-white/50">Global ID là nguồn chân lý · {trackingSource}</CardDescription></div>
              <div className="flex items-center gap-3 text-[11px] text-white/60"><span className="text-[#d7ff3f]">● Trống</span><span className="text-[#ff9b8c]">● Đã đỗ</span><span className="text-[#65b7ff]">● Đang chạy</span></div>
            </CardHeader>
            <CardContent className="p-0">
              <ParkingMap
                snapshot={snapshot}
                vehicles={vehicles}
                gateEditing={gateEditing}
                gatePoints={gatePoints}
                onGatePoint={(point) => setGatePoints((current) => current.length < 6 ? [...current, point] : current)}
              />
            </CardContent>
          </Card>

          <div className="grid gap-3 sm:grid-cols-2">
            {Object.entries(snapshot.cameras).map(([cameraId, health]) => <CameraPanel key={cameraId} cameraId={cameraId} health={health} />)}
          </div>
        </section>

        <aside className="space-y-4" aria-label="Điều khiển và sự kiện hệ thống">
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><Radar className="size-4 text-[#357a53]" /> Điều khiển vận hành</CardTitle><CardDescription>Thay đổi có xác nhận và phản hồi rõ ràng.</CardDescription></CardHeader>
            <CardContent className="space-y-5">
              <ResetIdDialog />
              <div className="border-t pt-5"><GateEditor editing={gateEditing} points={gatePoints} onEditingChange={setGateEditing} onPointsChange={setGatePoints} /></div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Sự kiện danh tính</CardTitle><CardDescription>{snapshot.identity_events.length} sự kiện trong snapshot #{snapshot.frame_index}</CardDescription></CardHeader>
            <CardContent className="space-y-4">
              {snapshot.identity_events.length === 0 && <p className="text-xs text-muted-foreground">Chưa có sự kiện mới.</p>}
              {snapshot.identity_events.slice(0, 8).map((event) => (
                <div key={event.event_id} className="grid grid-cols-[8px_1fr] gap-3">
                  <span className="mt-1.5 size-2 rounded-full bg-[#7ab68e]" />
                  <div><div className="flex items-baseline justify-between gap-2"><p className="text-xs font-medium">{event.type} · {event.global_id === null ? 'system' : `GID ${event.global_id}`}</p><time className="font-mono text-[9px] text-muted-foreground">{new Date(event.timestamp * 1_000).toLocaleTimeString('vi-VN')}</time></div><p className="mt-0.5 text-[11px] text-muted-foreground">{event.detail}</p></div>
                </div>
              ))}
            </CardContent>
          </Card>
        </aside>
      </div>
    </main>
  );
}
