'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, CarFront, CheckCircle2, CircleParking, Navigation, Route, Volume2 } from 'lucide-react';

import { AppHeader } from '@/components/app-header';
import { ConnectionBanner } from '@/components/connection-banner';
import { ParkingMap } from '@/components/parking-map';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useDriverSession } from '@/hooks/use-driver-session';
import { useRuntimePolling } from '@/hooks/use-runtime-polling';
import { useVoiceGuidance } from '@/hooks/use-voice-guidance';
import { useWebMcp } from '@/hooks/use-webmcp';
import { selectCounts, selectSessionVehicle } from '@/lib/display/resolve';
import type { WorldPoint } from '@/lib/domain/types';
import { buildInstructions } from '@/lib/guidance/instructions';
import { isOffRoute } from '@/lib/guidance/off-route';
import { routeToSlot } from '@/lib/routing/lane-graph';
import { useParkingState } from '@/lib/stores/parking-store';

export function DriverApp() {
  useRuntimePolling();
  const { snapshot, connection, trackingSource } = useParkingState();
  const { session, ended, loading, error, selectSpot, requestExit } = useDriverSession();
  const [candidate, setCandidate] = useState<string | null>(null);
  const [confirmedRoute, setConfirmedRoute] = useState<readonly WorldPoint[]>([]);
  const routeOwner = useRef<string | null>(null);
  const vehicle = session ? selectSessionVehicle(snapshot, session) : null;
  const confirmedSpot = session?.targetSpotId ?? session?.parkedSpotId ?? null;
  useEffect(() => {
    if (!session || !confirmedSpot) {
      if (routeOwner.current !== null) {
        routeOwner.current = null;
        setConfirmedRoute([]);
      }
      return;
    }
    // A route is a confirmed decision, not a derived value that may silently
    // change whenever the next vehicle snapshot arrives.  Wait for a real/replay
    // snapshot, capture it once, then use the frozen polyline for off-route checks.
    if (!vehicle || trackingSource === 'demo') return;
    const owner = `${session.sessionId}:${confirmedSpot}`;
    if (routeOwner.current === owner) return;
    routeOwner.current = owner;
    setConfirmedRoute(routeToSlot(snapshot, vehicle.position, confirmedSpot));
  }, [confirmedSpot, session, snapshot, trackingSource, vehicle]);
  const route = confirmedSpot ? confirmedRoute : [];
  const instructions = buildInstructions(route);
  const offRoute = Boolean(vehicle && route.length > 1 && isOffRoute(vehicle.position, route));
  useVoiceGuidance(offRoute ? 'Bạn đang lệch tuyến. Hãy dừng ở vị trí an toàn và kiểm tra bản đồ.' : instructions[0]?.text ?? null);
  const counts = selectCounts(snapshot);
  const available = snapshot.parking_slots.filter((slot) => !slot.occupied);

  const tools = useMemo(() => [{
    name: 'get_parking_status',
    title: 'Đọc trạng thái bãi xe',
    description: 'Trả về số ô trống và trạng thái phiên tài xế hiện tại.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    execute: () => ({ emptySpots: counts.empty, capacity: counts.capacity, sessionState: session?.state ?? 'UNKNOWN', targetSpotId: confirmedSpot }),
  }, {
    name: 'select_parking_spot',
    title: 'Chọn ô đỗ xe',
    description: 'Chọn một ô trống cho phiên tài xế hiện tại và bắt đầu tuyến điều hướng.',
    inputSchema: { type: 'object', properties: { spotId: { type: 'string' } }, required: ['spotId'], additionalProperties: false },
    execute: async (input: unknown) => {
      if (typeof input !== 'object' || input === null || typeof (input as { spotId?: unknown }).spotId !== 'string') throw new Error('spotId không hợp lệ');
      const spotId = (input as { spotId: string }).spotId;
      if (!available.some((spot) => spot.slot_id === spotId)) throw new Error('Ô đỗ không còn trống');
      await selectSpot(spotId);
      return { selected: spotId, state: 'NAVIGATING' };
    },
  }], [available, confirmedSpot, counts.capacity, counts.empty, selectSpot, session?.state]);
  useWebMcp(tools);

  if (ended) return <SessionEnded />;
  if (loading || !session) return <main className="grid min-h-screen place-items-center bg-background"><output>Đang tải phiên đỗ xe…</output></main>;

  async function confirmSelection(): Promise<void> {
    if (!candidate) return;
    await selectSpot(candidate);
    setCandidate(null);
  }

  const parked = session.state === 'PARKED';
  return (
    <main className="min-h-screen bg-background text-foreground">
      <AppHeader connection={connection.state} />
      <ConnectionBanner connection={connection} />
      <div className="mx-auto max-w-[1280px] px-4 py-5 sm:px-7 sm:py-8">
        <div className="mb-5 flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
          <div><p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#357a53]">Phiên {session.sessionId} · GID {session.globalVehicleId ?? '—'}</p><h1 className="mt-1 font-heading text-3xl font-semibold tracking-tight">{parked ? 'Xe của bạn đã đỗ an toàn' : confirmedSpot ? `Đi đến ô ${confirmedSpot}` : 'Chọn ô đỗ phù hợp'}</h1></div>
          <div className="flex items-center gap-3 rounded-xl bg-[#eef7ef] px-4 py-3 text-sm text-[#285c3e]"><CircleParking className="size-5" /><strong>{counts.empty} ô trống</strong><span className="text-[#285c3e]/55">/ {counts.capacity}</span></div>
        </div>
        {error && <output className="mb-4 block rounded-xl bg-[#fff5d8] px-4 py-3 text-sm text-[#745413]">{error} — đang dùng dữ liệu demo an toàn.</output>}
        {offRoute && <p role="alert" className="mb-4 rounded-xl bg-[#fff0e9] px-4 py-3 text-sm font-medium text-[#943d2f]">ĐANG ĐI SAI TUYẾN — hệ thống không tự đổi đường; hãy kiểm tra vị trí trước khi tiếp tục.</p>}

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
          <ParkingMap snapshot={snapshot} vehicles={vehicle ? [vehicle] : []} selectedSlotId={candidate ?? confirmedSpot} route={route} onSelectSlot={parked || confirmedSpot ? undefined : (spotId) => { if (available.some((spot) => spot.slot_id === spotId)) setCandidate(spotId); }} />
          <aside className="space-y-4">
            {parked ? (
              <Card className="border-[#7ab68e]/40 bg-[#eef7ef]">
                <CardHeader><CheckCircle2 className="mb-2 size-9 text-[#357a53]" /><CardTitle>Đã đỗ tại {session.parkedSpotId ?? confirmedSpot}</CardTitle><CardDescription>Marker vẫn neo vào tâm ô dù camera tạm mất dấu.</CardDescription></CardHeader>
                <CardContent><Button className="w-full" onClick={() => void requestExit()}><Navigation /> Bắt đầu chỉ đường ra</Button></CardContent>
              </Card>
            ) : confirmedSpot ? (
              <Card>
                <CardHeader><Route className="mb-2 size-7 text-[#357a53]" /><CardTitle>Đường đến {confirmedSpot}</CardTitle><CardDescription>Tuyến chỉ xuất hiện sau khi bạn xác nhận ô.</CardDescription></CardHeader>
                <CardContent className="space-y-3">
                  {instructions.map((instruction, index) => <div key={`${instruction.text}-${index}`} className="flex gap-3 rounded-xl bg-muted p-3 text-sm"><span className="grid size-7 shrink-0 place-items-center rounded-full bg-[#152019] text-xs text-white">{index + 1}</span><span>{instruction.text}</span></div>)}
                  <p className="flex items-center gap-2 text-xs text-muted-foreground"><Volume2 className="size-4" /> Hướng dẫn giọng nói phát một lần cho mỗi chỉ dẫn.</p>
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardHeader><CarFront className="mb-2 size-7 text-[#357a53]" /><CardTitle>Ô trống gần bạn</CardTitle><CardDescription>Chạm bản đồ hoặc chọn trong danh sách. Chưa có tuyến nào được tạo.</CardDescription></CardHeader>
                <CardContent className="grid max-h-[410px] grid-cols-2 gap-2 overflow-auto">
                  {available.map((spot) => <Button key={spot.slot_id} variant={candidate === spot.slot_id ? 'default' : 'outline'} className="min-h-12" onClick={() => setCandidate(spot.slot_id)}>{spot.slot_id}</Button>)}
                </CardContent>
              </Card>
            )}
          </aside>
        </div>
      </div>

      {candidate && !confirmedSpot && (
        <div className="fixed inset-x-0 bottom-0 z-30 border-t bg-background/95 p-4 shadow-[0_-12px_40px_rgba(15,24,17,.16)] backdrop-blur" data-testid="spot-confirmation">
          <div className="mx-auto flex max-w-[760px] flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"><div><p className="text-xs text-muted-foreground">Ô đang chọn</p><p className="text-lg font-semibold">{candidate} · đang trống</p></div><div className="flex gap-2"><Button variant="outline" onClick={() => setCandidate(null)}>Chọn lại</Button><Button onClick={() => void confirmSelection()}>Xác nhận ô {candidate}<ArrowRight /></Button></div></div>
        </div>
      )}
    </main>
  );
}

function SessionEnded() {
  return <main className="grid min-h-screen place-items-center bg-background p-5"><Card className="max-w-md text-center"><CardHeader><CheckCircle2 className="mx-auto mb-2 size-10 text-[#357a53]" /><CardTitle>Phiên đỗ xe đã kết thúc</CardTitle><CardDescription>Liên kết này không còn điều khiển xe. Quét mã mới tại kiosk nếu bạn quay lại.</CardDescription></CardHeader></Card></main>;
}
