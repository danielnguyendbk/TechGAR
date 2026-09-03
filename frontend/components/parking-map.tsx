'use client';

/* oxlint-disable jsx-a11y/prefer-tag-over-role -- SVG has no native button/img equivalents. */

import { useEffect, useMemo, useRef, useState } from 'react';
import { CarFront, LocateFixed, LockKeyhole, Minus, Plus } from 'lucide-react';

import { Button } from '@/components/ui/button';
import type { DisplayVehicle, RuntimeSnapshot, WorldPoint } from '@/lib/domain/types';
import { shouldAnimateMarker } from '@/lib/display/marker-motion';
import { fitSlotProjection, type AffineTransform, type SvgPoint } from '@/lib/projection/affine';
import { PARKING_GEOMETRY, VIEWBOX } from '@/lib/projection/parking-geometry';

interface ParkingMapProps {
  readonly snapshot: RuntimeSnapshot;
  readonly vehicles: readonly DisplayVehicle[];
  readonly selectedSlotId?: string | null;
  readonly route?: readonly WorldPoint[];
  readonly onSelectSlot?: (slotId: string) => void;
  readonly gateEditing?: boolean;
  readonly gatePoints?: readonly WorldPoint[];
  readonly onGatePoint?: (point: WorldPoint) => void;
  readonly compact?: boolean;
}

function pointsAttribute(points: readonly SvgPoint[]): string {
  return points.map((point) => `${point[0]},${point[1]}`).join(' ');
}

export function ParkingMap({
  snapshot,
  vehicles,
  selectedSlotId = null,
  route = [],
  onSelectSlot,
  gateEditing = false,
  gatePoints = [],
  onGatePoint,
  compact = false,
}: ParkingMapProps) {
  const [zoom, setZoom] = useState(1);
  const transform = useMemo(
    () => fitSlotProjection(snapshot.slot_layout, PARKING_GEOMETRY),
    [snapshot.slot_layout],
  );
  const projectedRoute = route.map((point) => transform.project(point));
  const projectedGates = gatePoints.map((point) => transform.project(point));

  function select(slotId: string): void {
    onSelectSlot?.(slotId);
  }

  function addGatePoint(event: React.MouseEvent<SVGSVGElement>): void {
    if (!gateEditing || !onGatePoint || gatePoints.length >= 6) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const svg: SvgPoint = [
      ((event.clientX - bounds.left) / bounds.width) * VIEWBOX.width,
      ((event.clientY - bounds.top) / bounds.height) * VIEWBOX.height,
    ];
    const world = transform.unproject(svg);
    if (world) onGatePoint(world);
  }

  return (
    <div className="relative overflow-hidden rounded-2xl bg-[#152019] shadow-[0_20px_70px_rgba(15,24,17,0.15)]">
      <div className="absolute right-3 top-3 z-10 flex gap-1 rounded-xl border border-white/10 bg-[#101712]/85 p-1 backdrop-blur">
        <Button aria-label="Thu nhỏ bản đồ" variant="ghost" size="icon" className="min-h-11 min-w-11 text-white hover:bg-white/10 hover:text-white" onClick={() => setZoom((value) => Math.max(0.8, value - 0.15))}><Minus /></Button>
        <Button aria-label="Đặt lại góc nhìn" variant="ghost" size="icon" className="min-h-11 min-w-11 text-white hover:bg-white/10 hover:text-white" onClick={() => setZoom(1)}><LocateFixed /></Button>
        <Button aria-label="Phóng to bản đồ" variant="ghost" size="icon" className="min-h-11 min-w-11 text-white hover:bg-white/10 hover:text-white" onClick={() => setZoom((value) => Math.min(1.6, value + 0.15))}><Plus /></Button>
      </div>
      {!transform.valid && (
        <p className="absolute left-3 top-3 z-10 rounded-lg bg-[#ffbe5c] px-2 py-1 text-[10px] font-semibold text-[#30200a]">Projection fallback</p>
      )}
      <svg
        viewBox={`0 0 ${VIEWBOX.width} ${VIEWBOX.height}`}
        className={compact ? 'min-h-[340px] w-full touch-none' : 'min-h-[480px] w-full touch-none'}
        role="button"
        tabIndex={0}
        aria-label="Bản đồ trực tiếp của bãi xe"
        onClick={addGatePoint}
        onKeyDown={(event) => { if (event.key === 'Escape') setZoom(1); }}
      >
        <defs>
          <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse">
            <path d="M 28 0 L 0 0 0 28" fill="none" stroke="rgba(255,255,255,.055)" strokeWidth="1" />
          </pattern>
          <filter id="marker-shadow" x="-80%" y="-80%" width="260%" height="260%">
            <feDropShadow dx="0" dy="5" stdDeviation="6" floodOpacity=".35" />
          </filter>
        </defs>
        <rect width={VIEWBOX.width} height={VIEWBOX.height} fill="#152019" />
        <rect width={VIEWBOX.width} height={VIEWBOX.height} fill="url(#grid)" />
        <g style={{ transform: `scale(${zoom})`, transformOrigin: 'center', transition: 'transform 180ms ease' }}>
          <rect x="28" y="28" width="864" height="504" rx="24" fill="#1a2820" stroke="rgba(255,255,255,.08)" />
          <path d="M40 188H880M40 372H880" stroke="rgba(215,255,63,.13)" strokeDasharray="8 10" />
          <path d="M40 280H880" stroke="rgba(255,255,255,.13)" strokeDasharray="16 12" />
          <text x="54" y="263" fill="rgba(255,255,255,.32)" fontSize="12">LÀN DI CHUYỂN CHÍNH</text>

          <g aria-label="Các ô đỗ xe">
            {snapshot.parking_slots.map((slot) => {
              const layout = snapshot.slot_layout.find((item) => item.slot_id === slot.slot_id);
              const polygon = PARKING_GEOMETRY[slot.slot_id]
                ?? layout?.polygon.map((point) => transform.project(point))
                ?? [];
              if (polygon.length === 0) return null;
              const selected = selectedSlotId === slot.slot_id;
              return (
                <g
                  key={slot.slot_id}
                  role={onSelectSlot ? 'button' : 'img'}
                  tabIndex={onSelectSlot ? 0 : undefined}
                  aria-label={`Ô ${slot.slot_id}, ${slot.occupied ? 'đã có xe' : 'trống'}`}
                  onClick={(event) => { event.stopPropagation(); if (!gateEditing) select(slot.slot_id); }}
                  onKeyDown={(event) => {
                    if ((event.key === 'Enter' || event.key === ' ') && !gateEditing) {
                      event.preventDefault(); select(slot.slot_id);
                    }
                  }}
                  className="cursor-pointer outline-none focus-visible:[&>polygon]:stroke-white"
                >
                  <polygon
                    points={pointsAttribute(polygon)}
                    fill={slot.occupied ? 'rgba(255,107,85,.18)' : 'rgba(215,255,63,.055)'}
                    stroke={selected ? '#65b7ff' : slot.occupied ? 'rgba(255,107,85,.8)' : 'rgba(215,255,63,.35)'}
                    strokeWidth={selected ? 4 : 1.4}
                  />
                  <text x={polygon[0][0] + 6} y={polygon[0][1] + 17} fill={slot.occupied ? '#ff9b8c' : 'rgba(215,255,63,.72)'} fontSize="10" fontWeight="700">{slot.slot_id}</text>
                </g>
              );
            })}
          </g>

          {projectedRoute.length > 1 && (
            <g aria-label="Tuyến đường đã xác nhận">
              <polyline points={pointsAttribute(projectedRoute)} fill="none" stroke="rgba(101,183,255,.2)" strokeWidth="14" strokeLinecap="round" strokeLinejoin="round" />
              <polyline points={pointsAttribute(projectedRoute)} fill="none" stroke="#65b7ff" strokeWidth="4" strokeDasharray="10 8" strokeLinecap="round" strokeLinejoin="round" />
            </g>
          )}

          <g aria-label="Các xe đang hiển thị">
            {vehicles.map((vehicle) => <VehicleMarker key={vehicle.globalId} vehicle={vehicle} transform={transform} />)}
          </g>

          {projectedGates.length > 0 && (
            <g aria-label="Điểm cấu hình cổng">
              {projectedGates.map((point, index) => <circle key={`${point[0]}-${point[1]}-${index}`} cx={point[0]} cy={point[1]} r="7" fill="#ffbe5c" stroke="white" strokeWidth="2" />)}
              <polyline points={pointsAttribute(projectedGates)} fill="none" stroke="#ffbe5c" strokeWidth="2" strokeDasharray="5 5" />
            </g>
          )}
        </g>
      </svg>
      {gateEditing && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-xl bg-[#ffbe5c] px-4 py-2 text-xs font-semibold text-[#30200a] shadow-lg">
          Chọn điểm cổng: {gatePoints.length}/6 · pan/zoom đã khóa khi chọn
        </div>
      )}
    </div>
  );
}

function VehicleMarker({ vehicle, transform }: { readonly vehicle: DisplayVehicle; readonly transform: AffineTransform }) {
  const element = useRef<SVGGElement | null>(null);
  const previous = useRef<WorldPoint | undefined>(undefined);
  const [x, y] = transform.project(vehicle.position);
  const parked = vehicle.state === 'parked';
  const missing = vehicle.state === 'temporarily_missing';

  useEffect(() => {
    const before = previous.current;
    if (before && element.current && shouldAnimateMarker(before, vehicle.position)) {
      const [fromX, fromY] = transform.project(before);
      element.current.animate(
        [{ transform: `translate(${fromX}px, ${fromY}px)` }, { transform: `translate(${x}px, ${y}px)` }],
        { duration: 350, easing: 'linear' },
      );
    }
    previous.current = vehicle.position;
  }, [transform, vehicle.position, x, y]);

  return (
    <g
      ref={element}
      role="img"
      aria-label={`Xe Global ID ${vehicle.globalId}, ${parked ? `đã đỗ tại ${vehicle.parkedSlotId}` : missing ? 'tạm mất dấu' : 'đang di chuyển'}`}
      style={{ transform: `translate(${x}px, ${y}px)` }}
      filter="url(#marker-shadow)"
    >
      <circle r="22" fill={parked ? '#2f7950' : missing ? '#566a5d' : '#163348'} stroke={parked ? '#8de1aa' : missing ? '#aab8ae' : '#65b7ff'} strokeWidth="3" opacity={missing ? .68 : 1} />
      {parked ? <LockKeyhole x={-8} y={-8} width="16" height="16" color="white" /> : <CarFront x={-8} y={-8} width="16" height="16" color="white" />}
      <rect x="-22" y="26" width="44" height="18" rx="9" fill="#0d1410" />
      <text x="0" y="39" textAnchor="middle" fill="white" fontSize="10" fontWeight="700">GID {vehicle.globalId}</text>
    </g>
  );
}
