import { useRef, useState, type KeyboardEvent, type PointerEvent, type WheelEvent } from "react";
import {
  DESTINATION_LABELS,
  STATUS_LABELS,
  type BrowseFilter,
  type DestinationNeed,
  type ParkingSpotState,
  type RecommendationResult,
  type SpotId,
} from "../domain/parking";
import { PARKING_GEOMETRY, type Point } from "../geometry/parkingGeometry";
import type { RouteResult } from "../routing/routeEngine";
import { MapControls } from "./MapControls";
import { ParkingSpotShape, type SpotHighlight } from "./ParkingSpotShape";
import type { ActiveVehicle, FrameSize } from "../app/App";

// ── Chuyển tọa độ camera → tọa độ bản đồ SVG (1200×900) ────────────────────
// Kích thước nguồn được truyền động từ frame_size trong vehicle_positions.json
// nên tự điều chỉnh theo mọi camera/video khác nhau.
function camToMap(cx: number, cy: number, fs: FrameSize): Point {
  return {
    x: (cx / fs.width)  * PARKING_GEOMETRY.width,
    y: (cy / fs.height) * PARKING_GEOMETRY.height,
  };
}

interface ViewBoxState {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface ParkingMapProps {
  spots: readonly ParkingSpotState[];
  filter: BrowseFilter;
  recommendation?: RecommendationResult;
  inspectedSpotId?: SpotId;
  confirmedSpotId?: SpotId;
  activeNeed?: DestinationNeed;
  route?: RouteResult | null;
  routePaused?: boolean;
  activeVehicles?: ActiveVehicle[];   // ← xe đang di chuyển thời gian thực
  frameSize?: FrameSize;              // ← kích thước frame camera (tự động từ JSON)
  onSpotClick: (spotId: SpotId) => void;
}

const INITIAL_VIEW: ViewBoxState = {
  x: 0,
  y: 0,
  width: PARKING_GEOMETRY.width,
  height: PARKING_GEOMETRY.height,
};

function clampView(view: ViewBoxState): ViewBoxState {
  const width = Math.min(PARKING_GEOMETRY.width, Math.max(360, view.width));
  const height = width * (PARKING_GEOMETRY.height / PARKING_GEOMETRY.width);
  return {
    x: Math.min(PARKING_GEOMETRY.width - width, Math.max(0, view.x)),
    y: Math.min(PARKING_GEOMETRY.height - height, Math.max(0, view.y)),
    width,
    height,
  };
}

function MapPinMarker({ point, small = false }: { point: Point; small?: boolean }) {
  const scale = small ? 0.72 : 1;
  return (
    <g className="map-pin-marker" transform={`translate(${point.x} ${point.y - 12}) scale(${scale})`} aria-hidden="true">
      <path d="M0 0 C-15 -18 -18 -28 -18 -38 A18 18 0 1 1 18 -38 C18 -28 15 -18 0 0Z" />
      <circle cx="0" cy="-38" r="6" />
    </g>
  );
}

interface RouteArrowPoint extends Point {
  angle: number;
}

function simplifyRoutePoints(points: Point[]): Point[] {
  if (points.length < 3) return points;
  const simplified: Point[] = [points[0]!];

  for (let index = 1; index < points.length - 1; index += 1) {
    const previous = simplified.at(-1)!;
    const current = points[index]!;
    const next = points[index + 1]!;
    const crossProduct = (current.x - previous.x) * (next.y - current.y) - (current.y - previous.y) * (next.x - current.x);
    if (Math.abs(crossProduct) > 0.01) simplified.push(current);
  }

  simplified.push(points.at(-1)!);
  return simplified;
}

function createRouteArrows(points: Point[]): RouteArrowPoint[] {
  const segments = points.slice(1).map((point, index) => {
    const start = points[index]!;
    return { start, end: point, length: Math.hypot(point.x - start.x, point.y - start.y) };
  });
  const totalLength = segments.reduce((total, segment) => total + segment.length, 0);
  if (totalLength < 140) return [];

  const arrows: RouteArrowPoint[] = [];
  for (let targetDistance = 100; targetDistance < totalLength - 45; targetDistance += 190) {
    let traversed = 0;
    const segment = segments.find((candidate) => {
      if (targetDistance <= traversed + candidate.length) return true;
      traversed += candidate.length;
      return false;
    });
    if (!segment || segment.length === 0) continue;
    const ratio = (targetDistance - traversed) / segment.length;
    arrows.push({
      x: segment.start.x + (segment.end.x - segment.start.x) * ratio,
      y: segment.start.y + (segment.end.y - segment.start.y) * ratio,
      angle: Math.atan2(segment.end.y - segment.start.y, segment.end.x - segment.start.x) * (180 / Math.PI),
    });
  }
  return arrows;
}

export function ParkingMap({
  spots,
  filter,
  recommendation,
  inspectedSpotId,
  confirmedSpotId,
  activeNeed,
  route,
  routePaused = false,
  activeVehicles = [],
  frameSize = { width: 1100, height: 720 },
  onSpotClick,
}: ParkingMapProps) {
  const [view, setView] = useState(INITIAL_VIEW);
  const pointers = useRef(new Map<number, Point>());
  const svgRef = useRef<SVGSVGElement>(null);
  const spotById = new Map(spots.map((spot) => [spot.id, spot]));
  const recommendedIds = new Set(recommendation ? [recommendation.best.spotId, ...recommendation.alternatives.map((spot) => spot.spotId)] : []);

  const zoomAt = (factor: number, ratioX = 0.5, ratioY = 0.5): void => {
    setView((current) => {
      const nextWidth = current.width * factor;
      const nextHeight = current.height * factor;
      return clampView({
        x: current.x + (current.width - nextWidth) * ratioX,
        y: current.y + (current.height - nextHeight) * ratioY,
        width: nextWidth,
        height: nextHeight,
      });
    });
  };

  const handlePointerDown = (event: PointerEvent<SVGSVGElement>): void => {
    if (typeof event.currentTarget.setPointerCapture === "function") {
      event.currentTarget.setPointerCapture(event.pointerId);
    }
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
  };

  const handlePointerMove = (event: PointerEvent<SVGSVGElement>): void => {
    const previous = pointers.current.get(event.pointerId);
    if (!previous) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const pointerValues = [...pointers.current.values()];

    if (pointerValues.length === 1) {
      const dx = event.clientX - previous.x;
      const dy = event.clientY - previous.y;
      setView((current) =>
        clampView({
          ...current,
          x: current.x - (dx / rect.width) * current.width,
          y: current.y - (dy / rect.height) * current.height,
        }),
      );
    } else if (pointerValues.length === 2) {
      const other = [...pointers.current.entries()].find(([id]) => id !== event.pointerId)?.[1];
      if (other) {
        const previousDistance = Math.hypot(previous.x - other.x, previous.y - other.y);
        const nextDistance = Math.hypot(event.clientX - other.x, event.clientY - other.y);
        if (previousDistance > 0 && nextDistance > 0) {
          const midX = (event.clientX + other.x) / 2;
          const midY = (event.clientY + other.y) / 2;
          zoomAt(previousDistance / nextDistance, (midX - rect.left) / rect.width, (midY - rect.top) / rect.height);
        }
      }
    }
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
  };

  const handlePointerEnd = (event: PointerEvent<SVGSVGElement>): void => {
    pointers.current.delete(event.pointerId);
  };

  const handleWheel = (event: WheelEvent<SVGSVGElement>): void => {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    zoomAt(event.deltaY > 0 ? 1.12 : 0.88, (event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height);
  };

  const handleSpotKeyDown = (event: KeyboardEvent<SVGRectElement>, spotId: SpotId): void => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSpotClick(spotId);
    }
  };

  const renderedRoutePoints = simplifyRoutePoints(route?.points ?? []);
  const routePoints = renderedRoutePoints.map((point) => `${point.x},${point.y}`).join(" ");
  const routeArrows = createRouteArrows(renderedRoutePoints);

  return (
    <section className="map-panel" aria-label="Sơ đồ bãi đỗ xe">
      <div className="map-canvas" data-testid="parking-map">
        <svg
          ref={svgRef}
          viewBox={`${view.x} ${view.y} ${view.width} ${view.height}`}
          preserveAspectRatio="xMidYMin meet"
          role="img"
          aria-label="Bản đồ 160 ô đỗ xe, khu E đến A từ trên xuống và khu F bên phải"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerEnd}
          onPointerCancel={handlePointerEnd}
          onWheel={handleWheel}
        >
          <defs>
            <pattern id="landscape-pattern" width="26" height="26" patternUnits="userSpaceOnUse">
              <rect width="26" height="26" fill="#1e6245" />
              <circle cx="6" cy="8" r="2" fill="#327a52" />
              <circle cx="20" cy="18" r="3" fill="#164f39" />
            </pattern>
          </defs>

          <rect width={PARKING_GEOMETRY.width} height={PARKING_GEOMETRY.height} rx="26" fill="url(#landscape-pattern)" />
          <rect className="access-road" {...PARKING_GEOMETRY.accessRoad} />
          {PARKING_GEOMETRY.zones.map((zone) => (
            <g key={zone.id}>
              <rect className="zone-road" {...zone.bounds} rx="34" />
              <line className="lane-centerline" x1={zone.bounds.x + 28} x2={zone.bounds.x + zone.bounds.width} y1={zone.laneY} y2={zone.laneY} />
              <g className="lane-arrow" transform={`translate(${zone.bounds.x + zone.bounds.width / 2} ${zone.laneY})`}>
                <path d="M16 -5 H-4 V-12 L-20 0 L-4 12 V5 H16Z" />
              </g>
              <g className="zone-badge" transform={`translate(${zone.bounds.x + zone.bounds.width / 2 - 45} ${zone.bounds.y - 8})`}>
                <rect width="90" height="26" rx="13" />
                <text x="45" y="18" textAnchor="middle">KHU {zone.id}</text>
              </g>
            </g>
          ))}
          {PARKING_GEOMETRY.zoneConnectors.map((connector) => (
            <g key={connector.id}>
              <rect className="road-connector" {...connector.bounds} />
              <line
                className="road-connector-edge"
                x1={PARKING_GEOMETRY.layout.zoneRightX - 4}
                x2={PARKING_GEOMETRY.layout.mainRoadX + 4}
                y1={connector.bounds.y}
                y2={connector.bounds.y}
              />
              <line
                className="road-connector-edge"
                x1={PARKING_GEOMETRY.layout.zoneRightX - 4}
                x2={PARKING_GEOMETRY.layout.mainRoadX + 4}
                y1={connector.bounds.y + connector.bounds.height}
                y2={connector.bounds.y + connector.bounds.height}
              />
              <line
                className="lane-centerline"
                x1={connector.centerline.start.x}
                x2={connector.centerline.end.x}
                y1={connector.centerline.start.y}
                y2={connector.centerline.end.y}
              />
            </g>
          ))}
          <rect className="zone-road f-strip" {...PARKING_GEOMETRY.fStripBounds} rx="30" />
          {PARKING_GEOMETRY.fAccessConnectors.map((connector) => (
            <g key={connector.id}>
              <rect className="road-connector road-connector--f" {...connector.bounds} />
              <line
                className="road-connector-edge road-connector-edge--f"
                x1={PARKING_GEOMETRY.layout.mainRoadX + PARKING_GEOMETRY.layout.mainRoadWidth - 4}
                x2={connector.centerline.end.x}
                y1={connector.bounds.y}
                y2={connector.bounds.y}
              />
              <line
                className="road-connector-edge road-connector-edge--f"
                x1={PARKING_GEOMETRY.layout.mainRoadX + PARKING_GEOMETRY.layout.mainRoadWidth - 4}
                x2={connector.centerline.end.x}
                y1={connector.bounds.y + connector.bounds.height}
                y2={connector.bounds.y + connector.bounds.height}
              />
            </g>
          ))}
          <g className="zone-badge" transform={`translate(${PARKING_GEOMETRY.fStripBounds.x + 2} ${PARKING_GEOMETRY.fStripBounds.y - 27})`}>
            <rect width="88" height="27" rx="13" />
            <text x="44" y="18" textAnchor="middle">KHU F</text>
          </g>
          <line
            className="access-centerline"
            x1={PARKING_GEOMETRY.layout.mainRoadCenterX}
            x2={PARKING_GEOMETRY.layout.mainRoadCenterX}
            y1={PARKING_GEOMETRY.exit.y + 27}
            y2={PARKING_GEOMETRY.entrance.y - 22}
          />
          <g className="access-label" transform={`translate(${PARKING_GEOMETRY.layout.mainRoadCenterX} ${PARKING_GEOMETRY.exit.y + 14})`}>
            <path d="M0 25 V-6 M-9 4 L0 -7 L9 4" />
            <text x="0" y="45" textAnchor="middle">LỐI RA</text>
          </g>
          <g className="access-label" transform={`translate(${PARKING_GEOMETRY.layout.mainRoadCenterX} ${PARKING_GEOMETRY.entrance.y - 15})`}>
            <path d="M0 -26 V4 M-9 -16 L0 -27 L9 -16" />
            <text x="0" y="28" textAnchor="middle">LỐI VÀO</text>
          </g>

          <g aria-hidden="true">
            {PARKING_GEOMETRY.spots.map((geometry) => {
              const spot = spotById.get(geometry.id);
              if (!spot) return null;
              const dimmed = filter === "empty" && spot.status !== "empty";
              let highlight: SpotHighlight | undefined;
              if (confirmedSpotId === spot.id) highlight = "confirmed";
              else if (inspectedSpotId === spot.id) highlight = "inspected";
              else if (recommendation?.best.spotId === spot.id) highlight = "recommended";
              else if (recommendedIds.has(spot.id)) highlight = "alternative";
              return (
                <ParkingSpotShape
                  key={spot.id}
                  geometry={geometry}
                  spot={spot}
                  highlight={highlight}
                  staleCamera={false}
                  dimmed={dimmed}
                />
              );
            })}
          </g>

          {activeNeed && (
            <g className="access-anchor" transform={`translate(${PARKING_GEOMETRY.anchors[activeNeed].x} ${PARKING_GEOMETRY.anchors[activeNeed].y})`}>
              <circle r="13" />
              <path d="M-5 2 L-1 6 L7 -5" />
              <text x="18" y="5">{DESTINATION_LABELS[activeNeed]}</text>
            </g>
          )}

          {routePoints && !routePaused && (
            <g className="active-route" data-testid="active-route" aria-label="Tuyến đường đang hoạt động">
              <polyline className="route-underlay" points={routePoints} />
              <polyline className="route-line" points={routePoints} />
              {routeArrows.map((arrow, index) => (
                <g
                  className="route-direction"
                  key={`${arrow.x}-${arrow.y}-${index}`}
                  transform={`translate(${arrow.x} ${arrow.y}) rotate(${arrow.angle})`}
                >
                  <path d="M-9 -6 L8 0 L-9 6 Z" />
                </g>
              ))}
            </g>
          )}

          {recommendation && !confirmedSpotId && (() => {
            const geometry = PARKING_GEOMETRY.spots.find((spot) => spot.id === recommendation.best.spotId);
            return geometry ? <MapPinMarker point={{ x: geometry.x + geometry.width / 2, y: geometry.y }} small /> : null;
          })()}
          {confirmedSpotId && (() => {
            const geometry = PARKING_GEOMETRY.spots.find((spot) => spot.id === confirmedSpotId);
            return geometry ? <MapPinMarker point={{ x: geometry.x + geometry.width / 2, y: geometry.y }} /> : null;
          })()}

          {/* ── Icon xe di chuyển thời gian thực (từ tracker của An) ────────── */}
          {activeVehicles.map((vehicle) => {
            const pos = camToMap(vehicle.x, vehicle.y, frameSize);
            const trailPoints = vehicle.trail
              .slice(-25)
              .map((p) => camToMap(p.x, p.y, frameSize))
              .map((p) => `${p.x},${p.y}`)
              .join(" ");
            return (
              <g key={vehicle.trackId}>
                {/* Đường trail động – màu vàng đứt nút */}
                {trailPoints && (
                  <polyline
                    points={trailPoints}
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth={3}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    opacity={0.55}
                    strokeDasharray="8 5"
                  />
                )}
                {/* Xe + Nhãn ID - Di chuyển mượt bằng CSS transform transition */}
                <g
                  style={{
                    transform: `translate(${pos.x}px, ${pos.y}px)`,
                    transition: "transform 0.25s linear",
                    willChange: "transform",
                  }}
                >
                  {/* Halo phát sáng */}
                  <circle cx={0} cy={0} r={18} fill="#ef4444" opacity={0.18}>
                    <animate attributeName="r" values="14;22;14" dur="1.6s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.22;0.06;0.22" dur="1.6s" repeatCount="indefinite" />
                  </circle>
                  {/* Nền tròn xe */}
                  <circle cx={0} cy={0} r={13} fill="#ef4444" stroke="#fff" strokeWidth={2.5} />
                  {/* Biểu tượng xe */}
                  <text
                    x={0}
                    y={0}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize={12}
                    fill="white"
                  >
                    {"\uD83D\uDE97"}
                  </text>
                  {/* Nhãn ID track */}
                  <rect
                    x={14}
                    y={-14}
                    width={28}
                    height={16}
                    rx={4}
                    fill="rgba(0,0,0,0.7)"
                  />
                  <text
                    x={28}
                    y={-6}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize={9}
                    fontWeight="700"
                    fill="#fbbf24"
                  >
                    #{vehicle.trackId}
                  </text>
                </g>
              </g>
            );
          })}

          <g className="spot-hit-layer">
            {PARKING_GEOMETRY.spots.map((geometry) => {
              const spot = spotById.get(geometry.id);
              if (!spot) return null;
              const hidden = filter === "empty" && spot.status !== "empty";
              return (
                <rect
                  key={spot.id}
                  x={geometry.x - 2}
                  y={geometry.y - 3}
                  width={geometry.width + 4}
                  height={geometry.height + 6}
                  rx="5"
                  role="button"
                  tabIndex={hidden ? -1 : 0}
                  aria-label={`Ô ${spot.id}, ${STATUS_LABELS[spot.status]}`}
                  aria-current={confirmedSpotId === spot.id ? "location" : undefined}
                  aria-disabled={hidden}
                  data-spot-id={spot.id}
                  data-status={spot.status}
                  data-testid={`spot-${spot.id}`}
                  onPointerDown={(event) => event.stopPropagation()}
                  onClick={() => !hidden && onSpotClick(spot.id)}
                  onKeyDown={(event) => !hidden && handleSpotKeyDown(event, spot.id)}
                />
              );
            })}
          </g>
        </svg>
        <MapControls
          onZoomIn={() => zoomAt(0.8)}
          onZoomOut={() => zoomAt(1.25)}
          onReset={() => setView(INITIAL_VIEW)}
        />
      </div>
    </section>
  );
}
