import { STATUS_LABELS, type ParkingSpotState } from "../domain/parking";
import type { SpotGeometry } from "../geometry/parkingGeometry";

export type SpotHighlight = "recommended" | "alternative" | "confirmed" | "inspected";

interface ParkingSpotShapeProps {
  geometry: SpotGeometry;
  spot: ParkingSpotState;
  highlight?: SpotHighlight;
  staleCamera?: boolean;
  dimmed?: boolean;
}

export function ParkingSpotShape({
  geometry,
  spot,
  highlight,
  staleCamera = false,
  dimmed = false,
}: ParkingSpotShapeProps) {
  const outlineClass = highlight ? ` parking-spot--${highlight}` : "";
  return (
    <g
      className={`parking-spot parking-spot--${spot.status}${outlineClass}${dimmed ? " parking-spot--dimmed" : ""}`}
      data-visible-spot={dimmed ? "false" : "true"}
    >
      <rect
        x={geometry.x}
        y={geometry.y}
        width={geometry.width}
        height={geometry.height}
        rx={4}
        vectorEffect="non-scaling-stroke"
      />
      <text
        x={geometry.x + geometry.width / 2}
        y={geometry.y + geometry.height / 2 + 4}
        textAnchor="middle"
        aria-hidden="true"
      >
        {spot.id}
      </text>
      {staleCamera && (
        <circle
          className="stale-camera-dot"
          cx={geometry.x + geometry.width - 5}
          cy={geometry.y + 5}
          r={3.5}
          aria-label={`Dữ liệu ${spot.id} có thể đã cũ`}
        />
      )}
      <title>{`${spot.id}: ${STATUS_LABELS[spot.status]}`}</title>
    </g>
  );
}
