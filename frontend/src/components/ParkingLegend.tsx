import { STATUS_LABELS, type ParkingStatus } from "../domain/parking";

const STATUSES: ParkingStatus[] = ["empty", "occupied", "transitioning", "unknown"];

export function ParkingLegend() {
  return (
    <div className="parking-legend" aria-label="Chú thích trạng thái">
      {STATUSES.map((status) => (
        <span key={status}>
          <i className={`legend-swatch legend-swatch--${status}`} aria-hidden="true" />
          {STATUS_LABELS[status]}
        </span>
      ))}
      <span>
        <i className="legend-swatch legend-swatch--selected" aria-hidden="true" />
        Được chọn
      </span>
    </div>
  );
}
