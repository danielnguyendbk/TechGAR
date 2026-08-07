import { CarFront, CircleHelp, CircleParking, RefreshCw } from "lucide-react";
import type { CameraId, CameraState, ParkingCounts } from "../domain/parking";
import { CameraHealthIndicator } from "./CameraHealthIndicator";

interface SummaryCardsProps {
  counts: ParkingCounts;
  cameras: Record<CameraId, CameraState>;
}

export function SummaryCards({ counts, cameras }: SummaryCardsProps) {
  const items = [
    { label: "Còn trống", value: counts.empty, icon: CircleParking, tone: "green" },
    { label: "Đã có xe", value: counts.occupied, icon: CarFront, tone: "red" },
    { label: "Đang chuyển tiếp", value: counts.transitioning, icon: RefreshCw, tone: "amber" },
    { label: "Không xác định", value: counts.unknown, icon: CircleHelp, tone: "gray" },
  ] as const;

  return (
    <section className="summary-strip" aria-label="Tổng quan bãi xe">
      {items.map(({ label, value, icon: Icon, tone }) => (
        <div className={`summary-card summary-card--${tone}`} key={label}>
          <Icon size={25} aria-hidden="true" />
          <span>
            <small>{label}</small>
            <strong>{value}</strong>
          </span>
        </div>
      ))}
      <div className="summary-card summary-card--camera">
        <CameraHealthIndicator cameras={cameras} compact />
      </div>
    </section>
  );
}
