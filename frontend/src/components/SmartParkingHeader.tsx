import { Clock3, SquareParking } from "lucide-react";
import type { CameraId, CameraState, DriverMode } from "../domain/parking";
import { areAllCamerasOnline } from "../stores/parkingStore";

interface SmartParkingHeaderProps {
  mode: DriverMode;
  cameras: Record<CameraId, CameraState>;
  lastUpdated?: string;
}

function formatTime(value?: string): string {
  if (!value) return "Đang đồng bộ";
  return new Intl.DateTimeFormat("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: "Asia/Ho_Chi_Minh",
  }).format(new Date(value));
}

export function SmartParkingHeader({ mode, cameras, lastUpdated }: SmartParkingHeaderProps) {
  const online = areAllCamerasOnline(cameras);
  return (
    <header className="app-header">
      <div className="brand-lockup">
        <span className="brand-mark" aria-hidden="true">
          <SquareParking size={31} strokeWidth={2.2} />
        </span>
        <span>
          <strong>Smart Parking</strong>
          <small>Bãi đỗ xe trung tâm thương mại</small>
        </span>
      </div>
      <div className="header-status" aria-live="polite">
        <span className={online ? "health-dot health-dot--online" : "health-dot health-dot--degraded"} />
        <span>{online ? "Hệ thống hoạt động bình thường" : "Dữ liệu camera đang suy giảm"}</span>
        <span className="updated-time">
          <Clock3 size={16} aria-hidden="true" />
          Cập nhật: {formatTime(lastUpdated)}
        </span>
        <span className="sr-only">Chế độ hiện tại: {mode}</span>
      </div>
    </header>
  );
}
