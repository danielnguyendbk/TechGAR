import { Clock3, SquareParking, Radio } from "lucide-react";
import type { CameraId, CameraState, DriverMode } from "../domain/parking";
import { areAllCamerasOnline, useParkingStore } from "../stores/parkingStore";

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
  const trackingSource = useParkingStore((state) => state.trackingSource);
  const setTrackingSource = useParkingStore((state) => state.setTrackingSource);

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

      {/* ── Selector nguồn dữ liệu (Dữ liệu mẫu vs Camera thật) ── */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "6px",
        background: "rgba(15, 23, 42, 0.6)",
        padding: "4px 8px",
        borderRadius: "8px",
        border: "1px solid #334155",
        fontSize: "12px",
      }}>
        <Radio size={14} style={{ color: trackingSource === "sample" ? "#38bdf8" : "#22c55e" }} />
        <span style={{ color: "#94a3b8", fontWeight: 500, marginRight: "4px" }}>Nguồn:</span>
        <button
          type="button"
          onClick={() => setTrackingSource("sample")}
          style={{
            background: trackingSource === "sample" ? "#0284c7" : "transparent",
            color: trackingSource === "sample" ? "#fff" : "#94a3b8",
            border: "none",
            borderRadius: "6px",
            padding: "4px 10px",
            cursor: "pointer",
            fontWeight: trackingSource === "sample" ? 600 : 400,
            fontSize: "12px",
            transition: "all 0.2s"
          }}
        >
          🧪 Dữ liệu mẫu
        </button>
        <button
          type="button"
          onClick={() => setTrackingSource("opencv")}
          style={{
            background: trackingSource === "opencv" ? "#16a34a" : "transparent",
            color: trackingSource === "opencv" ? "#fff" : "#94a3b8",
            border: "none",
            borderRadius: "6px",
            padding: "4px 10px",
            cursor: "pointer",
            fontWeight: trackingSource === "opencv" ? 600 : 400,
            fontSize: "12px",
            transition: "all 0.2s"
          }}
        >
          📷 Camera OpenCV
        </button>
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

