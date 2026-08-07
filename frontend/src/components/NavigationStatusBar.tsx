import { MapPin, Route, X } from "lucide-react";
import type { SpotId, ZoneId } from "../domain/parking";

interface NavigationStatusBarProps {
  spotId: SpotId;
  zone: ZoneId;
  paused: boolean;
  onCancel: () => void;
}

export function NavigationStatusBar({ spotId, zone, paused, onCancel }: NavigationStatusBarProps) {
  return (
    <div className={`navigation-status ${paused ? "navigation-status--paused" : ""}`} aria-live="polite">
      <span className="navigation-icon">{paused ? <Route size={22} /> : <MapPin size={22} />}</span>
      <span>
        <small>{paused ? "Tuyến đường đang tạm dừng" : "Đang chỉ đường đến"}</small>
        <strong>{spotId} · Khu {zone}</strong>
      </span>
      <button type="button" onClick={onCancel} aria-label="Hủy chỉ đường" title="Hủy chỉ đường" data-testid="cancel-route">
        <X size={20} />
      </button>
    </div>
  );
}
