import { CircleAlert, Navigation, X } from "lucide-react";
import { STATUS_LABELS, isSelectableStatus, type ParkingSpotState } from "../domain/parking";

interface SpotDetailSheetProps {
  spot: ParkingSpotState;
  onClose: () => void;
  onNavigate: () => void;
}

export function SpotDetailSheet({ spot, onClose, onNavigate }: SpotDetailSheetProps) {
  const selectable = isSelectableStatus(spot.status);
  return (
    <section className="driver-panel spot-detail" role="dialog" aria-modal="false" aria-labelledby="spot-detail-title">
      <button type="button" className="sheet-close" onClick={onClose} aria-label="Đóng chi tiết ô" title="Đóng">
        <X size={20} />
      </button>
      <div className="spot-detail-heading">
        <span className={`status-dot status-dot--${spot.status}`} />
        <div>
          <h2 id="spot-detail-title">Ô {spot.id}</h2>
          <p>Khu {spot.zone} · {STATUS_LABELS[spot.status]}</p>
        </div>
      </div>
      <p className="spot-camera">Độ tin cậy {Math.round(spot.confidence * 100)}%</p>
      {selectable ? (
        <button type="button" className="primary-action" onClick={onNavigate} data-testid="spot-navigate">
          <Navigation size={19} />
          Chỉ đường đến {spot.id}
        </button>
      ) : (
        <p className="not-selectable"><CircleAlert size={18} />Chỉ ô xanh đang trống mới có thể được chọn.</p>
      )}
    </section>
  );
}
