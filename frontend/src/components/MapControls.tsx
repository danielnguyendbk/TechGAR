import { LocateFixed, Minus, Plus } from "lucide-react";

interface MapControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
}

export function MapControls({ onZoomIn, onZoomOut, onReset }: MapControlsProps) {
  return (
    <div className="map-controls" aria-label="Điều khiển bản đồ">
      <button type="button" onClick={onZoomIn} aria-label="Phóng to bản đồ" title="Phóng to">
        <Plus size={19} />
      </button>
      <button type="button" onClick={onZoomOut} aria-label="Thu nhỏ bản đồ" title="Thu nhỏ">
        <Minus size={19} />
      </button>
      <button type="button" onClick={onReset} aria-label="Đặt lại góc nhìn" title="Đặt lại góc nhìn" data-testid="reset-view">
        <LocateFixed size={19} />
      </button>
    </div>
  );
}
