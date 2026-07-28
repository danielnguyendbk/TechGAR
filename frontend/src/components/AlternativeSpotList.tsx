import { ChevronRight } from "lucide-react";
import type { RankedSpot, SpotId } from "../domain/parking";

interface AlternativeSpotListProps {
  alternatives: RankedSpot[];
  onChoose: (spotId: SpotId) => void;
}

export function AlternativeSpotList({ alternatives, onChoose }: AlternativeSpotListProps) {
  return (
    <div className="alternative-list">
      <h3>Phương án khác</h3>
      {alternatives.length === 0 ? (
        <p>Hiện chưa có phương án thay thế.</p>
      ) : (
        <div className="alternative-options">
          {alternatives.map((spot) => (
            <button type="button" key={spot.spotId} onClick={() => onChoose(spot.spotId)}>
              <span className="spot-chip">{spot.spotId}</span>
              <span>
                Khu {spot.zone} · {spot.estimatedWalkingMinutes} phút
              </span>
              <ChevronRight size={18} aria-hidden="true" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
