import { Info, MapPin, Navigation, Timer, X } from "lucide-react";
import { useRef } from "react";
import {
  DESTINATION_LABELS,
  type DestinationNeed,
  type RecommendationResult,
  type SpotId,
} from "../domain/parking";
import { AlternativeSpotList } from "./AlternativeSpotList";
import { DestinationNeedSelector } from "./DestinationNeedSelector";
import { useFocusTrap } from "./useFocusTrap";

interface RecommendationPanelProps {
  need?: DestinationNeed;
  result?: RecommendationResult;
  onNeedChange: (need: DestinationNeed) => void;
  onChooseAlternative: (spotId: SpotId) => void;
  onConfirm: (spotId: SpotId) => void;
  onAbandon: () => void;
}

export function RecommendationPanel({
  need,
  result,
  onNeedChange,
  onChooseAlternative,
  onConfirm,
  onAbandon,
}: RecommendationPanelProps) {
  const panelRef = useRef<HTMLElement>(null);
  useFocusTrap(false, panelRef);

  return (
    <section ref={panelRef} className="driver-panel recommendation-panel" aria-labelledby="recommendation-title">
      <button type="button" className="sheet-close" onClick={onAbandon} aria-label="Bỏ gợi ý và xem toàn bộ bãi" title="Đóng">
        <X size={20} />
      </button>
      <h2 id="recommendation-title">Bạn muốn đến khu vực nào?</h2>
      <DestinationNeedSelector value={need} onChange={onNeedChange} />

      {!need && <p className="panel-intro">Chọn một nhu cầu để xem ba ô đang trống phù hợp nhất.</p>}
      {need && !result && <p className="panel-intro">Không có ô trống phù hợp ở thời điểm này.</p>}
      {need && result && (
        <>
          <div className="recommendation-primary" aria-live="polite" data-testid="recommendation-result">
            <div>
              <small>Đề xuất tốt nhất</small>
              <strong>{result.best.spotId}</strong>
              <span>Khu {result.best.zone}</span>
            </div>
            <dl>
              <div>
                <Timer size={18} aria-hidden="true" />
                <dt>Đi bộ</dt>
                <dd>{result.best.estimatedWalkingMinutes} phút</dd>
              </div>
              <div>
                <MapPin size={18} aria-hidden="true" />
                <dt>Khoảng cách</dt>
                <dd>{result.best.walkingDistance} m</dd>
              </div>
            </dl>
          </div>
          <p className="recommendation-reason">{result.best.reason} · {DESTINATION_LABELS[result.need]}</p>
          <AlternativeSpotList alternatives={result.alternatives} onChoose={onChooseAlternative} />
          <p className="disclaimer"><Info size={17} />Vị trí không được giữ trước và có thể thay đổi theo tình trạng thực tế.</p>
          <button type="button" className="primary-action confirm-route" onClick={() => onConfirm(result.best.spotId)} data-testid="recommendation-confirm">
            <Navigation size={19} />
            Chọn {result.best.spotId} và chỉ đường
          </button>
        </>
      )}
      <button type="button" className="text-action" onClick={onAbandon} data-testid="abandon-recommendation">
        Bỏ gợi ý và xem toàn bộ bãi
      </button>
    </section>
  );
}
