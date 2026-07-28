import { CircleAlert, Map, RefreshCw } from "lucide-react";
import { useRef } from "react";
import { getInvalidSpotWarningText, type InvalidSpotWarning } from "../domain/parking";
import { useFocusTrap } from "./useFocusTrap";

interface InvalidSpotWarningSheetProps {
  warning: InvalidSpotWarning;
  onSwitch: () => void;
  onContinueMap: () => void;
}

export function InvalidSpotWarningSheet({ warning, onSwitch, onContinueMap }: InvalidSpotWarningSheetProps) {
  const sheetRef = useRef<HTMLElement>(null);
  useFocusTrap(true, sheetRef);
  return (
    <>
      <div className="sheet-backdrop sheet-backdrop--warning" aria-hidden="true" />
      <section ref={sheetRef} className="driver-sheet warning-sheet" role="alertdialog" aria-modal="true" aria-labelledby="warning-title" tabIndex={-1}>
        <div className="warning-heading">
          <CircleAlert size={26} aria-hidden="true" />
          <div>
            <h2 id="warning-title">Vị trí đã thay đổi</h2>
            <p>{getInvalidSpotWarningText(warning.spotId, warning.status)}</p>
          </div>
        </div>
        {warning.alternativeSpotId && (
          <div className="next-alternative">
            <small>Phương án trống tiếp theo</small>
            <strong>{warning.alternativeSpotId}</strong>
          </div>
        )}
        {warning.alternativeSpotId && (
          <button type="button" className="primary-action" onClick={onSwitch} data-testid="switch-alternative">
            <RefreshCw size={19} />
            Chuyển sang {warning.alternativeSpotId}
          </button>
        )}
        <button type="button" className="secondary-action" onClick={onContinueMap} data-testid="continue-map">
          <Map size={19} />
          Tiếp tục xem bản đồ
        </button>
      </section>
    </>
  );
}
