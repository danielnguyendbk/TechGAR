import { ChevronRight, MapPinned, Search, X } from "lucide-react";
import { useRef } from "react";
import { useFocusTrap } from "./useFocusTrap";

interface EntryChoiceSheetProps {
  onRecommend: () => void;
  onEmptyOnly: () => void;
  onSkip: () => void;
}

export function EntryChoiceSheet({ onRecommend, onEmptyOnly, onSkip }: EntryChoiceSheetProps) {
  const sheetRef = useRef<HTMLElement>(null);
  useFocusTrap(true, sheetRef);

  return (
    <>
      <div className="sheet-backdrop" aria-hidden="true" />
      <section ref={sheetRef} className="driver-sheet entry-sheet" role="dialog" aria-modal="true" aria-labelledby="entry-title" tabIndex={-1}>
        <div className="sheet-handle" aria-hidden="true" />
        <button className="sheet-close" type="button" onClick={onSkip} aria-label="Đóng và xem bản đồ" title="Đóng">
          <X size={20} />
        </button>
        <h2 id="entry-title">Bạn muốn tìm chỗ đỗ theo cách nào?</h2>
        <div className="entry-options">
          <button type="button" className="choice-row" onClick={onRecommend} data-testid="entry-recommend">
            <span className="choice-icon choice-icon--blue"><MapPinned size={27} /></span>
            <span className="choice-copy">
              <strong>Nhận đề xuất vị trí đỗ xe</strong>
              <small>Chọn nhu cầu để nhận ba vị trí phù hợp</small>
            </span>
            <ChevronRight size={23} aria-hidden="true" />
          </button>
          <button type="button" className="choice-row" onClick={onEmptyOnly} data-testid="entry-empty">
            <span className="choice-icon choice-icon--green"><Search size={27} /></span>
            <span className="choice-copy">
              <strong>Chỉ xem các ô đang trống</strong>
              <small>Mở sơ đồ để tự chọn vị trí</small>
            </span>
            <ChevronRight size={23} aria-hidden="true" />
          </button>
        </div>
        <button type="button" className="text-action" onClick={onSkip} data-testid="entry-skip">Bỏ qua</button>
      </section>
    </>
  );
}
