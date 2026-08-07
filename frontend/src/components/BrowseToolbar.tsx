import { ListFilter, Sparkles } from "lucide-react";
import type { BrowseFilter } from "../domain/parking";

interface BrowseToolbarProps {
  filter: BrowseFilter;
  onFilterChange: (filter: BrowseFilter) => void;
  onFindSpot: () => void;
}

export function BrowseToolbar({ filter, onFilterChange, onFindSpot }: BrowseToolbarProps) {
  return (
    <div className="browse-actions">
      <div className="browse-toolbar" role="group" aria-label="Bộ lọc trạng thái ô đỗ">
        <ListFilter size={18} aria-hidden="true" />
        <button
          type="button"
          className={filter === "empty" ? "is-active" : ""}
          aria-pressed={filter === "empty"}
          onClick={() => onFilterChange("empty")}
          data-testid="filter-empty"
        >
          Chỉ hiện ô trống
        </button>
        <button
          type="button"
          className={filter === "all" ? "is-active" : ""}
          aria-pressed={filter === "all"}
          onClick={() => onFilterChange("all")}
          data-testid="filter-all"
        >
          Hiện tất cả trạng thái
        </button>
      </div>
      <button type="button" className="primary-action browse-recommend-action" onClick={onFindSpot} data-testid="browse-recommend">
        <Sparkles size={19} />
        Tìm chỗ phù hợp
      </button>
    </div>
  );
}
