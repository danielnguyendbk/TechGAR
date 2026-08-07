import { Gamepad2, ShoppingBag, Wrench } from "lucide-react";
import { DESTINATION_LABELS, type DestinationNeed } from "../domain/parking";

interface DestinationNeedSelectorProps {
  value?: DestinationNeed;
  onChange: (need: DestinationNeed) => void;
}

const NEEDS = [
  { id: "shopping", icon: ShoppingBag },
  { id: "services", icon: Wrench },
  { id: "entertainment", icon: Gamepad2 },
] as const;

export function DestinationNeedSelector({ value, onChange }: DestinationNeedSelectorProps) {
  return (
    <div className="need-selector" role="radiogroup" aria-label="Nhu cầu điểm đến">
      {NEEDS.map(({ id, icon: Icon }) => (
        <button
          type="button"
          key={id}
          className={value === id ? "is-active" : ""}
          role="radio"
          aria-checked={value === id}
          onClick={() => onChange(id)}
          data-testid={`need-${id}`}
        >
          <Icon size={18} aria-hidden="true" />
          {DESTINATION_LABELS[id]}
        </button>
      ))}
    </div>
  );
}
