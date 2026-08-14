import React from 'react';
import type { ParkingSlotLayout, ParkingStatus } from '../../types/parking';

interface ParkingSlotProps {
  layout: ParkingSlotLayout;
  status: ParkingStatus;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onHoverEnter: (e: React.MouseEvent, layout: ParkingSlotLayout, status: ParkingStatus) => void;
  onHoverLeave: () => void;
  onHoverMove: (e: React.MouseEvent) => void;
}

export const ParkingSlot: React.FC<ParkingSlotProps> = ({
  layout,
  status,
  isSelected,
  onSelect,
  onHoverEnter,
  onHoverLeave,
  onHoverMove,
}) => {
  const { id, points, x = 0, y = 0, width = 0, height = 0 } = layout;

  // Calculate approximate center for placing the text label
  const centerX = x + width / 2;
  const centerY = y + height / 2;

  const handleSelect = (e: React.MouseEvent) => {
    e.stopPropagation();
    onSelect(id);
  };

  return (
    <g className={`parking-slot-group ${isSelected ? 'selected' : ''}`}>
      <polygon
        points={points}
        className={`parking-slot-poly status-${status} ${isSelected ? 'is-selected' : ''}`}
        onClick={handleSelect}
        onMouseEnter={(e) => onHoverEnter(e, layout, status)}
        onMouseLeave={onHoverLeave}
        onMouseMove={onHoverMove}
      />
      <text
        x={centerX}
        y={centerY}
        className="parking-slot-text"
      >
        {id}
      </text>
    </g>
  );
};
