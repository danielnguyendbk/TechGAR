import React from 'react';

export const ParkingLegend: React.FC = () => {
  const legendItems = [
    { key: 'available', label: 'Trống', className: 'color-available' },
    { key: 'occupied', label: 'Có xe', className: 'color-occupied' },
    { key: 'incoming', label: 'Sắp có xe', className: 'color-incoming' },
  ];

  return (
    <div className="parking-legend">
      {legendItems.map((item) => (
        <div key={item.key} className="parking-legend-item">
          <div className={`parking-legend-color ${item.className}`} />
          <span>{item.label}</span>
        </div>
      ))}
    </div>
  );
};
