import React, { useEffect, useMemo, useRef, useState } from 'react';
import type {
  ActiveVehicle,
  BackendParkingSlot,
  BackendParkingSlotsResponse,
  ParkingSpotState,
  ParkingSlotLayout,
  ParkingStatus,
} from '../../types/parking';
import { parkingLayout } from '../../data/parkingLayout';
import { ParkingSlot } from './ParkingSlot';
import { ParkingLegend } from './ParkingLegend';
import { getVectorVehicleMarkers, type VectorVehicleMarker } from './vehicleCalibration';
import parkingBg from '../../assets/parking/parking-lot-empty.png';
import './parking-map.css';

interface ParkingMapProps {
  spots: ParkingSpotState[];
  activeVehicles?: ActiveVehicle[];
  selectedSpotId?: string | null;
  onSelectSpot?: (spotId: string) => void;
}

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  id: string;
  zone: string;
  status: ParkingStatus;
}

interface DisplayVehicleMarker extends VectorVehicleMarker {
  isVisible: boolean;
  lastSeenAt: number;
}

const MISSING_VEHICLE_GRACE_MS = 2000;
const MARKER_FADE_MS = 650;

const zoneTranslation: Record<string, string> = {
  LEFT_OUTER: 'Cột trái ngoài',
  LEFT_INNER: 'Cột trái trong',
  MIDDLE_LEFT: 'Cột giữa trái',
  MIDDLE_RIGHT: 'Cột giữa phải',
  RIGHT_LEFT: 'Cột phải trái',
  RIGHT_RIGHT: 'Cột phải phải',
};

const statusTranslation: Record<ParkingStatus, string> = {
  available: 'Trống',
  occupied: 'Có xe',
  incoming: 'Sắp có xe',
  unavailable: 'Đang chờ dữ liệu',
};

export const ParkingMap: React.FC<ParkingMapProps> = ({
  spots,
  activeVehicles = [],
  selectedSpotId,
  onSelectSpot,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [backendSlots, setBackendSlots] = useState<BackendParkingSlot[]>([]);
  const [displayVehicleMarkers, setDisplayVehicleMarkers] = useState<DisplayVehicleMarker[]>([]);
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false,
    x: 0,
    y: 0,
    id: '',
    zone: '',
    status: 'available',
  });

  // Helper to get status of a spot
  const getSpotStatus = (id: string): ParkingStatus => {
    const spot = spots.find((s) => s.id === id);
    return spot ? spot.status : 'unavailable';
  };

  useEffect(() => {
    let isMounted = true;

    const fetchBackendSlots = async () => {
      try {
        const res = await fetch('http://localhost:5050/api/parking-slots-lmr');
        if (!res.ok) throw new Error('Parking slots API returned error code');
        const data = (await res.json()) as BackendParkingSlotsResponse;
        if (isMounted && Array.isArray(data?.slots)) {
          setBackendSlots(data.slots);
        }
      } catch (err) {
        if (isMounted) {
          setBackendSlots([]);
          console.error('Error fetching backend parking slots:', err);
        }
      }
    };

    fetchBackendSlots();

    return () => {
      isMounted = false;
    };
  }, []);

  const handleHoverEnter = (
    e: React.MouseEvent,
    layout: ParkingSlotLayout,
    status: ParkingStatus
  ) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left + 15; // 15px offset
    const y = e.clientY - rect.top + 15;

    setTooltip({
      visible: true,
      x,
      y,
      id: layout.id,
      zone: layout.zone,
      status,
    });
  };

  const handleHoverLeave = () => {
    setTooltip((prev) => ({ ...prev, visible: false }));
  };

  const handleHoverMove = (e: React.MouseEvent) => {
    if (!containerRef.current || !tooltip.visible) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left + 15;
    const y = e.clientY - rect.top + 15;

    setTooltip((prev) => ({ ...prev, x, y }));
  };

  const handleSelect = (id: string) => {
    if (onSelectSpot) {
      onSelectSpot(id);
    }
  };

  // Find layout of the selected spot to render outline on top
  const selectedLayout = parkingLayout.find((slot) => slot.id === selectedSpotId);
  const selectedStatus = selectedSpotId ? getSpotStatus(selectedSpotId) : null;
  const vehicleMarkers = useMemo(
    () => getVectorVehicleMarkers(activeVehicles, backendSlots, parkingLayout, spots),
    [activeVehicles, backendSlots, spots]
  );

  useEffect(() => {
    const now = Date.now();

    setDisplayVehicleMarkers((previousMarkers) => {
      const nextById = new Map(vehicleMarkers.map((marker) => [marker.id, marker]));
      const previousById = new Map(previousMarkers.map((marker) => [marker.id, marker]));
      const merged = new Map<string, DisplayVehicleMarker>();

      for (const marker of vehicleMarkers) {
        const previous = previousById.get(marker.id);
        merged.set(marker.id, {
          ...marker,
          isVisible: !marker.isParked,
          lastSeenAt: now,
          x: marker.isParked && previous ? previous.x : marker.x,
          y: marker.isParked && previous ? previous.y : marker.y,
        });
      }

      for (const previous of previousMarkers) {
        if (nextById.has(previous.id)) continue;

        const isWithinGrace = now - previous.lastSeenAt <= MISSING_VEHICLE_GRACE_MS;
        if (previous.isVisible && isWithinGrace) {
          merged.set(previous.id, previous);
        } else if (previous.isVisible) {
          merged.set(previous.id, { ...previous, isVisible: false });
        } else {
          merged.set(previous.id, previous);
        }
      }

      return Array.from(merged.values());
    });
  }, [vehicleMarkers]);

  useEffect(() => {
    if (displayVehicleMarkers.length === 0) return;

    const timeoutId = window.setTimeout(() => {
      const now = Date.now();
      setDisplayVehicleMarkers((markers) =>
        markers
          .map((marker) => {
            if (!marker.isVisible) return marker;
            if (now - marker.lastSeenAt <= MISSING_VEHICLE_GRACE_MS) return marker;
            return { ...marker, isVisible: false };
          })
          .filter((marker) => marker.isVisible || now - marker.lastSeenAt <= MISSING_VEHICLE_GRACE_MS + MARKER_FADE_MS)
      );
    }, MARKER_FADE_MS);

    return () => window.clearTimeout(timeoutId);
  }, [displayVehicleMarkers]);

  return (
    <div className="parking-map-wrapper">
      <div className="parking-map-container" ref={containerRef}>
        {/* Background Image */}
        <img
          src={parkingBg}
          alt="Smart Parking Lot Background"
          className="parking-map-bg"
        />

        {/* SVG Overlay */}
        <svg
          viewBox="0 0 1672 941"
          preserveAspectRatio="xMidYMid meet"
          className="parking-map-svg"
        >
          {/* Render regular slots */}
          {parkingLayout.map((slot) => {
            const status = getSpotStatus(slot.id);
            const isSelected = slot.id === selectedSpotId;
            return (
              <ParkingSlot
                key={slot.id}
                layout={slot}
                status={status}
                isSelected={isSelected}
                onSelect={handleSelect}
                onHoverEnter={handleHoverEnter}
                onHoverLeave={handleHoverLeave}
                onHoverMove={handleHoverMove}
              />
            );
          })}

          {/* Render selected slot outline again on top so glow is not clipped or covered */}
          {selectedLayout && selectedStatus && (
            <g pointerEvents="none">
              <polygon
                points={selectedLayout.points}
                className={`parking-slot-poly status-${selectedStatus} is-selected`}
                style={{ fill: 'none' }}
              />
            </g>
          )}

          {displayVehicleMarkers.length > 0 && (
            <g className="vehicle-dot-layer" pointerEvents="none">
              {displayVehicleMarkers.map((vehicle) => (
                <g
                  key={vehicle.id}
                  className={`vehicle-dot-group ${vehicle.isVisible ? 'is-visible' : 'is-fading'}`}
                  style={{ transform: `translate(${vehicle.x}px, ${vehicle.y}px)` }}
                >
                  <circle r="12" className="vehicle-dot-pulse" />
                  <circle r="8" className="vehicle-dot" />
                  <text y="3" className="vehicle-dot-label">
                    {vehicle.trackId}
                  </text>
                </g>
              ))}
            </g>
          )}
        </svg>

        {/* Dynamic Tooltip */}
        {tooltip.visible && (
          <div
            className="parking-tooltip"
            style={{
              left: `${tooltip.x}px`,
              top: `${tooltip.y}px`,
            }}
          >
            <div className="parking-tooltip-id">Vị trí: {tooltip.id}</div>
            <div className="parking-tooltip-detail">
              <span className="parking-tooltip-label">Khu vực:</span>
              <span className="parking-tooltip-value">
                {zoneTranslation[tooltip.zone] || tooltip.zone}
              </span>
            </div>
            <div className="parking-tooltip-detail">
              <span className="parking-tooltip-label">Trạng thái:</span>
              <span className={`parking-tooltip-value status-${tooltip.status}`}>
                {statusTranslation[tooltip.status]}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Legend below the map */}
      <ParkingLegend />
    </div>
  );
};
