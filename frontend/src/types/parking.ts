export type ParkingStatus = 'available' | 'occupied' | 'incoming' | 'unavailable';

export interface ParkingSlotLayout {
  id: string;
  zone: string;
  points?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  cx?: number;   // center X in 1672x941 space
  cy?: number;   // center Y in 1672x941 space
  rotation?: number;
}

export interface ParkingSpotState {
  id: string;
  status: ParkingStatus;
}

export interface ActiveVehicle {
  id: string;
  trackId: number;
  x: number;
  y: number;
  status: string;
}

export interface BackendPoint {
  x: number;
  y: number;
}

export interface BackendParkingSlot {
  id: string;
  polygon: BackendPoint[];
  center: BackendPoint;
}

export interface BackendParkingSlotsResponse {
  imageWidth: number;
  imageHeight: number;
  slots: BackendParkingSlot[];
}
