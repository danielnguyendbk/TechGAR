import {
  MAIN_ZONE_ORDER,
  formatSpotId,
  type DestinationNeed,
  type MainZoneId,
  type ParkingRow,
  type SpotId,
  type ZoneId,
} from "../domain/parking";

export interface Point {
  x: number;
  y: number;
}

export interface Rect extends Point {
  width: number;
  height: number;
}

export interface SpotGeometry extends Rect {
  id: SpotId;
  zone: ZoneId;
  number: number;
  row: ParkingRow;
  entryPoint: Point;
}

export interface ZoneGeometry {
  id: MainZoneId;
  bounds: Rect;
  laneY: number;
  upperSpotIds: SpotId[];
  lowerSpotIds: SpotId[];
}

export interface RoadConnectorGeometry {
  id: string;
  bounds: Rect;
  centerline: {
    start: Point;
    end: Point;
  };
}

export interface ParkingLayout {
  zoneLeftX: number;
  zoneRightX: number;
  mainRoadX: number;
  mainRoadWidth: number;
  mainRoadCenterX: number;
  zoneFLeftX: number;
  zoneFWidth: number;
  entranceY: number;
  exitY: number;
  zoneLaneY: Record<MainZoneId, number>;
}

export interface AccessAnchor extends Point {
  id: DestinationNeed;
  label: string;
}

export interface ParkingGeometry {
  width: number;
  height: number;
  layout: ParkingLayout;
  spots: SpotGeometry[];
  zones: ZoneGeometry[];
  zoneConnectors: RoadConnectorGeometry[];
  fAccessConnectors: RoadConnectorGeometry[];
  fStripBounds: Rect;
  accessRoad: Rect;
  entrance: Point;
  exit: Point;
  anchors: Record<DestinationNeed, AccessAnchor>;
}

const MAP_WIDTH = 1200;
const MAP_HEIGHT = 900;
const ZONE_X = 58;
const ZONE_WIDTH = 870;
const ZONE_HEIGHT = 142;
const ZONE_GAP = 24;
const FIRST_ZONE_Y = 34;
const SPOT_X = 92;
const SPOT_WIDTH = 50;
const SPOT_HEIGHT = 34;
const SPOT_GAP = 6;
const MAIN_ROAD_X = 950;
const MAIN_ROAD_WIDTH = 94;
const MAIN_ROAD_CENTER_X = MAIN_ROAD_X + MAIN_ROAD_WIDTH / 2;
const ZONE_F_LEFT_X = 1070;
const ZONE_F_WIDTH = 92;
const ENTRANCE_Y = 858;
const EXIT_Y = 28;
const CONNECTOR_HEIGHT = 46;

function createMainZone(zone: MainZoneId, zoneIndex: number): { zone: ZoneGeometry; spots: SpotGeometry[] } {
  const y = FIRST_ZONE_Y + zoneIndex * (ZONE_HEIGHT + ZONE_GAP);
  const laneY = y + ZONE_HEIGHT / 2;
  const upperY = y + 18;
  const lowerY = y + ZONE_HEIGHT - 18 - SPOT_HEIGHT;
  const spots: SpotGeometry[] = [];
  const upperSpotIds: SpotId[] = [];
  const lowerSpotIds: SpotId[] = [];

  for (let column = 0; column < 15; column += 1) {
    const x = SPOT_X + column * (SPOT_WIDTH + SPOT_GAP);
    const topNumber = column + 1;
    const bottomNumber = column + 16;
    const topId = formatSpotId(zone, topNumber);
    const bottomId = formatSpotId(zone, bottomNumber);

    upperSpotIds.push(topId);
    lowerSpotIds.push(bottomId);
    spots.push({
      id: topId,
      zone,
      number: topNumber,
      row: "top",
      x,
      y: upperY,
      width: SPOT_WIDTH,
      height: SPOT_HEIGHT,
      entryPoint: { x: x + SPOT_WIDTH / 2, y: laneY - 15 },
    });
    spots.push({
      id: bottomId,
      zone,
      number: bottomNumber,
      row: "bottom",
      x,
      y: lowerY,
      width: SPOT_WIDTH,
      height: SPOT_HEIGHT,
      entryPoint: { x: x + SPOT_WIDTH / 2, y: laneY + 15 },
    });
  }

  return {
    zone: {
      id: zone,
      bounds: { x: ZONE_X, y, width: ZONE_WIDTH, height: ZONE_HEIGHT },
      laneY,
      upperSpotIds,
      lowerSpotIds,
    },
    spots,
  };
}

export function generateParkingGeometry(): ParkingGeometry {
  const zones: ZoneGeometry[] = [];
  const spots: SpotGeometry[] = [];

  MAIN_ZONE_ORDER.forEach((zoneId, index) => {
    const generated = createMainZone(zoneId, index);
    zones.push(generated.zone);
    spots.push(...generated.spots);
  });

  const fStripBounds: Rect = { x: ZONE_F_LEFT_X, y: 118, width: ZONE_F_WIDTH, height: 632 };
  for (let index = 0; index < 10; index += 1) {
    const number = index + 1;
    const id = formatSpotId("F", number);
    const y = fStripBounds.y + 18 + index * 59;
    spots.push({
      id,
      zone: "F",
      number,
      row: "vertical",
      x: fStripBounds.x + 13,
      y,
      width: 66,
      height: 45,
      entryPoint: { x: fStripBounds.x + 5, y: y + 22.5 },
    });
  }

  const zoneConnectors: RoadConnectorGeometry[] = zones.map((zone) => {
    const connectorStartX = zone.bounds.x + zone.bounds.width - 10;
    return {
      id: `connector-${zone.id}`,
      bounds: {
        x: connectorStartX,
        y: zone.laneY - CONNECTOR_HEIGHT / 2,
        width: MAIN_ROAD_CENTER_X - connectorStartX,
        height: CONNECTOR_HEIGHT,
      },
      centerline: {
        start: { x: connectorStartX, y: zone.laneY },
        end: { x: MAIN_ROAD_CENTER_X, y: zone.laneY },
      },
    };
  });

  const fAccessConnectors: RoadConnectorGeometry[] = spots
    .filter((spot) => spot.zone === "F")
    .map((spot) => ({
      id: `connector-${spot.id}`,
      bounds: {
        x: MAIN_ROAD_CENTER_X,
        y: spot.entryPoint.y - 9,
        width: spot.entryPoint.x - MAIN_ROAD_CENTER_X,
        height: 18,
      },
      centerline: {
        start: { x: MAIN_ROAD_CENTER_X, y: spot.entryPoint.y },
        end: { ...spot.entryPoint },
      },
    }));

  const zoneLaneY = Object.fromEntries(zones.map((zone) => [zone.id, zone.laneY])) as Record<MainZoneId, number>;

  return {
    width: MAP_WIDTH,
    height: MAP_HEIGHT,
    layout: {
      zoneLeftX: ZONE_X,
      zoneRightX: ZONE_X + ZONE_WIDTH,
      mainRoadX: MAIN_ROAD_X,
      mainRoadWidth: MAIN_ROAD_WIDTH,
      mainRoadCenterX: MAIN_ROAD_CENTER_X,
      zoneFLeftX: ZONE_F_LEFT_X,
      zoneFWidth: ZONE_F_WIDTH,
      entranceY: ENTRANCE_Y,
      exitY: EXIT_Y,
      zoneLaneY,
    },
    spots,
    zones,
    zoneConnectors,
    fAccessConnectors,
    fStripBounds,
    accessRoad: { x: MAIN_ROAD_X, y: 0, width: MAIN_ROAD_WIDTH, height: MAP_HEIGHT },
    entrance: { x: MAIN_ROAD_CENTER_X, y: ENTRANCE_Y },
    exit: { x: MAIN_ROAD_CENTER_X, y: EXIT_Y },
    anchors: {
      shopping: { id: "shopping", label: "Shopping", x: 430, y: 882 },
      services: { id: "services", label: "Dịch vụ", x: 30, y: 450 },
      entertainment: { id: "entertainment", label: "Giải trí", x: 430, y: 18 },
    },
  };
}

export const PARKING_GEOMETRY = generateParkingGeometry();

export const SPOT_GEOMETRY_BY_ID = new Map<SpotId, SpotGeometry>(
  PARKING_GEOMETRY.spots.map((spot) => [spot.id, spot]),
);

export function pointInsideRect(point: Point, rect: Rect): boolean {
  return (
    point.x > rect.x &&
    point.x < rect.x + rect.width &&
    point.y > rect.y &&
    point.y < rect.y + rect.height
  );
}

export function distanceBetween(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
