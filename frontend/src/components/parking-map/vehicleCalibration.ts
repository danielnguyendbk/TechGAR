import type {
  ActiveVehicle,
  BackendParkingSlot,
  BackendPoint,
  ParkingSlotLayout,
  ParkingSpotState,
} from '../../types/parking';

interface VectorPoint {
  x: number;
  y: number;
}

interface AffineTransform {
  ax: number;
  bx: number;
  cx: number;
  ay: number;
  by: number;
  cy: number;
}

type TransformByZone = Partial<Record<string, AffineTransform>>;

export interface VectorVehicleMarker {
  id: string;
  trackId: number;
  x: number;
  y: number;
  isParked: boolean;
}

const ZONE_PREFIXES = ['L', 'M', 'R'];

const parsePoints = (points?: string): VectorPoint[] => {
  if (!points) return [];
  return points
    .trim()
    .split(/\s+/)
    .map((pair) => {
      const [x, y] = pair.split(',').map(Number);
      return { x, y };
    })
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
};

const polygonCenter = (points: VectorPoint[]): VectorPoint | null => {
  if (points.length === 0) return null;
  return {
    x: points.reduce((total, point) => total + point.x, 0) / points.length,
    y: points.reduce((total, point) => total + point.y, 0) / points.length,
  };
};

const solve3x3 = (matrix: number[][], values: number[]): [number, number, number] | null => {
  const rows = matrix.map((row, index) => [...row, values[index]]);

  for (let col = 0; col < 3; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < 3; row += 1) {
      if (Math.abs(rows[row][col]) > Math.abs(rows[pivot][col])) {
        pivot = row;
      }
    }

    if (Math.abs(rows[pivot][col]) < 1e-8) return null;

    [rows[col], rows[pivot]] = [rows[pivot], rows[col]];
    const pivotValue = rows[col][col];
    for (let i = col; i < 4; i += 1) {
      rows[col][i] /= pivotValue;
    }

    for (let row = 0; row < 3; row += 1) {
      if (row === col) continue;
      const factor = rows[row][col];
      for (let i = col; i < 4; i += 1) {
        rows[row][i] -= factor * rows[col][i];
      }
    }
  }

  return [rows[0][3], rows[1][3], rows[2][3]];
};

const fitAxis = (
  pairs: Array<{ source: BackendPoint; target: VectorPoint }>,
  targetAxis: 'x' | 'y'
): [number, number, number] | null => {
  const normal = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const rhs = [0, 0, 0];

  for (const pair of pairs) {
    const row = [pair.source.x, pair.source.y, 1];
    const target = pair.target[targetAxis];

    for (let i = 0; i < 3; i += 1) {
      rhs[i] += row[i] * target;
      for (let j = 0; j < 3; j += 1) {
        normal[i][j] += row[i] * row[j];
      }
    }
  }

  return solve3x3(normal, rhs);
};

const fitAffine = (
  pairs: Array<{ source: BackendPoint; target: VectorPoint }>
): AffineTransform | null => {
  if (pairs.length < 3) return null;

  const xFit = fitAxis(pairs, 'x');
  const yFit = fitAxis(pairs, 'y');
  if (!xFit || !yFit) return null;

  return {
    ax: xFit[0],
    bx: xFit[1],
    cx: xFit[2],
    ay: yFit[0],
    by: yFit[1],
    cy: yFit[2],
  };
};

const applyTransform = (transform: AffineTransform, point: BackendPoint): VectorPoint => ({
  x: transform.ax * point.x + transform.bx * point.y + transform.cx,
  y: transform.ay * point.x + transform.by * point.y + transform.cy,
});

const distanceSquared = (a: BackendPoint, b: BackendPoint): number => {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
};

const findNearestSlot = (
  point: BackendPoint,
  backendSlots: BackendParkingSlot[]
): BackendParkingSlot | null => {
  let nearest: BackendParkingSlot | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (const slot of backendSlots) {
    const distance = distanceSquared(point, slot.center);
    if (distance < nearestDistance) {
      nearest = slot;
      nearestDistance = distance;
    }
  }

  return nearest;
};

const pointInPolygon = (point: BackendPoint, polygon: BackendPoint[]): boolean => {
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersects =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;

    if (intersects) inside = !inside;
  }

  return inside;
};

export const buildTransformsByZone = (
  backendSlots: BackendParkingSlot[],
  vectorSlots: ParkingSlotLayout[]
): TransformByZone => {
  const vectorCenters = new Map<string, VectorPoint>();

  for (const slot of vectorSlots) {
    const center = polygonCenter(parsePoints(slot.points));
    if (center) vectorCenters.set(slot.id, center);
  }

  return ZONE_PREFIXES.reduce<TransformByZone>((transforms, prefix) => {
    const pairs = backendSlots
      .filter((slot) => slot.id.startsWith(prefix))
      .map((slot) => {
        const target = vectorCenters.get(slot.id);
        if (!target) return null;
        return { source: slot.center, target };
      })
      .filter((pair): pair is { source: BackendPoint; target: VectorPoint } => pair !== null);

    const transform = fitAffine(pairs);
    if (transform) transforms[prefix] = transform;
    return transforms;
  }, {});
};

export const getVectorVehicleMarkers = (
  vehicles: ActiveVehicle[],
  backendSlots: BackendParkingSlot[],
  vectorSlots: ParkingSlotLayout[],
  spots: ParkingSpotState[]
): VectorVehicleMarker[] => {
  const transforms = buildTransformsByZone(backendSlots, vectorSlots);
  const spotStatusById = new Map(spots.map((spot) => [spot.id, spot.status]));

  return vehicles
    .map((vehicle) => {
      const vehiclePoint = { x: vehicle.x, y: vehicle.y };
      const containingOccupiedSlot = backendSlots.find(
        (slot) =>
          spotStatusById.get(slot.id) === 'occupied' &&
          pointInPolygon(vehiclePoint, slot.polygon)
      );

      const nearestSlot = findNearestSlot(vehiclePoint, backendSlots);
      const zone = nearestSlot?.id[0];
      const transform = zone ? transforms[zone] : null;
      if (!transform) return null;

      const vectorPoint = applyTransform(transform, vehiclePoint);
      return {
        id: vehicle.id,
        trackId: vehicle.trackId,
        x: vectorPoint.x,
        y: vectorPoint.y,
        isParked: Boolean(containingOccupiedSlot),
      };
    })
    .filter((marker): marker is VectorVehicleMarker => marker !== null);
};
