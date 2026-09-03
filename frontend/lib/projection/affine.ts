import type { SlotLayout, WorldPoint } from '@/lib/domain/types';

export type SvgPoint = readonly [number, number];

export interface AffineTransform {
  readonly matrix: readonly [number, number, number, number, number, number];
  readonly valid: boolean;
  readonly residualRms: number;
  project(point: WorldPoint): SvgPoint;
  unproject(point: SvgPoint): WorldPoint | null;
}

export interface PointPair {
  readonly world: WorldPoint;
  readonly svg: SvgPoint;
}

function solve3(matrix: number[][], vector: number[]): readonly [number, number, number] | null {
  const augmented = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < 3; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < 3; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) < 1e-9) return null;
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    for (let k = column; k < 4; k += 1) augmented[column][k] /= divisor;
    for (let row = 0; row < 3; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let k = column; k < 4; k += 1) augmented[row][k] -= factor * augmented[column][k];
    }
  }
  return [augmented[0][3], augmented[1][3], augmented[2][3]];
}

export function identityAffine(): AffineTransform {
  return {
    matrix: [1, 0, 0, 0, 1, 0],
    valid: false,
    residualRms: Number.POSITIVE_INFINITY,
    project: ([x, y]) => [x, y],
    unproject: ([x, y]) => [x, y],
  };
}

export function fitAffine(pairs: readonly PointPair[]): AffineTransform {
  if (pairs.length < 3) return identityAffine();
  let xx = 0; let xy = 0; let yy = 0; let x = 0; let y = 0;
  let sxX = 0; let syX = 0; let sX = 0;
  let sxY = 0; let syY = 0; let sY = 0;
  for (const pair of pairs) {
    const [wx, wy] = pair.world;
    const [vx, vy] = pair.svg;
    xx += wx * wx; xy += wx * wy; yy += wy * wy; x += wx; y += wy;
    sxX += wx * vx; syX += wy * vx; sX += vx;
    sxY += wx * vy; syY += wy * vy; sY += vy;
  }
  const normal = [[xx, xy, x], [xy, yy, y], [x, y, pairs.length]];
  const horizontal = solve3(normal.map((row) => [...row]), [sxX, syX, sX]);
  const vertical = solve3(normal.map((row) => [...row]), [sxY, syY, sY]);
  if (!horizontal || !vertical) return identityAffine();
  const [a, b, tx] = horizontal;
  const [c, d, ty] = vertical;
  const determinant = a * d - b * c;
  if (Math.abs(determinant) < 1e-9) return identityAffine();
  const project = ([wx, wy]: WorldPoint): SvgPoint => [a * wx + b * wy + tx, c * wx + d * wy + ty];
  const unproject = ([vx, vy]: SvgPoint): WorldPoint => {
    const px = vx - tx;
    const py = vy - ty;
    return [(d * px - b * py) / determinant, (-c * px + a * py) / determinant];
  };
  const squared = pairs.reduce((total, pair) => {
    const projected = project(pair.world);
    return total + (projected[0] - pair.svg[0]) ** 2 + (projected[1] - pair.svg[1]) ** 2;
  }, 0);
  return {
    matrix: [a, b, tx, c, d, ty],
    valid: true,
    residualRms: Math.sqrt(squared / pairs.length),
    project,
    unproject,
  };
}

function center(points: readonly (WorldPoint | SvgPoint)[]): readonly [number, number] {
  const total = points.reduce<readonly [number, number]>(
    (sum, point) => [sum[0] + point[0], sum[1] + point[1]], [0, 0],
  );
  return [total[0] / points.length, total[1] / points.length];
}

export function fitSlotProjection(
  layout: readonly SlotLayout[],
  geometry: Readonly<Record<string, readonly SvgPoint[]>>,
): AffineTransform {
  const pairs: PointPair[] = [];
  for (const slot of layout) {
    const target = geometry[slot.slot_id];
    if (!target || slot.polygon.length < 3) continue;
    pairs.push({ world: center(slot.polygon) as WorldPoint, svg: center(target) as SvgPoint });
  }
  return fitAffine(pairs);
}

