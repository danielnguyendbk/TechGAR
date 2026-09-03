import type { SvgPoint } from './affine';

export const VIEWBOX = { width: 920, height: 560 } as const;

function polygon(index: number): readonly SvgPoint[] {
  const row = index < 12 ? 0 : 1;
  const column = index % 12;
  const x = 40 + column * 69;
  const y = row === 0 ? 42 : 420;
  return [[x, y], [x + 48, y], [x + 48, y + 92], [x, y + 92]];
}

export const PARKING_GEOMETRY: Readonly<Record<string, readonly SvgPoint[]>> = Object.fromEntries(
  Array.from({ length: 24 }, (_, index) => [
    `D${String(index + 1).padStart(2, '0')}`,
    polygon(index),
  ]),
);

