import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { ParkingMap } from '@/components/parking-map';
import { makeDemoSnapshot } from '@/lib/demo-data';
import type { RuntimeSnapshot } from '@/lib/domain/types';

function largeSnapshot(): RuntimeSnapshot {
  const base = makeDemoSnapshot();
  const slot_layout = Array.from({ length: 160 }, (_, index) => {
    const column = index % 16;
    const row = Math.floor(index / 16);
    const x = 5 + column * 5.5;
    const y = 5 + row * 6;
    return { slot_id: `P${String(index + 1).padStart(3, '0')}`, camera_id: row < 5 ? 'C1' : 'C2', polygon: [[x, y], [x + 4, y], [x + 4, y + 5], [x, y + 5]] } as const;
  });
  return {
    ...base,
    vehicles: [],
    slot_layout,
    parking_slots: slot_layout.map((slot) => ({ slot_id: slot.slot_id, occupied: false, status: 'empty' as const, owning_global_id: null, overlap_score: 0, dwell_duration: 0, confirmation_confidence: 0 })),
  };
}

describe('SVG render performance', () => {
  it('renders 160 slots below the 100 ms p95 gate', () => {
    const snapshot = largeSnapshot();
    const samples: number[] = [];
    renderToStaticMarkup(<ParkingMap snapshot={snapshot} vehicles={[]} />);
    for (let run = 0; run < 20; run += 1) {
      const start = performance.now();
      renderToStaticMarkup(<ParkingMap snapshot={snapshot} vehicles={[]} />);
      samples.push(performance.now() - start);
    }
    samples.sort((a, b) => a - b);
    const p95 = samples[Math.ceil(samples.length * 0.95) - 1];
    expect(p95).toBeLessThanOrEqual(100);
  });
});
