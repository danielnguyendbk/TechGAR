import type { WorldPoint } from '@/lib/domain/types';

export interface RouteInstruction {
  readonly at: WorldPoint;
  readonly text: string;
}

export function buildInstructions(route: readonly WorldPoint[]): RouteInstruction[] {
  if (route.length < 2) return [];
  const instructions: RouteInstruction[] = [{ at: route[0], text: 'Bắt đầu đi theo tuyến màu xanh.' }];
  for (let index = 1; index < route.length - 1; index += 1) {
    const before = route[index - 1];
    const current = route[index];
    const after = route[index + 1];
    const ax = current[0] - before[0];
    const ay = current[1] - before[1];
    const bx = after[0] - current[0];
    const by = after[1] - current[1];
    const cross = ax * by - ay * bx;
    if (Math.abs(cross) < 1e-6) continue;
    instructions.push({ at: current, text: cross > 0 ? 'Rẽ trái ở phía trước.' : 'Rẽ phải ở phía trước.' });
  }
  instructions.push({ at: route[route.length - 1], text: 'Bạn đã đến ô đỗ đã chọn.' });
  return instructions;
}

