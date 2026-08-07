// voiceGuidance.ts - Bộ Hướng dẫn Giọng nói Web Speech API & Cảnh báo Đi Sai Đường

export interface VoiceOptions {
  muted?: boolean;
}

class VoiceManager {
  private lastSpokenText: string = "";
  private lastSpokenTime: number = 0;
  private isMuted: boolean = false;

  public setMuted(muted: boolean) {
    this.isMuted = muted;
    if (muted && typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }

  public getMuted(): boolean {
    return this.isMuted;
  }

  public stop() {
    this.lastSpokenText = "";
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel();
      } catch (_) {}
    }
  }

  public speak(text: string, cooldownMs: number = 6000) {
    if (this.isMuted) return;
    const now = Date.now();
    // Tránh lặp lại câu chính xác khi chưa hết thời gian chờ cooldown (đặc biệt ngắt vấp tiếng Cảnh Cảnh...)
    if (text === this.lastSpokenText && now - this.lastSpokenTime < cooldownMs) {
      return;
    }

    this.lastSpokenText = text;
    this.lastSpokenTime = now;

    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel(); // Ngắt câu cũ trước khi phát câu mới khác
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "vi-VN";
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
      } catch (_) {}
    }
  }
}

export const voiceManager = new VoiceManager();

/**
 * Tính khoảng cách vuông góc từ điểm (px, py) tới đoạn thẳng (x1,y1)-(x2,y2)
 */
export function distanceToSegment(
  px: number,
  py: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  if (dx === 0 && dy === 0) return Math.hypot(px - x1, py - y1);
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
  const projX = x1 + t * dx;
  const projY = y1 + t * dy;
  return Math.hypot(px - projX, py - projY);
}

/**
 * Kiểm tra xem xe có đi sai tuyến đường không (khoảng cách vượt quá ngưỡng thresholdPx)
 */
export function checkIsOffRoute(
  vehiclePos: { x: number; y: number },
  routePoints: Array<{ x: number; y: number }>,
  thresholdPx: number = 75
): boolean {
  if (!vehiclePos || routePoints.length < 2) return false;
  let minDistance = Number.POSITIVE_INFINITY;
  for (let i = 0; i < routePoints.length - 1; i++) {
    const p1 = routePoints[i];
    const p2 = routePoints[i + 1];
    if (p1 && p2) {
      const d = distanceToSegment(vehiclePos.x, vehiclePos.y, p1.x, p1.y, p2.x, p2.y);
      if (d < minDistance) minDistance = d;
    }
  }
  return minDistance > thresholdPx;
}

/**
 * Tính toán câu lệnh rẽ trái / rẽ phải / đi thẳng dựa trên điểm tiếp theo của lộ trình
 */
export function getNavigationInstruction(
  vehiclePos: { x: number; y: number },
  routePoints: Array<{ x: number; y: number }>,
  isExit: boolean = false,
  targetSpotId: string | null = null
): string | null {
  if (!vehiclePos || routePoints.length < 2) return null;

  // 1. Kiểm tra xe đã đến đích chưa (gần điểm cuối < 35px)
  const destPoint = routePoints[routePoints.length - 1];
  if (destPoint) {
    const distToDest = Math.hypot(vehiclePos.x - destPoint.x, vehiclePos.y - destPoint.y);
    if (distToDest < 35) {
      if (isExit) {
        return "Bạn đã đến lối ra. Chúc bạn thượng lộ bình an!";
      } else if (targetSpotId) {
        return `Bạn đã đến vị trí ô đỗ ${targetSpotId}. Vui lòng lùi xe vào đỗ.`;
      } else {
        return "Bạn đã đến điểm đích.";
      }
    }
  }

  // 2. Tìm điểm nút tiếp theo gần xe nhất
  let closestIndex = 0;
  let minDist = Number.POSITIVE_INFINITY;
  for (let i = 0; i < routePoints.length; i++) {
    const pt = routePoints[i];
    if (pt) {
      const d = Math.hypot(vehiclePos.x - pt.x, vehiclePos.y - pt.y);
      if (d < minDist) {
        minDist = d;
        closestIndex = i;
      }
    }
  }

  // Nếu còn điểm tiếp theo trên tuyến đường
  if (closestIndex < routePoints.length - 1) {
    const pCurrent = routePoints[closestIndex];
    const pNext = routePoints[closestIndex + 1];

    if (pCurrent && pNext) {
      // Tính hướng góc đi của đoạn đường tiếp theo
      const angleRad = Math.atan2(pNext.y - pCurrent.y, pNext.x - pCurrent.x);
    const angleDeg = (angleRad * 180) / Math.PI;

    // Phân tích hướng cơ bản
    if (isExit) {
      return "Tiếp tục đi theo đường dẫn ra cổng xuất bãi.";
    }

    if (angleDeg > -45 && angleDeg <= 45) {
      return "Phía trước rẽ phải vào làn đỗ.";
    } else if (angleDeg > 45 && angleDeg <= 135) {
      return "Phía trước đi thẳng.";
    } else if (angleDeg < -45 && angleDeg >= -135) {
      return "Phía trước đi thẳng.";
    } else {
      return "Phía trước rẽ trái vào làn đỗ.";
    }
  }
}

  return "Tiếp tục di chuyển theo đường chỉ dẫn.";
}
