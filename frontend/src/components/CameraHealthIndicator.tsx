import { Camera, CircleAlert } from "lucide-react";
import type { CameraState } from "../domain/parking";

interface CameraHealthIndicatorProps {
  cameras: Record<string, CameraState>;
  compact?: boolean;
}

export function CameraHealthIndicator({ cameras, compact = false }: CameraHealthIndicatorProps) {
  const onlineCount = Object.values(cameras).filter((camera) => camera.health === "online").length;
  const degraded = onlineCount < 2;
  const Icon = degraded ? CircleAlert : Camera;

  return (
    <div
      className={`camera-health ${degraded ? "camera-health--degraded" : "camera-health--online"}`}
      aria-label={`Camera ${onlineCount}/2 ${degraded ? "online, dữ liệu suy giảm" : "online"}`}
      data-testid="camera-health"
    >
      <Icon size={compact ? 18 : 24} aria-hidden="true" />
      <span>
        <small>Camera</small>
        <strong>{onlineCount}/2 online</strong>
      </span>
    </div>
  );
}
