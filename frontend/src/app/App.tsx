import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { EntryQRKiosk } from "../components/EntryQRKiosk";
import { BrowseToolbar } from "../components/BrowseToolbar";
import { EntryChoiceSheet } from "../components/EntryChoiceSheet";
import { InvalidSpotWarningSheet } from "../components/InvalidSpotWarningSheet";
import { MockControlPanel } from "../components/MockControlPanel";
import { NavigationStatusBar } from "../components/NavigationStatusBar";
import { ParkingLegend } from "../components/ParkingLegend";
import { ParkingMap } from "../components/ParkingMap";
import { RecommendationPanel } from "../components/RecommendationPanel";
import { SmartParkingHeader } from "../components/SmartParkingHeader";
import { SpotDetailSheet } from "../components/SpotDetailSheet";
import { SummaryCards } from "../components/SummaryCards";
import { getSpotOwner, type DestinationNeed, type ParkingSpotState, type ParkingStatus, type SpotId } from "../domain/parking";
import { mockParkingDataSource } from "../mocks/MockParkingDataSource";
import { recommendParkingSpots } from "../recommendation/recommendationEngine";
import { LANE_GRAPH } from "../routing/laneGraph";
import { findVehicleRoute, findExitRoute } from "../routing/routeEngine";
import { voiceManager, checkIsOffRoute, getNavigationInstruction } from "../routing/voiceGuidance";
import { useDriverFlowStore } from "../stores/driverFlowStore";
import { deriveParkingCounts, useParkingStore } from "../stores/parkingStore";

const NON_EMPTY_STATUSES: ReadonlySet<ParkingStatus> = new Set(["transitioning", "occupied", "unknown"]);

// ── Kiểu dữ liệu vị trí xe từ tracker ─────────────────────────────────────
export interface ActiveVehicle {
  trackId: number;
  x: number;
  y: number;
  trail: Array<{ x: number; y: number }>;
}

export interface FrameSize {
  width: number;
  height: number;
}

interface SessionInfo {
  sessionId: string;
  state: string;
  targetSpotId: string | null;
  parkedSpotId: string | null;
  vehicleTrackId: number | null;
  activeTrackId: number | null;
  claimed: boolean;
}

interface AppProps {
  sessionId?: string | null;
}

export function App({ sessionId }: AppProps = {}) {
  const [activeVehicles, setActiveVehicles] = useState<ActiveVehicle[]>([]);
  const [frameSize, setFrameSize]           = useState<FrameSize>({ width: 1200, height: 900 });
  const [sessionInfo, setSessionInfo]       = useState<SessionInfo | null>(null);

  const sessionTrackIdRef = useRef<number | null>(sessionId ? Number(sessionId) : null);

  const spotsById = useParkingStore((state) => state.spots);
  const cameras = useParkingStore((state) => state.cameras);
  const lastEventTime = useParkingStore((state) => state.lastEventTime);
  const applySnapshot = useParkingStore((state) => state.applySnapshot);
  const applyEvent = useParkingStore((state) => state.applyEvent);
  const trackingSource = useParkingStore((state) => state.trackingSource);

  const mode = useDriverFlowStore((state) => state.mode);
  const browseFilter = useDriverFlowStore((state) => state.browseFilter);
  const activeNeed = useDriverFlowStore((state) => state.activeNeed);
  const recommendation = useDriverFlowStore((state) => state.recommendation);
  const inspectedSpotId = useDriverFlowStore((state) => state.inspectedSpotId);
  const confirmedSpotId = useDriverFlowStore((state) => state.confirmedSpotId);
  const warning = useDriverFlowStore((state) => state.warning);
  const enterBrowse = useDriverFlowStore((state) => state.enterBrowse);
  const startRecommendation = useDriverFlowStore((state) => state.startRecommendation);
  const chooseNeed = useDriverFlowStore((state) => state.chooseNeed);
  const setRecommendation = useDriverFlowStore((state) => state.setRecommendation);
  const chooseRecommendedSpot = useDriverFlowStore((state) => state.chooseRecommendedSpot);
  const inspectSpot = useDriverFlowStore((state) => state.inspectSpot);
  const confirmSpot = useDriverFlowStore((state) => state.confirmSpot);
  const setBrowseFilter = useDriverFlowStore((state) => state.setBrowseFilter);
  const showInvalidSpotWarning = useDriverFlowStore((state) => state.showInvalidSpotWarning);
  const cancelNavigation = useDriverFlowStore((state) => state.cancelNavigation);

  const spots = useMemo(
    () => Object.values(spotsById).filter((spot): spot is ParkingSpotState => spot !== undefined),
    [spotsById],
  );
  const counts = useMemo(() => deriveParkingCounts(spots), [spots]);
  const inspectedSpot = inspectedSpotId ? spotsById[inspectedSpotId] : undefined;
  const confirmedSpot = confirmedSpotId ? spotsById[confirmedSpotId] : undefined;

  // ── Session values ──
  const sessionState = sessionInfo?.state ?? null;
  const sessionTargetSpot = sessionInfo?.targetSpotId ?? null;
  const sessionParkedSpot = sessionInfo?.parkedSpotId ?? null;
  const targetVehicleId = sessionInfo?.activeTrackId ?? sessionInfo?.vehicleTrackId ?? (sessionId ? Number(sessionId) : null);

  // ── Backend URL Helper ──
  const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || "").replace(/\/$/, "");
  const getBackendUrl = useCallback((path: string) => {
    const cleanPath = path.startsWith("/") ? path : `/${path}`;
    return BACKEND_URL ? `${BACKEND_URL}${cleanPath}` : cleanPath;
  }, [BACKEND_URL]);

  // ── Helper API call ──
  const callSessionApi = useCallback(async (endpoint: string, payload: object) => {
    try {
      const res = await fetch(getBackendUrl(`/api/session/${endpoint}`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (res.ok) return true;
    } catch (_) {
      try {
        const res = await fetch(`http://127.0.0.1:8000/api/session/${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (res.ok) return true;
      } catch (e) {
        console.warn("Session API call failed:", e);
      }
    }
    return false;
  }, [getBackendUrl]);

  // ── Auto-claim session khi mở trang cá nhân lần đầu ──
  const claimedRef = useRef(false);
  useEffect(() => {
    if (!sessionId || claimedRef.current) return;
    const doClaim = async () => {
      claimedRef.current = true;
      await callSessionApi("claim", { sessionId });
    };
    void doClaim();
  }, [sessionId, callSessionApi]);

  // ── Cập nhật target spot khi user confirm ──
  const updateSessionTarget = useCallback(async (spotId: SpotId) => {
    if (!sessionId) return;
    await callSessionApi("select", { sessionId, spotId });
  }, [sessionId, callSessionApi]);

  // ── Fetch dữ liệu ô đỗ & tracking feed ──
  useEffect(() => {
    let active = true;

    void mockParkingDataSource.getSnapshot().then((snapshot) => {
      if (active) applySnapshot(snapshot);
    });

    const fetchRealtimeStatus = async () => {
      try {
        const statusFile = trackingSource === "sample"
          ? "parking_status_sample.json"
          : "parking_status.json";
        
        const res = await fetch(getBackendUrl(`/${statusFile}?t=${Date.now()}`));
        if (!res.ok) {
          if (trackingSource === "sample") {
            const fallback = await fetch(getBackendUrl(`/parking_status.json?t=${Date.now()}`));
            if (!fallback.ok || !active) return;
            const data = await fallback.json();
            applyParkingData(data);
          }
          return;
        }
        if (!active) return;
        const data = await res.json();
        applyParkingData(data);
      } catch (_) {}
    };

    const applyParkingData = (data: any) => {
      if (!data) return;
      applyEvent({
        type: "camera.health.changed",
        cameraId: "cam-left",
        health: "online",
        updatedAt: data.timestamp || new Date().toISOString(),
      });
      applyEvent({
        type: "camera.health.changed",
        cameraId: "cam-right",
        health: "online",
        updatedAt: data.timestamp || new Date().toISOString(),
      });

      if (data.slots) {
        Object.keys(data.slots).forEach((spotId) => {
          const slotData = data.slots[spotId];
          applyEvent({
            type: "spot.status.changed",
            cameraId: getSpotOwner(spotId as SpotId),
            spotId: spotId as any,
            status: slotData.status,
            confidence: 0.99,
            revision: Date.now(),
            updatedAt: data.timestamp || new Date().toISOString(),
          });
        });
      }
    };

    void fetchRealtimeStatus();
    const interval = setInterval(fetchRealtimeStatus, 1000);

    // ── Polling session info (500ms) ──
    const fetchSessionInfo = async () => {
      if (!sessionId || !active) return;
      try {
        const res = await fetch(getBackendUrl(`/navigation_sessions.json?t=${Date.now()}`));
        if (!res.ok || !active) return;
        const sessions = await res.json();
        const s = sessions?.[sessionId];
        if (s && active) {
          setSessionInfo({
            sessionId: s.sessionId ?? sessionId,
            state: s.state ?? "WAITING_FOR_SCAN",
            targetSpotId: s.targetSpotId ?? null,
            parkedSpotId: s.parkedSpotId ?? null,
            vehicleTrackId: s.vehicleTrackId ?? null,
            activeTrackId: s.activeTrackId ?? null,
            claimed: s.claimed ?? false,
          });
          sessionTrackIdRef.current = s.activeTrackId ?? s.vehicleTrackId ?? Number(sessionId);
        }
      } catch (_) {}
    };

    if (sessionId) {
      void fetchSessionInfo();
    }
    const sessionInterval = sessionId ? setInterval(fetchSessionInfo, 500) : null;

    // ── Polling vehicle positions ──
    const STALE_THRESHOLD_MS = 5000;

    const fetchVehiclePositions = async () => {
      try {
        const file = trackingSource === "opencv" ? "vehicle_positions.json" : "vehicle_positions_sample.json";
        const res = await fetch(getBackendUrl(`/${file}?t=${Date.now()}`));
        if (!res.ok || !active) return;
        const data = await res.json();

        if (data?.timestamp && trackingSource === "opencv") {
          const dataAge = Date.now() - new Date(data.timestamp).getTime();
          if (dataAge > STALE_THRESHOLD_MS) {
            if (active) setActiveVehicles([]);
            return;
          }
        }

        let vehicles: ActiveVehicle[] = Object.entries(
          (data?.active_vehicles ?? {}) as Record<string, { position: { x: number; y: number }; trail?: Array<{ x: number; y: number }> }>
        ).map(([idStr, v]) => ({
          trackId: Number(idStr),
          x: v.position?.x ?? 0,
          y: v.position?.y ?? 0,
          trail: v.trail ?? [],
        }));

        // 🎯 NẾU TRANG CÁ NHÂN (có sessionId) → CHỈ GIỮ LẠI DUY NHẤT XE CỦA SESSION ĐÓ
        if (sessionId) {
          const targetId = sessionTrackIdRef.current ?? Number(sessionId);
          vehicles = vehicles.filter((v) => v.trackId === targetId);
        }

        if (data?.frame_size?.width && data?.frame_size?.height) {
          setFrameSize({ width: data.frame_size.width, height: data.frame_size.height });
        }
        if (active) setActiveVehicles(vehicles);
      } catch (_) {}
    };

    void fetchVehiclePositions();
    const vehicleInterval = setInterval(fetchVehiclePositions, 300);

    return () => {
      active = false;
      clearInterval(interval);
      clearInterval(vehicleInterval);
      if (sessionInterval) clearInterval(sessionInterval);
    };
  }, [applyEvent, applySnapshot, trackingSource, sessionId]);

  useEffect(() => {
    if (mode !== "recommendation" || !activeNeed || spots.length === 0) return;
    const result = recommendParkingSpots(spots, activeNeed, {
      calculatedAt: lastEventTime ?? "2026-07-25T08:00:00.000Z",
    });
    setRecommendation(result ?? undefined);
  }, [activeNeed, lastEventTime, mode, setRecommendation, spots]);

  useEffect(() => {
    if (mode !== "navigation" || !confirmedSpot || warning || !NON_EMPTY_STATUSES.has(confirmedSpot.status)) return;
    const need: DestinationNeed = activeNeed ?? "services";
    const alternatives = recommendParkingSpots(spots, need, {
      calculatedAt: lastEventTime ?? "2026-07-25T08:00:00.000Z",
    });
    const alternativeSpotId = [alternatives?.best, ...(alternatives?.alternatives ?? [])]
      .find((candidate) => candidate && candidate.spotId !== confirmedSpot.id)?.spotId;
    showInvalidSpotWarning({
      spotId: confirmedSpot.id,
      status: confirmedSpot.status as Exclude<ParkingStatus, "empty">,
      alternativeSpotId,
    });
  }, [activeNeed, confirmedSpot, lastEventTime, mode, showInvalidSpotWarning, spots, warning]);

  const [isRouteDismissed, setIsRouteDismissed] = useState<boolean>(false);

  // ── Tính đường đi (Planned Route) ──
  const route = useMemo(() => {
    if (isRouteDismissed || sessionState === "PARKED" || sessionState === "CLOSED" || sessionState === "WAITING_FOR_SCAN") return null;

    // 1. Chế độ lấy xe ra (Exit Navigation): Vẽ đường từ ô đỗ THỰC TẾ ra CỔNG EXIT
    if (sessionId && sessionState === "EXIT_NAVIGATION") {
      const exitSpot = (sessionParkedSpot || sessionTargetSpot || confirmedSpotId) as SpotId | null;
      if (exitSpot) {
        return findExitRoute(LANE_GRAPH, exitSpot);
      }
    }

    // 2. Chế độ dẫn đường vào ô đỗ (Inbound Navigation)
    const targetForRoute = (sessionId && sessionTargetSpot) ? sessionTargetSpot : confirmedSpotId;
    if (!targetForRoute) return null;

    if (sessionId && sessionTargetSpot) {
      return findVehicleRoute(LANE_GRAPH, targetForRoute as SpotId);
    }

    if (mode !== "navigation" || !confirmedSpotId || !confirmedSpot || confirmedSpot.status !== "empty" || warning) return null;
    return findVehicleRoute(LANE_GRAPH, confirmedSpotId);
  }, [isRouteDismissed, sessionId, sessionState, sessionParkedSpot, sessionTargetSpot, confirmedSpotId, confirmedSpot, mode, warning]);

  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [isOffRoute, setIsOffRoute] = useState<boolean>(false);
  const [navInstruction, setNavInstruction] = useState<string | null>(null);

  // ── Theo dõi Giọng nói Dẫn đường & Cảnh báo Đi sai đường ──
  useEffect(() => {
    // Tắt ngay giọng nói và đường chỉ dẫn nếu xe đã đỗ (PARKED), đã đóng phiên (CLOSED) hoặc ẩn đường
    if (
      !sessionId ||
      !route ||
      route.points.length < 2 ||
      isRouteDismissed ||
      (sessionState !== "NAVIGATING_TO_SPOT" && sessionState !== "EXIT_NAVIGATION")
    ) {
      setIsOffRoute(false);
      setNavInstruction(null);
      voiceManager.stop();
      return;
    }

    const targetVehicleId = sessionTrackIdRef.current ?? Number(sessionId);
    const targetVehicle = activeVehicles.find((v) => v.trackId === targetVehicleId);

    if (!targetVehicle) return;

    // 1. Kiểm tra xe đi sai đường (ngưỡng 80px)
    const offRoute = checkIsOffRoute({ x: targetVehicle.x, y: targetVehicle.y }, route.points, 80);
    setIsOffRoute(offRoute);

    if (offRoute) {
      // Cooldown 7000ms ngăn chặn giọng nói bị ngắt quãng giật lặp "Cảnh cảnh..."
      voiceManager.speak("Cảnh báo: Bạn đang đi sai tuyến đường chỉ dẫn!", 7000);
      setNavInstruction("⚠️ BẠN ĐANG ĐI SAI TUYẾN ĐƯỜNG CHỈ DẪN!");
    } else {
      // 2. Phát giọng nói hướng dẫn rẽ / đi thẳng / tới ô đỗ (cooldown 6000ms)
      const isExit = sessionState === "EXIT_NAVIGATION";
      const instruction = getNavigationInstruction(
        { x: targetVehicle.x, y: targetVehicle.y },
        route.points,
        isExit,
        sessionTargetSpot
      );

      setNavInstruction(instruction);
      if (instruction) {
        voiceManager.speak(instruction, 6000);
      }
    }
  }, [route, activeVehicles, sessionId, isRouteDismissed, sessionState, sessionTargetSpot, isMuted]);

  // ── Xử lý khi bấm vào ô đỗ ──
  const handleConfirmSpot = useCallback((spotId: SpotId): void => {
    setIsRouteDismissed(false);
    confirmSpot(spotId);
    if (sessionId) {
      void updateSessionTarget(spotId);
    }
  }, [confirmSpot, sessionId, updateSessionTarget]);

  const handleSpotClick = (spotId: SpotId): void => {
    const clickedSpot = spotsById[spotId];
    if (sessionId && clickedSpot?.status === "empty") {
      handleConfirmSpot(spotId);
      return;
    }
    if (mode === "browse") {
      inspectSpot(spotId);
      return;
    }
    if (mode === "recommendation" && recommendation) {
      const isCandidate = [recommendation.best, ...recommendation.alternatives].some((spot) => spot.spotId === spotId);
      if (isCandidate) chooseRecommendedSpot(spotId);
    }
  };

  const handleNeedChange = (need: DestinationNeed): void => chooseNeed(need);

  const switchAlternative = (): void => {
    if (!warning?.alternativeSpotId) return;
    const alternative = spotsById[warning.alternativeSpotId];
    if (alternative?.status === "empty") handleConfirmSpot(alternative.id);
  };

  const handleDismissRoute = useCallback(async () => {
    setIsRouteDismissed(false);
    cancelNavigation();
    voiceManager.stop();
    if (sessionId) {
      setSessionInfo((prev) => prev ? { ...prev, targetSpotId: null, state: "SELECTING_SPOT" } : null);
      await callSessionApi("select", { sessionId, spotId: null });
    }
  }, [sessionId, cancelNavigation, callSessionApi]);

  const handleStartExit = useCallback(async () => {
    if (!sessionId) return;
    setIsRouteDismissed(false);
    cancelNavigation();
    setSessionInfo((prev) => prev ? { ...prev, state: "EXIT_NAVIGATION", targetSpotId: null } : null);
    await callSessionApi("exit", { sessionId });
  }, [sessionId, cancelNavigation, callSessionApi]);

  const getSessionStatusLabel = () => {
    switch (sessionState) {
      case "WAITING_FOR_SCAN": return "ĐANG KẾT NỐI";
      case "SELECTING_SPOT":   return "ĐANG CHỌN Ô ĐỖ";
      case "NAVIGATING_TO_SPOT": return "ĐANG DẪN ĐƯỜNG";
      case "PARKED":           return "ĐÃ ĐỖ";
      case "EXIT_NAVIGATION":  return "ĐANG RA CỔNG";
      case "CLOSED":           return "ĐÃ HOÀN THÀNH";
      default:                 return "ĐANG TẢI...";
    }
  };

  const getSessionStatusColor = () => {
    switch (sessionState) {
      case "WAITING_FOR_SCAN": return "#64748b";
      case "SELECTING_SPOT":   return "#f59e0b";
      case "NAVIGATING_TO_SPOT": return "#3b82f6";
      case "PARKED":           return "#22c55e";
      case "EXIT_NAVIGATION":  return "#eab308";
      case "CLOSED":           return "#64748b";
    }
  };

  return (
    <div className="app-shell">
      <SmartParkingHeader mode={mode} cameras={cameras} lastUpdated={lastEventTime} />

      <main className="app-main">
        <SummaryCards counts={counts} cameras={cameras} />
        {mode === "browse" && (
          <BrowseToolbar filter={browseFilter} onFilterChange={setBrowseFilter} onFindSpot={startRecommendation} />
        )}
        {mode === "navigation" && confirmedSpot && (
          <NavigationStatusBar spotId={confirmedSpot.id} zone={confirmedSpot.zone} paused={Boolean(warning)} onCancel={cancelNavigation} />
        )}

        {/* ── Bảng trạng thái phiên làm việc (Session Status Banner) ── */}
        {sessionId && (
          <div style={{
            background: "rgba(15, 23, 42, 0.95)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            borderRadius: "12px",
            padding: "16px 24px",
            marginBottom: "16px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "16px",
            flexWrap: "wrap",
            backdropFilter: "blur(8px)",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
              <div style={{
                background: getSessionStatusColor(),
                width: "44px",
                height: "44px",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "20px",
                fontWeight: "bold",
                color: "#fff",
                flexShrink: 0,
              }}>
                {sessionState === "PARKED" ? "✓" : "🚗"}
              </div>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <h3 style={{ margin: 0, fontSize: "16px", color: "#f8fafc" }}>
                    Phiên xe #{targetVehicleId}
                  </h3>
                  <span style={{ fontSize: "12px", color: "#94a3b8", background: "rgba(255,255,255,0.08)", padding: "2px 8px", borderRadius: "4px" }}>
                    Mã session: {sessionId}
                  </span>
                </div>
                
                {sessionState === "EXIT_NAVIGATION" && (
                  <p style={{ margin: "4px 0 0 0", fontSize: "14px", color: "#fbbf24" }}>
                    Đang hướng dẫn Xe #{targetVehicleId} rời bãi từ ô <strong>{sessionParkedSpot}</strong> ra CỔNG EXIT.
                  </p>
                )}

                {sessionState === "NAVIGATING_TO_SPOT" && sessionTargetSpot && (
                  <p style={{ margin: "4px 0 0 0", fontSize: "14px", color: "#38bdf8" }}>
                    Tuyến đường chỉ dẫn đang hướng Xe #{targetVehicleId} tới ô <strong>{sessionTargetSpot}</strong>.
                  </p>
                )}

                {sessionState === "SELECTING_SPOT" && (
                  <p style={{ margin: "4px 0 0 0", fontSize: "14px", color: "#f59e0b" }}>
                    Bản đồ đang định vị Xe #{targetVehicleId}. Bạn có thể chọn ô đỗ mong muốn trên bản đồ.
                  </p>
                )}

                {sessionState === "PARKED" && (
                  <p style={{ margin: "4px 0 0 0", fontSize: "14px", color: "#4ade80" }}>
                    Xe #{targetVehicleId} đang được đỗ an toàn tại ô <strong>{sessionParkedSpot}</strong>.
                  </p>
                )}
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <button
                onClick={() => {
                  const newMute = !isMuted;
                  setIsMuted(newMute);
                  voiceManager.setMuted(newMute);
                }}
                style={{
                  background: isMuted ? "rgba(100, 116, 139, 0.2)" : "rgba(34, 197, 94, 0.2)",
                  border: `1px solid ${isMuted ? "#64748b" : "#22c55e"}`,
                  color: isMuted ? "#94a3b8" : "#4ade80",
                  padding: "8px 14px",
                  borderRadius: "8px",
                  fontWeight: 600,
                  cursor: "pointer",
                  fontSize: "13px",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px"
                }}
              >
                {isMuted ? "🔇 Tắt giọng nói" : "🔊 Giọng nói Bật"}
              </button>

              {/* Nút Hủy chọn ô đỗ khi đang dẫn đường vào */}
              {(sessionState === "NAVIGATING_TO_SPOT" || (sessionTargetSpot && sessionState !== "EXIT_NAVIGATION" && sessionState !== "PARKED")) && (
                <button
                  onClick={handleDismissRoute}
                  style={{
                    background: "rgba(239, 68, 68, 0.2)",
                    border: "1px solid #ef4444",
                    color: "#fca5a5",
                    padding: "8px 14px",
                    borderRadius: "8px",
                    fontWeight: 600,
                    cursor: "pointer",
                    fontSize: "13px"
                  }}
                >
                  ❌ Hủy chọn ô đỗ
                </button>
              )}

              {/* Nút Ẩn/Hiện đường chỉ dẫn lối ra khi đang ra bãi */}
              {sessionState === "EXIT_NAVIGATION" && (
                isRouteDismissed ? (
                  <button
                    onClick={() => setIsRouteDismissed(false)}
                    style={{
                      background: "rgba(56, 189, 248, 0.15)",
                      border: "1px solid #38bdf8",
                      color: "#38bdf8",
                      padding: "8px 14px",
                      borderRadius: "8px",
                      fontWeight: 600,
                      cursor: "pointer",
                      fontSize: "13px"
                    }}
                  >
                    🧭 Mở lại chỉ dẫn lối ra
                  </button>
                ) : (
                  <button
                    onClick={() => setIsRouteDismissed(true)}
                    style={{
                      background: "rgba(239, 68, 68, 0.2)",
                      border: "1px solid #ef4444",
                      color: "#fca5a5",
                      padding: "8px 14px",
                      borderRadius: "8px",
                      fontWeight: 600,
                      cursor: "pointer",
                      fontSize: "13px"
                    }}
                  >
                    ❌ Ẩn đường lối ra
                  </button>
                )
              )}

              {sessionState === "PARKED" && (
                <button
                  onClick={handleStartExit}
                  style={{
                    background: "#38bdf8",
                    color: "#0f172a",
                    border: "none",
                    padding: "10px 16px",
                    borderRadius: "8px",
                    fontWeight: "bold",
                    cursor: "pointer",
                    fontSize: "14px",
                    boxShadow: "0 4px 12px rgba(56, 189, 248, 0.3)"
                  }}
                >
                  🚗 Lấy xe ra
                </button>
              )}

              {sessionState !== "PARKED" && (
                <span style={{
                  background: getSessionStatusColor(),
                  color: "#fff",
                  padding: "6px 14px",
                  borderRadius: "20px",
                  fontSize: "12px",
                  fontWeight: 600,
                  whiteSpace: "nowrap",
                }}>
                  {getSessionStatusLabel()}
                </span>
              )}
            </div>
          </div>
        )}

        {/* ── Bảng Cảnh báo đi sai đường (Off-Route Warning) ── */}
        {sessionId && isOffRoute && (
          <div style={{
            background: "rgba(220, 38, 38, 0.95)",
            color: "#fff",
            padding: "12px 20px",
            borderRadius: "10px",
            marginBottom: "16px",
            display: "flex",
            alignItems: "center",
            gap: "12px",
            fontWeight: "bold",
            fontSize: "14px",
            boxShadow: "0 0 20px rgba(220, 38, 38, 0.5)",
            border: "1px solid #ef4444"
          }}>
            <span style={{ fontSize: "24px" }}>⚠️</span>
            <span>CẢNH BÁO: BẠN ĐANG ĐI SAI TUYẾN ĐƯỜNG CHỈ DẪN! VUI LÒNG QUAN SÁT SƠ ĐỒ BÃI ĐỖ.</span>
          </div>
        )}

        {/* ── Bảng Chỉ dẫn giọng nói realtime (Voice Instruction Status) ── */}
        {sessionId && !isOffRoute && navInstruction && !isRouteDismissed && (
          <div style={{
            background: "linear-gradient(135deg, #0284c7 0%, #0369a1 100%)",
            color: "#fff",
            padding: "10px 18px",
            borderRadius: "10px",
            marginBottom: "16px",
            display: "flex",
            alignItems: "center",
            gap: "12px",
            fontWeight: 600,
            fontSize: "14px",
            boxShadow: "0 4px 12px rgba(2, 132, 199, 0.3)"
          }}>
            <span style={{ fontSize: "20px" }}>🗣</span>
            <span>{navInstruction}</span>
          </div>
        )}

        <div className={`parking-workspace parking-workspace--${mode}`}>
          <div className="map-column">
            <ParkingMap
              spots={spots}
              cameras={cameras}
              filter={mode === "browse" ? browseFilter : "all"}
              recommendation={mode === "recommendation" ? recommendation : undefined}
              inspectedSpotId={inspectedSpotId}
              confirmedSpotId={confirmedSpotId ?? (sessionTargetSpot as SpotId | undefined)}
              activeNeed={activeNeed}
              route={route}
              routePaused={Boolean(warning)}
              activeVehicles={activeVehicles}
              frameSize={frameSize}
              onSpotClick={handleSpotClick}
            />
            <ParkingLegend />
          </div>

          {mode === "recommendation" && (
            <RecommendationPanel
              need={activeNeed}
              result={recommendation}
              onNeedChange={handleNeedChange}
              onChooseAlternative={chooseRecommendedSpot}
              onConfirm={handleConfirmSpot}
              onAbandon={() => enterBrowse("all")}
            />
          )}
          {mode === "browse" && inspectedSpot && (
            <SpotDetailSheet
              spot={inspectedSpot}
              onClose={() => inspectSpot(undefined)}
              onNavigate={() => confirmedSpotId !== inspectedSpot.id && handleConfirmSpot(inspectedSpot.id)}
            />
          )}
        </div>
      </main>

      {/* ── Sheet lựa chọn nhu cầu "Bạn muốn tìm chỗ đỗ theo cách nào?" ── */}
      {mode === "entry" && (
        <EntryChoiceSheet
          onRecommend={startRecommendation}
          onEmptyOnly={() => enterBrowse("empty")}
          onSkip={() => enterBrowse("all")}
        />
      )}
      {warning && (
        <InvalidSpotWarningSheet warning={warning} onSwitch={switchAlternative} onContinueMap={cancelNavigation} />
      )}
      {import.meta.env.DEV && (
        <MockControlPanel
          source={mockParkingDataSource}
          recommendedSpotId={recommendation?.best.spotId}
          selectedSpotId={confirmedSpotId}
        />
      )}

      {/* ── QR Kiosk chỉ hiển thị trên Trang Chung (không có sessionId) ── */}
      {!sessionId && <EntryQRKiosk />}
    </div>
  );
}
