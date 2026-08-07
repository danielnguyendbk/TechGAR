import { useState, useEffect } from "react";

interface KioskSession {
  sessionId: string;
  trackId: number;
  qrUrl: string;
  createdAt: string;
}

/**
 * EntryQRKiosk – Hiển thị QR cho xe đang ở cổng (WAITING_FOR_SCAN).
 * 
 * Quy tắc:
 * - Hiển thị QR cho xe mới nhất đang chờ ở cổng (state === "WAITING_FOR_SCAN" & !claimed)
 * - Giữ mã QR hiển thị chừng nào xe còn ở trạng thái WAITING_FOR_SCAN tại cổng
 * - Khi xe được quét (claimed = true / state sang SELECTING/NAVIGATING) hoặc hết xe ở cổng -> Tự ẩn QR
 * - Tự chuyển sang xe tiếp theo khi có xe mới vào cổng
 * - Chỉ hiển thị trên trang chung (không sessionId)
 */
export function EntryQRKiosk() {
  const [activeKiosk, setActiveKiosk] = useState<KioskSession | null>(null);
  const [minimized, setMinimized]     = useState<boolean>(false);

  useEffect(() => {
    let active = true;

    const checkGateSessions = async () => {
      try {
        const res = await fetch(`/navigation_sessions.json?t=${Date.now()}`);
        if (!res.ok || !active) return;
        const sessions = await res.json() as Record<string, {
          sessionId: string;
          vehicleTrackId: number | null;
          activeTrackId: number | null;
          state: string;
          claimed: boolean;
        }>;

        // Tìm tất cả các session đang chờ ở cổng (WAITING_FOR_SCAN và chưa claimed)
        const waitingSessions = Object.values(sessions).filter(
          (s) => s.state === "WAITING_FOR_SCAN"
                 && s.vehicleTrackId !== undefined
                 && s.vehicleTrackId !== null
                 && !s.claimed
        );

        if (waitingSessions.length > 0) {
          // Lấy xe mới nhất ở cổng (track_id lớn nhất hoặc xếp cuối)
          const latestGateSession = waitingSessions[waitingSessions.length - 1];
          const trackId = latestGateSession.vehicleTrackId!;
          const sid = latestGateSession.sessionId;

          // Cập nhật kiosk nếu chưa hiển thị xe này
          if (!activeKiosk || activeKiosk.sessionId !== sid) {
            const fullUrl = `${window.location.origin}/?session=${sid}`;
            const qrImageUrl = `https://api.qrserver.com/v1/create-qr-code/?size=160x160&data=${encodeURIComponent(fullUrl)}`;

            setActiveKiosk({
              sessionId: sid,
              trackId: trackId,
              qrUrl: qrImageUrl,
              createdAt: new Date().toLocaleTimeString("vi-VN"),
            });
            setMinimized(false);
          }
        } else {
          // Không còn xe nào ở cổng -> Ẩn Kiosk
          if (activeKiosk) {
            setActiveKiosk(null);
          }
        }
      } catch (_) {}
    };

    void checkGateSessions();
    const interval = setInterval(checkGateSessions, 500);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [activeKiosk]);

  if (!activeKiosk) return null;

  const targetNavUrl = `/?session=${activeKiosk.sessionId}`;

  return (
    <div style={{
      position: "fixed",
      bottom: "24px",
      right: "24px",
      zIndex: 9999,
      background: "rgba(15, 23, 42, 0.95)",
      backdropFilter: "blur(12px)",
      border: "1px solid #38bdf8",
      borderRadius: "16px",
      boxShadow: "0 20px 40px rgba(0,0,0,0.5), 0 0 20px rgba(56, 189, 248, 0.2)",
      width: minimized ? "260px" : "320px",
      transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
      color: "#fff",
      overflow: "hidden",
      fontFamily: "system-ui, -apple-system, sans-serif"
    }}>
      {/* ── Header Kiosk ── */}
      <div style={{
        background: "linear-gradient(135deg, #0284c7 0%, #0369a1 100%)",
        padding: "10px 16px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "18px" }}>🚥</span>
          <strong style={{ fontSize: "14px", letterSpacing: "0.5px" }}>CỔNG VÀO · QUÉT MÃ QR</strong>
        </div>
        <button
          onClick={() => setMinimized(!minimized)}
          style={{
            background: "transparent",
            border: "none",
            color: "#fff",
            cursor: "pointer",
            fontSize: "14px",
            opacity: 0.8
          }}
        >
          {minimized ? "▲" : "▼"}
        </button>
      </div>

      {!minimized && (
        <div style={{ padding: "16px", textAlign: "center" }}>
          <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "8px" }}>
            Phát hiện <strong>Xe #{activeKiosk.trackId}</strong> vào bãi đỗ ({activeKiosk.createdAt})
          </div>

          {/* Thông báo hướng dẫn */}
          <div style={{
            background: "rgba(245, 158, 11, 0.15)",
            border: "1px solid #f59e0b",
            borderRadius: "8px",
            padding: "8px 12px",
            marginBottom: "12px",
            fontSize: "13px",
            color: "#fbbf24"
          }}>
            📋 Quét mã để bắt đầu sử dụng hệ thống
          </div>

          {/* Khung Mã QR */}
          <div style={{
            background: "#fff",
            padding: "12px",
            borderRadius: "12px",
            display: "inline-block",
            marginBottom: "12px",
            boxShadow: "0 4px 10px rgba(0,0,0,0.3)"
          }}>
            <img
              src={activeKiosk.qrUrl}
              alt="Mã QR Dẫn đường"
              style={{ width: "140px", height: "140px", display: "block" }}
            />
          </div>

          {/* URL Session */}
          <div style={{
            fontSize: "11px",
            color: "#64748b",
            marginBottom: "12px",
            fontFamily: "monospace",
            wordBreak: "break-all",
          }}>
            {window.location.origin}{targetNavUrl}
          </div>

          {/* Nút Mở Trang Cá Nhân */}
          <a
            href={targetNavUrl}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: "block",
              width: "100%",
              boxSizing: "border-box",
              background: "#38bdf8",
              color: "#0f172a",
              fontWeight: 700,
              padding: "10px 0",
              borderRadius: "8px",
              textDecoration: "none",
              fontSize: "13px",
              boxShadow: "0 4px 12px rgba(56, 189, 248, 0.3)",
              transition: "transform 0.1s ease"
            }}
          >
            📱 Mở giao diện riêng (Xe #{activeKiosk.trackId})
          </a>
        </div>
      )}
    </div>
  );
}
