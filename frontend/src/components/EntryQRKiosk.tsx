import { useState, useEffect } from "react";
import * as api from "../api/backendApi";

interface KioskSession {
  sessionId: string;
  globalVehicleId: number;
  qrUrl: string;
  createdAt: string;
}

/**
 * EntryQRKiosk — QR display for vehicles at the gate.
 *
 * Review fixes:
 *   #24 — Uses sorted waiting sessions by createdAt (not last object key)
 *   #32 — Reads from backend API, not JSON files
 *   #8  — sessionId is a random token, not track_id
 */
export function EntryQRKiosk() {
  const [activeKiosk, setActiveKiosk] = useState<KioskSession | null>(null);
  const [minimized, setMinimized] = useState(false);

  useEffect(() => {
    let active = true;

    const checkGateSessions = async () => {
      try {
        const sessions = await api.getWaitingSessions();
        if (!active) return;

        if (sessions && sessions.length > 0) {
          // Fix #24: Backend returns sorted by createdAt, take the latest
          const latest = sessions[sessions.length - 1];
          const gvid = latest.globalVehicleId;
          const sid = latest.sessionId;

          if (!activeKiosk || activeKiosk.sessionId !== sid) {
            const fullUrl = `${window.location.origin}/?session=${sid}`;
            const qrImageUrl = `https://api.qrserver.com/v1/create-qr-code/?size=160x160&data=${encodeURIComponent(fullUrl)}`;

            setActiveKiosk({
              sessionId: sid,
              globalVehicleId: gvid,
              qrUrl: qrImageUrl,
              createdAt: new Date().toLocaleTimeString("vi-VN"),
            });
            setMinimized(false);
          }
        } else {
          if (activeKiosk) setActiveKiosk(null);
        }
      } catch (_) {}
    };

    void checkGateSessions();
    const interval = setInterval(checkGateSessions, 1000);

    return () => { active = false; clearInterval(interval); };
  }, [activeKiosk]);

  if (!activeKiosk) return null;

  const targetNavUrl = `/?session=${activeKiosk.sessionId}`;

  return (
    <div style={{
      position: "fixed", bottom: "24px", right: "24px", zIndex: 9999,
      background: "rgba(15, 23, 42, 0.95)", backdropFilter: "blur(12px)",
      border: "1px solid #38bdf8", borderRadius: "16px",
      boxShadow: "0 20px 40px rgba(0,0,0,0.5), 0 0 20px rgba(56, 189, 248, 0.2)",
      width: minimized ? "260px" : "320px",
      transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
      color: "#fff", overflow: "hidden", fontFamily: "system-ui, -apple-system, sans-serif",
    }}>
      <div style={{
        background: "linear-gradient(135deg, #0284c7 0%, #0369a1 100%)",
        padding: "10px 16px", display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "18px" }}>🚥</span>
          <strong style={{ fontSize: "14px", letterSpacing: "0.5px" }}>CỔNG VÀO · QUÉT MÃ QR</strong>
        </div>
        <button onClick={() => setMinimized(!minimized)} style={{
          background: "transparent", border: "none", color: "#fff",
          cursor: "pointer", fontSize: "14px", opacity: 0.8,
        }}>{minimized ? "▲" : "▼"}</button>
      </div>

      {!minimized && (
        <div style={{ padding: "16px", textAlign: "center" }}>
          <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "8px" }}>
            Phát hiện <strong>Xe #{activeKiosk.globalVehicleId}</strong> vào bãi ({activeKiosk.createdAt})
          </div>

          <div style={{
            background: "rgba(245, 158, 11, 0.15)", border: "1px solid #f59e0b",
            borderRadius: "8px", padding: "8px 12px", marginBottom: "12px",
            fontSize: "13px", color: "#fbbf24",
          }}>📋 Quét mã để bắt đầu sử dụng hệ thống</div>

          <div style={{
            background: "#fff", padding: "12px", borderRadius: "12px",
            display: "inline-block", marginBottom: "12px",
            boxShadow: "0 4px 10px rgba(0,0,0,0.3)",
          }}>
            <img src={activeKiosk.qrUrl} alt="Mã QR" style={{ width: "140px", height: "140px", display: "block" }} />
          </div>

          <div style={{
            fontSize: "11px", color: "#64748b", marginBottom: "8px",
            fontFamily: "monospace", wordBreak: "break-all",
          }}>
            Mã phiên: <strong>{activeKiosk.sessionId}</strong>
          </div>

          <a href={targetNavUrl} target="_blank" rel="noopener noreferrer" style={{
            display: "block", width: "100%", boxSizing: "border-box",
            background: "#38bdf8", color: "#0f172a", fontWeight: 700,
            padding: "10px 0", borderRadius: "8px", textDecoration: "none", fontSize: "13px",
          }}>📱 Mở giao diện (Xe #{activeKiosk.globalVehicleId})</a>
        </div>
      )}
    </div>
  );
}
