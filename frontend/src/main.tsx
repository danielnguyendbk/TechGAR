import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app/App";
import "./styles/index.css";

const root = document.getElementById("root");
if (!root) throw new Error("Không tìm thấy phần tử #root");

// ── Định tuyến đơn giản dựa theo URL ──────────────────────────────────────
// ?session=1   →  Trang dẫn đường cá nhân cho xe #1 (QR)
// ?session=2   →  Trang dẫn đường cá nhân cho xe #2 (QR)
// (không có tham số) →  Dashboard quản lý bãi xe chính (có QR kiosk)
const params    = new URLSearchParams(window.location.search);
const sessionId = params.get("session");

createRoot(root).render(
  <StrictMode>
    <App sessionId={sessionId} />
  </StrictMode>,
);
