import { FlaskConical, Play, Plus, X } from "lucide-react";
import { useState } from "react";
import type { SpotId } from "../domain/parking";
import {
  MOCK_SCENARIOS,
  type MockParkingDataSource,
  type MockScenarioId,
} from "../mocks/MockParkingDataSource";

import { useParkingStore } from "../stores/parkingStore";

interface MockControlPanelProps {
  source: MockParkingDataSource;
  recommendedSpotId?: SpotId;
  selectedSpotId?: SpotId;
}

export function MockControlPanel({ source, recommendedSpotId, selectedSpotId }: MockControlPanelProps) {
  const [open, setOpen] = useState(false);
  const [scenarioId, setScenarioId] = useState<MockScenarioId>("normal-independent");
  const [, refresh] = useState(0);
  const trackingSource = useParkingStore((state) => state.trackingSource);
  const setTrackingSource = useParkingStore((state) => state.setTrackingSource);

  const queue = (): void => {
    source.queueScenario(scenarioId, { recommendedSpotId, selectedSpotId });
    refresh((value) => value + 1);
  };

  const step = (): void => {
    source.stepScenario();
    refresh((value) => value + 1);
  };

  if (!open) {
    return (
      <button type="button" className="mock-toggle" onClick={() => setOpen(true)} aria-label="Mở điều khiển dữ liệu mô phỏng" title="Dữ liệu mô phỏng" data-testid="mock-toggle">
        <FlaskConical size={19} />
      </button>
    );
  }

  return (
    <aside className="mock-panel" aria-label="Điều khiển dữ liệu mô phỏng">
      <header>
        <span><FlaskConical size={18} />Kịch bản camera</span>
        <button type="button" onClick={() => setOpen(false)} aria-label="Đóng điều khiển mô phỏng" title="Đóng"><X size={18} /></button>
      </header>
      <select value={scenarioId} onChange={(event) => setScenarioId(event.target.value as MockScenarioId)} data-testid="mock-scenario">
        {MOCK_SCENARIOS.map((scenario) => <option key={scenario.id} value={scenario.id}>{scenario.label}</option>)}
      </select>

      <div style={{ margin: "12px 0", padding: "8px", background: "rgba(0,0,0,0.2)", borderRadius: "6px" }}>
        <strong style={{ fontSize: "12px", display: "block", marginBottom: "6px" }}>Nguồn Tracking:</strong>
        <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "13px", cursor: "pointer", marginBottom: "4px" }}>
          <input
            type="radio"
            name="trackingSource"
            checked={trackingSource === "opencv"}
            onChange={() => setTrackingSource("opencv")}
          />
          Camera / Backend OpenCV
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "13px", cursor: "pointer" }}>
          <input
            type="radio"
            name="trackingSource"
            checked={trackingSource === "sample"}
            onChange={() => setTrackingSource("sample")}
          />
          Dữ liệu mẫu (Simulator)
        </label>
      </div>

      <div className="mock-actions">
        <button type="button" onClick={queue} data-testid="mock-queue"><Plus size={16} />Xếp kịch bản</button>
        <button type="button" onClick={step} data-testid="mock-step"><Play size={16} />Chạy bước tiếp</button>
      </div>
      <dl>
        <div><dt>Đang chờ</dt><dd>{source.getPendingCount()}</dd></div>
        <div><dt>Bị từ chối</dt><dd>{source.getRejectedCount()}</dd></div>
      </dl>
      <p>{source.getLastDiagnostic()}</p>
    </aside>
  );
}
