import { useState, useEffect } from 'react';
import { ParkingMap } from './components/parking-map/ParkingMap';
import type { ActiveVehicle, ParkingSpotState, ParkingStatus } from './types/parking';
import { parkingLayout } from './data/parkingLayout';
import './App.css';

const pendingApiSpots: ParkingSpotState[] = parkingLayout.map((slot) => ({
  id: slot.id,
  status: 'unavailable',
}));

// Initialize mock data from the parking layout
const mockSpots: ParkingSpotState[] = parkingLayout.map((slot, index) => {
  // Distribute states logically:
  // - First few spots in left columns: occupied or available
  // - Some incoming spots
  let status: ParkingStatus = 'available';
  if (index % 7 === 0) {
    status = 'occupied';
  } else if (index % 11 === 3) {
    status = 'incoming';
  } else if (index % 4 === 1) {
    status = 'occupied';
  }

  return {
    id: slot.id,
    status,
  };
});

const zoneNames: Record<string, string> = {
  LEFT_OUTER: 'Cột trái ngoài (L01 - L11, L23)',
  LEFT_INNER: 'Cột trái trong (L12 - L22, L24)',
  MIDDLE_LEFT: 'Cột giữa trái (M01 - M08, M10 - M11, M23)',
  MIDDLE_RIGHT: 'Cột giữa phải (M12 - M19, M21 - M22, M24)',
  RIGHT_LEFT: 'Cột phải trái (R01 - R11, R23)',
  RIGHT_RIGHT: 'Cột phải phải (R13 - R22, R24)',
};

function App() {
  const [spots, setSpots] = useState<ParkingSpotState[]>(pendingApiSpots);
  const [selectedSpotId, setSelectedSpotId] = useState<string | null>(null);
  const [dataSource, setDataSource] = useState<'api' | 'mock'>('api');
  const [apiStatus, setApiStatus] = useState<'connected' | 'disconnected' | 'loading'>('loading');
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [activeVehicles, setActiveVehicles] = useState<ActiveVehicle[]>([]);

  // Poll API for real-time status when in API mode
  useEffect(() => {
    if (dataSource !== 'api') {
      setApiStatus('connected'); // Mock always "connected" in context of UI
      setLastUpdated(null);
      setActiveVehicles([]);
      setSpots(mockSpots);
      return;
    }

    let isMounted = true;
    setApiStatus('loading');
    setLastUpdated(null);
    setSpots(pendingApiSpots);
    const fetchStatus = async () => {
      try {
        const res = await fetch('http://localhost:5050/api/parking-status');
        if (!res.ok) throw new Error('API server returned error code');
        const data = await res.json();
        
        if (isMounted) {
          setApiStatus('connected');
          setLastUpdated(new Date().toLocaleTimeString('vi-VN'));
          
          if (data && data.slots) {
            setSpots((prevSpots) =>
              prevSpots.map((spot) => {
                const backendSlot = data.slots[spot.id];
                if (backendSlot) {
                  let status: ParkingStatus = 'available';
                  if (backendSlot.status === 'occupied') {
                    status = 'occupied';
                  } else if (backendSlot.status === 'incoming') {
                    status = 'incoming';
                  }
                  return {
                    id: spot.id,
                    status,
                  };
                }
                return spot;
              })
            );
          }
        }
      } catch (err) {
        if (isMounted) {
          setApiStatus('disconnected');
          setActiveVehicles([]);
          setSpots(pendingApiSpots);
          console.error('Error fetching parking status:', err);
        }
      }
    };

    const fetchVehicles = async () => {
      try {
        const res = await fetch('http://localhost:5050/api/frontend-vehicle-positions');
        if (!res.ok) throw new Error('Vehicle positions API returned error code');
        const data = await res.json();
        if (!isMounted) return;
        const vehicles = Object.entries(data?.active_vehicles ?? {})
          .map(([id, value]) => {
            const vehicle = value as {
              track_id?: number;
              status?: string;
              position?: { x?: number; y?: number };
            };
            const x = Number(vehicle.position?.x);
            const y = Number(vehicle.position?.y);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
            return {
              id,
              trackId: Number(vehicle.track_id ?? id),
              x,
              y,
              status: vehicle.status ?? 'confirmed',
            };
          })
          .filter((vehicle): vehicle is ActiveVehicle => vehicle !== null);
        setActiveVehicles(vehicles);
      } catch (err) {
        if (isMounted) {
          setActiveVehicles([]);
          console.error('Error fetching vehicle positions:', err);
        }
      }
    };

    fetchStatus();
    fetchVehicles();
    const interval = setInterval(fetchStatus, 1000);
    const vehicleInterval = setInterval(fetchVehicles, 500);

    return () => {
      isMounted = false;
      clearInterval(interval);
      clearInterval(vehicleInterval);
    };
  }, [dataSource]);


  // Statistics calculation
  const total = spots.length;
  const available = spots.filter((s) => s.status === 'available').length;
  const occupied = spots.filter((s) => s.status === 'occupied').length;
  const incoming = spots.filter((s) => s.status === 'incoming').length;
  const occupancyRate = total > 0 ? Math.round((occupied / total) * 100) : 0;

  // Selected spot info
  const selectedSpot = spots.find((s) => s.id === selectedSpotId);
  const selectedLayout = parkingLayout.find((s) => s.id === selectedSpotId);

  // Function to change status of selected spot
  const handleStatusChange = (newStatus: ParkingStatus) => {
    if (!selectedSpotId) return;
    setSpots((prevSpots) =>
      prevSpots.map((spot) =>
        spot.id === selectedSpotId ? { ...spot, status: newStatus } : spot
      )
    );
  };

  // Reset all spots to random states for demo purposes
  const handleRandomize = () => {
    const statuses: ParkingStatus[] = ['available', 'occupied', 'incoming'];
    setSpots((prevSpots) =>
      prevSpots.map((spot) => ({
        ...spot,
        status: statuses[Math.floor(Math.random() * statuses.length)],
      }))
    );
  };

  // Set all spots to empty/available
  const handleClearAll = () => {
    setSpots((prevSpots) =>
      prevSpots.map((spot) => ({
        ...spot,
        status: 'available',
      }))
    );
  };

  return (
    <div className="app-dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-logo">
          <div className="logo-icon"></div>
          <div>
            <h1>SmartParking AI</h1>
            <p>Hệ thống Giám sát & Quản lý Bãi đỗ xe thông minh</p>
          </div>
        </div>
        
        {/* Toggle Mode and Status Indicators */}
        <div className="header-controls">
          <div className="datasource-toggle">
            <button 
              className={`toggle-btn ${dataSource === 'api' ? 'active' : ''}`}
              onClick={() => setDataSource('api')}
            >
              📡 Dữ liệu Real-time (API)
            </button>
            <button 
              className={`toggle-btn ${dataSource === 'mock' ? 'active' : ''}`}
              onClick={() => setDataSource('mock')}
            >
              🧪 Chế độ Mô phỏng (Mock)
            </button>
          </div>
          
          <div className="connection-status">
            {dataSource === 'api' ? (
              apiStatus === 'connected' ? (
                <span className="status-badge-conn success">
                  <span className="indicator-dot blinking"></span> Live API: Connected {lastUpdated && `(${lastUpdated})`}
                </span>
              ) : apiStatus === 'loading' ? (
                <span className="status-badge-conn loading">
                  <span className="indicator-dot animate-pulse"></span> Connecting to API...
                </span>
              ) : (
                <span className="status-badge-conn danger">
                  <span className="indicator-dot"></span> API Offline (Port 5050)
                </span>
              )
            ) : (
              <span className="status-badge-conn mock-info">
                ⚠️ Chế độ Offline
              </span>
            )}
          </div>
        </div>

        <div className="header-actions">
          {dataSource === 'mock' && (
            <>
              <button className="btn-secondary" onClick={handleClearAll}>
                Giải phóng bãi xe
              </button>
              <button className="btn-primary" onClick={handleRandomize}>
                Mô phỏng dữ liệu ngẫu nhiên
              </button>
            </>
          )}
        </div>
      </header>

      {/* Stats Summary Row */}
      <section className="stats-container">
        <div className="stat-card">
          <span className="stat-label">Tổng vị trí đỗ</span>
          <span className="stat-value">{total}</span>
          <div className="stat-bar" style={{ background: '#475569', width: '100%' }}></div>
        </div>
        <div className="stat-card">
          <span className="stat-label">Còn trống (Xanh)</span>
          <span className="stat-value text-available">{available}</span>
          <div
            className="stat-bar bg-available"
            style={{ width: `${(available / total) * 100}%` }}
          ></div>
        </div>
        <div className="stat-card">
          <span className="stat-label">Đang đỗ (Đỏ)</span>
          <span className="stat-value text-occupied">{occupied}</span>
          <div
            className="stat-bar bg-occupied"
            style={{ width: `${(occupied / total) * 100}%` }}
          ></div>
        </div>
        <div className="stat-card">
          <span className="stat-label">Sắp đỗ (Vàng)</span>
          <span className="stat-value text-incoming">{incoming}</span>
          <div
            className="stat-bar bg-incoming"
            style={{ width: `${(incoming / total) * 100}%` }}
          ></div>
        </div>
        <div className="stat-card">
          <span className="stat-label">Tỉ lệ lấp đầy</span>
          <span className="stat-value">{occupancyRate}%</span>
          <div className="stat-bar bg-blue" style={{ width: `${occupancyRate}%` }}></div>
        </div>
      </section>

      {/* Main Grid Content */}
      <main className="dashboard-grid">
        {/* Left/Middle Column: Interactive Map & Live AI Camera Feed */}
        <div className="visuals-container">
          <section className="map-section-card">
            <div className="card-header">
              <h2>Sơ đồ Bãi xe Bản đồ Vector</h2>
              <div className="status-badge live">LIVE MAP</div>
            </div>
            <div className="card-body">
              <ParkingMap
                spots={spots}
                activeVehicles={activeVehicles}
                selectedSpotId={selectedSpotId}
                onSelectSpot={setSelectedSpotId}
              />
            </div>
          </section>

          <section className="map-section-card video-card">
            <div className="card-header">
              <h2>Camera Phân tích AI (Thời gian thực)</h2>
              <div className="status-badge camera-badge">LIVE CAMERA</div>
            </div>
            <div className="card-body video-body">
              {dataSource === 'api' ? (
                <div className="live-video-frame">
                  <img
                  src="http://localhost:5050/api/video-feed"
                  alt="AI Annotated Camera Stream"
                  className="live-video-stream"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                    const parent = e.currentTarget.parentElement;
                    if (parent) {
                      const existingErr = parent.querySelector('.video-error-placeholder');
                      if (!existingErr) {
                        const errorMsg = document.createElement('div');
                        errorMsg.className = 'video-error-placeholder';
                        errorMsg.innerText = '⚠️ Không thể kết nối tới luồng Camera AI (Port 5050)';
                        parent.appendChild(errorMsg);
                      }
                    }
                  }}
                  />
                  <div className="vehicle-marker-layer" aria-hidden="true">
                    {activeVehicles.map((vehicle) => (
                      <div
                        key={vehicle.id}
                        className="vehicle-map-marker"
                        style={{
                          left: `${Math.max(0, Math.min(100, (vehicle.x / 640) * 100))}%`,
                          top: `${Math.max(0, Math.min(100, (vehicle.y / 640) * 100))}%`,
                        }}
                      >
                        <span className="vehicle-map-marker-label">{vehicle.trackId}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="video-mock-placeholder">
                  <div className="mock-camera-lens">📷</div>
                  <h3>Chế độ Offline (Mock Mode)</h3>
                  <p>Hãy chuyển sang chế độ 📡 Dữ liệu Real-time (API) để hiển thị live stream từ Camera.</p>
                </div>
              )}
            </div>
          </section>
        </div>

        {/* Right Side: Spot Controls & Detail info */}
        <section className="control-section-card">
          <div className="card-header">
            <h2>Bảng điều khiển vị trí</h2>
          </div>
          <div className="card-body">
            {selectedSpot && selectedLayout ? (
              <div className="spot-detail-active">
                <div className="spot-id-badge">
                  <span>VỊ TRÍ CHI TIẾT</span>
                  <h3>{selectedSpot.id}</h3>
                </div>

                <div className="spot-info-list">
                  <div className="spot-info-row">
                    <span className="info-label">Khu vực:</span>
                    <span className="info-val">{zoneNames[selectedLayout.zone] || selectedLayout.zone}</span>
                  </div>
                  <div className="spot-info-row">
                    <span className="info-label">Trạng thái hiện tại:</span>
                    <span className={`info-val status-badge-state status-${selectedSpot.status}`}>
                      {selectedSpot.status === 'available' && 'Trống (Xanh)'}
                      {selectedSpot.status === 'occupied' && 'Có xe (Đỏ)'}
                      {selectedSpot.status === 'incoming' && 'Sắp có xe (Vàng)'}
                      {selectedSpot.status === 'unavailable' && 'Đang chờ dữ liệu'}
                    </span>
                  </div>
                </div>

                <div className="spot-action-area">
                  <h4>Cập nhật trạng thái thủ công</h4>
                  <p>Mô phỏng thay đổi dữ liệu nhận từ API hoặc Camera AI</p>

                  <div className="action-buttons-grid">
                    <button
                      className="btn-action btn-to-available"
                      onClick={() => handleStatusChange('available')}
                      disabled={selectedSpot.status === 'available'}
                    >
                      <span className="dot bg-available"></span> Đánh dấu Trống
                    </button>
                    <button
                      className="btn-action btn-to-occupied"
                      onClick={() => handleStatusChange('occupied')}
                      disabled={selectedSpot.status === 'occupied'}
                    >
                      <span className="dot bg-occupied"></span> Đánh dấu Có xe
                    </button>
                    <button
                      className="btn-action btn-to-incoming"
                      onClick={() => handleStatusChange('incoming')}
                      disabled={selectedSpot.status === 'incoming'}
                    >
                      <span className="dot bg-incoming"></span> Đánh dấu Sắp đỗ
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="spot-detail-empty">
                <div className="empty-state-icon">ℹ</div>
                <h3>Chưa chọn vị trí</h3>
                <p>Hãy click chọn một ô đỗ xe trên sơ đồ bản đồ để xem chi tiết và mô phỏng thay đổi trạng thái.</p>
              </div>
            )}

            <div className="instructions-card">
              <h4>Hướng dẫn tương tác</h4>
              <ul>
                <li>Di chuột qua ô đỗ xe để xem nhanh thông tin vị trí & trạng thái qua tooltip.</li>
                <li>Click vào một ô đỗ để chọn vị trí đó (ô được chọn sẽ phát sáng viền xanh dương).</li>
                <li>Dùng các nút phía trên để cập nhật nhanh trạng thái của ô đỗ xe đã chọn.</li>
                <li>Kéo thanh cuộn nếu màn hình hiển thị quá nhỏ để xem toàn vẹn bản đồ.</li>
              </ul>
            </div>
          </div>
        </section>
      </main>

      <footer className="dashboard-footer">
        <p>&copy; 2026 - Dự án Nghiên cứu Khoa học Bãi đỗ xe Thông minh AI. Hoàn thiện bởi Antigravity.</p>
      </footer>
    </div>
  );
}

export default App;
