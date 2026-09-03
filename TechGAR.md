**\# Kế hoạch hoàn thiện TechGAR \`ver\_new\`**

**\#\# 1\. Kết luận kiểm tra hiện trạng**

**Bản mới đã có phần lớn thuật toán nền, nhưng chưa tạo thành một hệ thống quản lý danh tính khép kín:**

**\- Local tracking đã có Kalman theo thời gian, LAPJV/Hungarian, LOST recovery, template matching và bảo vệ khi hai xe nhập blob.**  
**\- Global association đã kết hợp vị trí Mahalanobis, hướng, hình học, thời gian, topology và appearance.**  
**\- ReID hiện tại chỉ là đặc trưng màu theo lưới 3×3 trong \[appearance.py\](\</D:/Documents/SCIENTIFIC RESEARCH/An4/TechGAR-ver\_new/techgar/appearance.py:20\>), chưa phải vehicle ReID học sâu.**  
**\- \[registry.py\](\</D:/Documents/SCIENTIFIC RESEARCH/An4/TechGAR-ver\_new/techgar/registry.py:252\>) có recovery-before-mint nhưng vẫn có thể tạo GID ở bất kỳ vị trí nào.**  
**\- \[api.py\](\</D:/Documents/SCIENTIFIC RESEARCH/An4/TechGAR-ver\_new/techgar/api.py:68\>) và \[sessions.py\](\</D:/Documents/SCIENTIFIC RESEARCH/An4/TechGAR-ver\_new/techgar/sessions.py:54\>) giữ hai bộ session khác nhau, có thể lệch trạng thái.**  
**\- Client được phép gửi \`global\_vehicle\_id\` khi claim; như vậy frontend có thể tự gán GID.**  
**\- Endpoint \`parked\` tự lấy \`targetSpotId\` làm \`parkedSpotId\`, không chờ Slot Engine xác nhận.**  
**\- Reset đưa bộ đếm GID về 1; registry, gallery, alias, session, reservation và slot ownership đều chưa được lưu bền vững.**  
**\- Frontend tự chuyển sang dữ liệu demo khi Session API lỗi, có nguy cơ hiển thị xe và ô đỗ giả trong chế độ live.**  
**\- API chạy độc lập không khởi động pipeline thật; MJPEG hiện chỉ phát ảnh heartbeat 1 pixel.**  
**\- Backend hiện thu thập 102 test nhưng có 5 test fail và 1 error do manifest yêu cầu video MP4 không tồn tại trong clone. Frontend chưa thể kiểm chứng lại vì \`node\_modules\` chưa có.**  
**\- Calibration hiện chỉ có 4 điểm/camera, trong khi chính dự án yêu cầu tối thiểu 6 điểm; chưa đủ điều kiện nghiệm thu thực tế.**  
**\- Dữ liệu M04 có 976 cặp frame và video gốc trong dự án cũ, nhưng identity GT không đủ dày để kết luận IDF1/IDSW toàn cảnh.**

**Bản cũ chỉ được dùng để rút ra invariant, tình huống lỗi và cách đo; tuyệt đối không sao chép source, threshold hoặc state machine. Cách triển khai bám theo \`karpathy-guidelines\`: sửa từ nguồn sự thật và test invariant trước, không rewrite toàn bộ hay thêm model để che lỗi lifecycle.**

**\#\# 2\. Kiến trúc danh tính mục tiêu**

**Luồng duy nhất:**

**\`Detection → Local Track → World Projection/Fusion → Candidate Association → Global Registry → Slot/Session → API\`**

**Các invariant bắt buộc:**

**\- Chỉ \`GlobalIdentityRegistry\` được cấp, alias hoặc retire GID.**  
**\- GID là số tăng đơn điệu theo \`site\_id\`, lưu trong SQLite và không bao giờ tái sử dụng.**  
**\- Local ID chỉ có ý nghĩa trong một camera/runtime; Local ID không được gửi cho người dùng.**  
**\- Track mới chỉ được cấp GID sau khi:**

  **1\. Trưởng thành đủ số frame/thời gian.**  
  **2\. Có chuyển động thật, không phải nhiễu tĩnh hoặc blob cắt biên.**  
  **3\. Cắt \`entry\_gate\` đúng hướng.**  
  **4\. Không khớp với bất kỳ active/lost/recovery identity hợp lệ nào.**  
  **5\. Không nằm trong occlusion group hoặc trường hợp association còn mơ hồ.**

**\- Detection xuất hiện giữa bãi chỉ mang trạng thái \`UNKNOWN\`, không được mint GID.**  
**\- Khi hai ứng viên gần điểm số nhau, kết quả phải là \`DEFER\`, không đoán.**  
**\- Một GID không được sở hữu hai local track đang active trong cùng camera.**  
**\- Blob merged không cập nhật gallery và không cấp ID.**  
**\- Một slot chỉ có tối đa một owner và một reservation.**  
**\- Xe \`PARKED\` được neo bằng slot ownership; mất detection không làm mất GID.**  
**\- Session chỉ đóng khi xe cắt exit gate đúng hướng; người dùng bấm “ra khỏi bãi” chỉ chuyển sang dẫn đường ra.**  
**\- Alias GID phải remap đồng thời local owner, session, reservation, slot ownership và gallery trong một transaction.**  
**\- Hai GID trưởng thành không tự động merge. Hệ thống tạo \`alias\_proposal\`; chỉ merge khi có chuỗi bằng chứng mạnh hoặc operator xác nhận.**

**\#\# 3\. Các giai đoạn triển khai**

**\#\#\# Giai đoạn A — Dữ liệu và commissioning**

**\- Tạo data-root ngoài Git, cấu hình bằng \`TECHGAR\_DATA\_ROOT\`; manifest chỉ chứa đường dẫn tương đối.**  
**\- Viết công cụ import chỉ sao chép dữ liệu từ dự án cũ, không sao chép code.**  
**\- Import M04 từ \`TechGAR\_Parking/backend/data/droidcam\_shared\_m\_04/...\`, lưu provenance, kích thước và SHA-256.**  
**\- Đánh dấu \`vd\_17\` và \`vd\_18\` là \`unavailable\` cho đến khi có video thật; unit test không phụ thuộc dữ liệu ngoài, còn lệnh nghiệm thu \`--require-real-data\` phải fail rõ ràng nếu thiếu.**  
**\- Làm lại calibration cho hai camera bằng tối thiểu 6 điểm độc lập, phân bố đều; chuẩn hóa world unit thành mét và đo residual trên điểm hold-out.**  
**\- Xác định riêng \`entry\_gate\`, \`exit\_gate\`, entry/exit corridor, blind gap, hướng hợp lệ và 48 polygon slot.**  
**\- Commissioning bắt buộc bắt đầu khi bãi trống. Nếu thấy xe có sẵn, slot được ghi \`occupied-unknown\`; hệ thống không tự cấp GID.**  
**\- Thu thêm ba nhóm recording độc lập:**

  **\- Calibration/tuning.**  
  **\- Validation ngưỡng.**  
  **\- Held-out acceptance chưa dùng chỉnh tham số.**

**\- Ground truth phải chứa bbox/ground anchor/physical vehicle ID trên từng frame đồng bộ, entry–handoff–park–unpark–exit và slot occupancy. Thiếu GT thì metric trả \`N/A\` hoặc \`insufficient\_gt\`.**

**\#\#\# Giai đoạn B — Sửa Local Tracking và cấp GID**

**\- Giữ Kalman, timestamp-based prediction, LAPJV và template recovery của bản mới.**  
**\- Association phải xét cả ACTIVE và LOST trước khi tạo local track mới.**  
**\- Thêm log cho từng quyết định: bbox, prediction, Mahalanobis, IoU, kích thước, hướng, classic appearance, neural appearance, total cost, margin và lý do reject.**  
**\- Candidate trước cổng dùng \`candidate\_id\`, chưa chiếm GID.**  
**\- Thêm bộ phát hiện crossing theo đoạn chuyển động trước–sau gate, kiểm tra hướng và chống lặp bằng event key.**  
**\- Khi crossing hợp lệ, allocator SQLite cấp GID trong transaction và tạo identity \`PROVISIONAL\`; chỉ publish sau maturity gate cuối.**  
**\- Soft reset không xóa bộ đếm. Moving identity được phục hồi dưới \`RECOVERY\_PENDING\` với covariance tăng; parked identity giữ nguyên slot.**  
**\- Recovery window sau restart lấy theo topology: \`max(10 giây, max\_handoff\_dt \+ 5 giây)\`. Hết hạn thì retire có lý do, nhưng GID cũ không bao giờ được dùng lại.**

**\#\#\# Giai đoạn C — ReID hybrid cho Windows CPU**

**\- Tạo giao diện \`AppearanceEncoder\` chung:**

  **\- \`ClassicColorGridEncoder\` làm fallback.**  
  **\- \`VehicleReIDEncoder\` dùng model học sâu.**  
  **\- Mỗi descriptor kèm \`model\_id\`, version, dimension, camera và quality.**

**\- Model ban đầu là \`vehicle-reid-0001\`, ONNX khoảng 8.8 MB, input \`1×3×208×208\`, output 512 chiều; chạy bằng ONNX Runtime CPU. Model này được thiết kế cho vehicle ReID và dùng cosine distance. Artifact phải được tải bằng script với URL, license và SHA-256 cố định từ \[model manifest chính thức\](https://github.com/openvinotoolkit/open\_model\_zoo/blob/master/models/public/vehicle-reid-0001/model.yml); ONNX Runtime có CPU execution provider chính thức trên Windows. (\[tài liệu ONNX Runtime\](https://onnxruntime.ai/docs/get-started/with-windows.html))**  
**\- Chỉ trích embedding từ crop đủ chất lượng: không merged, không cắt biên, không blur nặng, đủ kích thước và không che khuất quá mức.**  
**\- Cache embedding theo local track; mặc định inference tối đa 2 lần/giây/track, ưu tiên entry, trước blind gap và sau handoff.**  
**\- Gallery giữ tối đa 12 mẫu tin cậy, cân bằng theo camera và thời điểm; không lưu ảnh crop mặc định.**  
**\- Association theo thứ tự:**

  **1\. Topology và thời gian loại ứng viên bất khả thi.**  
  **2\. Kalman/Mahalanobis và hướng tạo candidate set.**  
  **3\. Hình học và kích thước kiểm tra consistency.**  
  **4\. Classic và neural ReID xếp hạng các candidate còn lại.**  
  **5\. Margin không đủ thì \`DEFER\`.**

**\- Không bao giờ tìm GID bằng appearance trên toàn bãi.**  
**\- Ngưỡng ReID được fit từ positive/negative pairs của tập tuning, sau đó khóa trước held-out acceptance.**  
**\- Nếu model thiếu hoặc inference lỗi, runtime công khai \`reid\_status=degraded\` và dùng classic descriptor; không fallback âm thầm.**

**\#\#\# Giai đoạn D — Persistence và một nguồn session duy nhất**

**Dùng SQLite WAL với migration version. Các bảng chính:**

**\- \`identity\_sequence\`: bộ đếm GID theo site.**  
**\- \`identities\`: lifecycle và canonical GID.**  
**\- \`identity\_aliases\`: secondary → canonical.**  
**\- \`identity\_events\`: append-only audit.**  
**\- \`identity\_checkpoints\`: Kalman state, covariance, gallery metadata, camera, slot và corridor evidence.**  
**\- \`sessions\`: QR/session lifecycle, revision và fingerprint.**  
**\- \`reservations\`: lease theo session/GID/slot.**  
**\- \`runtime\_epochs\`: runtime ID, source mode, config hash và thời điểm khởi động.**

**Quy tắc ghi:**

**\- Mint, promote, handoff, park, unpark, alias, reservation, session và exit phải commit ngay.**  
**\- Kinematic checkpoint ghi tối đa mỗi giây; sau crash phải phục hồi với uncertainty tăng.**  
**\- Session và identity event nằm cùng transaction khi một thao tác tác động cả hai.**  
**\- Mọi mutation có \`revision\`, \`updated\_at\`, idempotency key và optimistic concurrency.**  
**\- Soft reset dựng lại tracker nhưng giữ database, GID, parked ownership và session.**  
**\- \`close-all\` là API quản trị riêng, xác nhận hai bước; chỉ đóng active state, không reset sequence.**

**\#\#\# Giai đoạn E — QR, reservation và vòng đời parking**

**\- Khi GID được activate tại entry gate, backend tự tạo một QR token ngẫu nhiên cho đúng GID đó.**  
**\- QR hết hạn sau 60 giây; token chỉ lưu dạng hash.**  
**\- Claim có body rỗng, idempotent và không nhận \`global\_vehicle\_id\`.**  
**\- State machine công khai:**

  **\`WAITING\_FOR\_SCAN → SELECTING\_SPOT → NAVIGATING → PARKED → EXIT\_NAVIGATION → CLOSED\`**

  **\`RECOVERY\_PENDING\` là trạng thái tạm thời khi restart/mất identity.**

**\- Chọn ô tạo lease nguyên tử 5 phút. Lease được gia hạn khi session còn active và GID vẫn có tiến triển hợp lệ về phía ô.**  
**\- Đổi ô phải tạo lease mới thành công trước rồi mới giải phóng lease cũ.**  
**\- Nếu ô mục tiêu bị một xe vật lý khác chiếm trong hai lần quan sát liên tiếp, backend chọn lại phương án trống xếp hạng tiếp theo, tăng session revision và phát event cho frontend thông báo.**  
**\- Endpoint do client gọi để tự đánh dấu \`parked\` phải bị xóa.**  
**\- Chỉ Slot Engine mới được set \`parkedSpotId\`. Nếu xe đỗ khác ô đã chọn, vị trí thực tế thắng và reservation cũ được giải phóng.**  
**\- \`EXIT\_NAVIGATION → PARKED\` bị cấm.**  
**\- Session chỉ thành \`CLOSED\` sau physical exit event; QR hết hạn không đồng nghĩa session bị đóng.**

**\#\#\# Giai đoạn F — Hợp nhất runtime và API v2**

**\- Thay hai session store bằng một \`RuntimeCoordinator\` duy nhất bao quanh pipeline, registry, SQLite store, session và reservation.**  
**\- \`techgar-api\` phải khởi động capture/replay adapter thật hoặc fail readiness; không được chạy API rỗng như hiện tại.**  
**\- Demo và live dùng cùng domain service, chỉ khác frame source.**  
**\- MJPEG lấy frame thật từ subscriber-gated encoder; bỏ heartbeat 1 pixel.**  
**\- Gate/topology editor ghi cấu hình versioned vào database/file commissioning, không chỉ nằm trong RAM.**  
**\- Thêm \`/api/health\` và \`/api/ready\`; readiness fail khi thiếu camera, calibration, database migration hoặc model bắt buộc.**

**Public contract chuyển sang \`schema\_version: "2.0"\`:**

**\- Runtime snapshot thêm \`site\_id\`, \`runtime\_id\`, \`registry\_revision\`, \`source\_mode\`, \`commissioning\_status\`, \`reid\_status\` và camera timestamps.**  
**\- Vehicle thêm \`canonical\_global\_id\`, \`identity\_confidence\`, \`association\_reason\` và trạng thái \`recovery\_pending/unknown\`.**  
**\- Session thêm \`revision\`, \`qrExpiresAt\`, \`reservation\`, \`parkedAt\`, \`exitStartedAt\` và \`closedAt\`.**  
**\- \`POST /api/sessions/{token}/claim\`: body rỗng.**  
**\- \`PUT /api/sessions/{id}/reservation\`: chọn hoặc đổi ô bằng expected revision.**  
**\- \`DELETE /api/sessions/{id}/reservation\`: giải phóng lease.**  
**\- \`POST /api/sessions/{id}/exit\`: yêu cầu dẫn đường ra.**  
**\- Xóa public \`POST .../parked\`.**  
**\- Thêm operator API cho alias proposal, alias confirmation, soft reset và close-all có xác nhận.**

**\#\#\# Giai đoạn G — Frontend**

**\- Chế độ live và demo tách bằng cấu hình rõ ràng. Lỗi API trong live phải hiển thị offline/error, tuyệt đối không chuyển sang demo.**  
**\- Frontend dùng session revision đơn điệu và bỏ response cũ đến trễ.**  
**\- Kiosk chỉ hiển thị QR động chưa hết hạn.**  
**\- Driver UI chỉ thấy GID của session đã claim và 48 slot thật từ snapshot.**  
**\- Giữ test render 160 slot như capacity benchmark, nhưng không biến fixture 160 ô thành cấu hình site thật.**  
**\- Hiển thị riêng \`occupied\`, \`reserved\`, \`occupied-unknown\`, \`recovery-pending\` và \`offline\`.**  
**\- Khi backend đổi ô vì conflict, cập nhật route và thông báo/voice đúng một lần theo event ID.**  
**\- Monitor bổ sung candidate/defer/quarantine, ReID provider, FPS, latency, database state, camera skew và commissioning status.**  
**\- Reset giao diện thành hai thao tác riêng: soft recovery và close-all có xác nhận hai bước.**

**\#\# 4\. Kiểm thử bắt buộc**

**\#\#\# Unit và invariant**

**\- GID vẫn tăng sau reset, restart và retire.**  
**\- Hai process cấp ID đồng thời không thể nhận cùng GID.**  
**\- Blob giữa bãi không mint.**  
**\- Cắt entry gate sai hướng không mint.**  
**\- LOST track được xét trước new track.**  
**\- Merged blob không ghi gallery và không tạo GID.**  
**\- Invalid topology không handoff dù appearance giống hoàn toàn.**  
**\- Ambiguous association trả \`DEFER\`.**  
**\- Alias remap session, reservation, slot và local ownership nguyên tử.**  
**\- Một GID không sở hữu hai active track cùng camera.**  
**\- Một slot không có hai lease hoặc hai owner.**  
**\- QR hết hạn đúng 60 giây; claim lặp không tạo session mới.**  
**\- Client không thể chọn GID hoặc tự đánh dấu parked.**  
**\- Session không đóng khi chỉ mất detection.**  
**\- Soft reset phục hồi parked identity; close-all không reset sequence.**  
**\- Model lỗi tạo trạng thái degraded và fallback có audit.**

**\#\#\# Synthetic/integration**

**\- Hai xe cùng màu giao nhau, nhập blob rồi tách ra: giữ hai GID, counter không tăng.**  
**\- Một xe qua blind gap và đổi camera: giữ GID.**  
**\- Xe xuất hiện giữa bãi: \`UNKNOWN\`.**  
**\- Restart khi xe đang di chuyển và khi xe đã đỗ.**  
**\- Camera stall/reconnect, frame trễ, overload và out-of-order timestamp.**  
**\- Hai session tranh cùng một ô.**  
**\- Xe lạ chiếm ô đã reservation và hệ thống đổi sang phương án tiếp theo.**  
**\- Đỗ sai ô, rời ô, đi ra cổng.**  
**\- SQLite crash/reopen và migration từ schema cũ.**  
**\- API v2 ↔ frontend contract, MJPEG thật và session polling race.**  
**\- E2E desktop/mobile cho kiosk → claim → chọn ô → parked → exit → closed.**

**\#\#\# Replay và nghiệm thu thật**

**\- M04 dùng để chẩn đoán crossing, parked retention và regression; không dùng để tuyên bố dense identity metric khi GT chưa đủ.**  
**\- Held-out real replay phải đạt:**

  **\- \`IDSW \= 0\` trên trajectory bắt buộc.**  
  **\- Mỗi physical vehicle có đúng một canonical GID.**  
  **\- Không có GID được mint giữa bãi.**  
  **\- Invalid handoff bằng 0\.**  
  **\- IDF1 ≥ 0.95 và MOTA ≥ 0.90 khi dense GT đủ.**  
  **\- Slot F1 ≥ 0.97.**  
  **\- Slot ownership accuracy ≥ 0.98.**  
  **\- Reservation collision bằng 0\.**  
  **\- Session survival qua gap/restart \= 100%.**  
  **\- GID reuse qua restart \= 0\.**  
  **\- Mean throughput ≥ 10 FPS trên máy Windows CPU đích.**  
  **\- Browser displayed-state latency p95 \< 1 giây.**  
  **\- Soak test tối thiểu 2 giờ không tăng ID bất thường, memory không tăng liên tục và database phục hồi được sau restart.**

**Synthetic, replay và live acceptance phải báo cáo tách biệt; test pass không được gọi là production acceptance.**

**\#\# 5\. Đầu ra hoàn chỉnh**

**\- Source chỉ thay đổi trong \`TechGAR-ver\_new\`; dự án cũ giữ nguyên làm reference/data source.**  
**\- Migration SQLite và công cụ backup/restore.**  
**\- Model downloader kèm license, URL và checksum.**  
**\- Data importer/validator và manifest không chứa absolute path.**  
**\- Bộ calibration/topology/48-slot config đã commissioning.**  
**\- Dense GT schema, annotation workflow và ba tập tuning/validation/held-out.**  
**\- Backend unit/integration/replay reports.**  
**\- Frontend Vitest, typecheck, lint, build và Playwright reports.**  
**\- Báo cáo acceptance tổng hợp, ghi rõ PASS/FAIL/N/A cho từng gate.**  
**\- Runbook PowerShell đầy đủ cho setup, import data, model download, migration, test, replay, live runtime, frontend và backup.**  
**\- Tài liệu clean-room ghi từng “ý tưởng từ bản cũ → invariant/test mới”, không chép đoạn code hoặc threshold cũ.**

**\#\# 6\. Các mặc định đã khóa**

**\- Mốc đầu: 2 camera, 48 ô; kiến trúc vẫn hỗ trợ mở rộng.**  
**\- Runtime đích: Windows CPU.**  
**\- ReID: hybrid, topology/chuyển động là cổng chính; neural embedding chỉ là bằng chứng phụ.**  
**\- GID: bền vững theo site và không tái sử dụng.**  
**\- Mint: chỉ tại entry gate đúng hướng.**  
**\- QR: động theo xe, hết hạn sau 60 giây.**  
**\- Reservation: lease 5 phút, có gia hạn và tự giải phóng.**  
**\- Commissioning: bắt đầu từ bãi trống.**  
**\- Dữ liệu cũ được dùng; code cũ không được sao chép.**  
**\- Persistence: SQLite WAL, event append-only và checkpoint.**  
**\- Reset: tách soft recovery và close-all có xác nhận.**

