# PLAN 7 — REAL-REPLAY IDENTITY FAILURE ANALYSIS AND ARCHITECTURAL FIX

Ngày lập: 2026-09-03  
Dataset điều tra chính: `droidcam_shared_vd_18`  
Phạm vi: Detection → Local Track → Projection/Fusion → Global Registry → Slot Owner

## 1. Mục tiêu và nguyên tắc

Mục tiêu là loại bỏ ID switch, ID fragmentation và slot-owner sai bằng bất biến
kiến trúc, không “chữa” replay bằng cách nới/siết threshold cho riêng video.

Thứ tự bắt buộc:

1. lưu đầy đủ bằng chứng của từng quyết định;
2. tái hiện lỗi deterministic;
3. phân loại lỗi theo tầng chịu trách nhiệm;
4. viết regression test cho bất biến;
5. sửa tầng thấp nhất gây lỗi;
6. chạy lại cùng replay và so sánh;
7. chỉ tính IDF1/IDSW khi có dense physical-ID ground truth.

Không dùng ground truth làm input dự đoán. Không dùng kết quả code cũ làm output mới.
Không tuyên bố IDF1/IDSW từ proxy.

## 2. Bằng chứng baseline đã thu được

### 2.1 M04 cũ

- Evaluator cũ đã FAIL do một canonical GID bị dùng cho nhiều physical vehicle,
  nhiều checkpoint thiếu GID và slot ownership bằng 0.
- New pipeline chạy 100 frame từng báo 27 occupied slot nhưng chỉ có 1 published
  Global ID. Đây là lỗi quyền sở hữu/lifecycle, không thể sửa bằng threshold IoU.
- `droidcam_shared_m_04` không có dense identity ground truth; chỉ có slot/event GT.

### 2.2 vd18 — replay 120 frame trước sửa

Output: `outputs/droidcam_shared_vd_18/20260903_181410_aa4dfe63`.

- 1 GID minted, 0 retired.
- 29 association defer, 99 mint bị chặn.
- 11 frame liên tiếp có một GID sở hữu hai Local ID đang được quan sát trên cùng cam2.
- Slot vision tự báo 20–21 occupied slot ngay từ đầu khi chưa có GID; audit cũ đếm
  2.518 frame-slot warning thay vì episode.
- Tại frame 110, cam2 Local ID 1 và 2 cùng mang owner GID 1. Cả hai world observation
  đều bị gate từ chối nhưng lịch sử owner vẫn còn, làm Global ID đứng yên trong khi
  local boxes tiếp tục nhảy.

### 2.3 vd18 — replay 120 frame sau lớp sửa đầu tiên

Output: `outputs/droidcam_shared_vd_18/20260903_181927_38054376`.

- Structural/proxy audit: 0 error, 0 warning.
- Cùng 120 frame, thời gian batch giảm từ 22,5 s xuống 13,4 s.
- Đây chưa phải kết luận toàn video và chưa phải ID metric có ground truth.

### 2.4 vd18 — full replay sau sửa

Output: `outputs/droidcam_shared_vd_18/20260903_183452_d864462f`.

- 1.321/1.321 synchronized pair, 0 skew reject, 0 overload.
- 4 provisional GID được mint; 3 GID được publish ở cuối; 1 provisional noise
  hypothesis hết hạn và chưa từng hiển thị.
- 0 owner reassignment, 0 owner-map transfer, 0 same-frame mint/retire storm,
  0 invalid slot owner và 0 anonymous occupied slot.
- Còn 19 active-local successor switch, 58 episode superseded Local ID tái xuất và
  561 quarantined observation. Đây là bằng chứng Phase C Local Tracker vẫn còn
  fragmentation/split-merge churn; Global Registry hiện đã chặn chúng khỏi đổi GID.
- Processing p95 khoảng 106 ms; tốc độ timestamp của recording trung bình 6,51 pair/s.
- Vì chưa có dense identity GT, 3 published GID không được coi là số physical vehicle
  đúng và report vẫn là `structural_and_proxy_only`.
- Contact sheet tại `outputs/diagnostics/vd18_identity_switch_contact_sheet.jpg` cho
  thấy nhiều contour ở mép ROI/chai nước/biên vật thể và split quanh xe. Vì chưa có
  commissioned entry gate, một motion artifact vẫn có thể vượt maturity; đây là lý do
  Phase C/F14 chưa được ký dù hard ownership đã pass.

## 3. Danh mục lỗi và hướng giải quyết

| ID | Lỗi | Nguyên nhân kiến trúc | Hướng giải quyết | Gate |
|---|---|---|---|---|
| F01 | Cùng Local ID đổi sang GID khác | owner map chỉ được ghi nhưng association không dùng; `_apply_match` ghi đè im lặng | owner là hard constraint; Registry là safety boundary và quarantine mọi transfer | 0 `owner_reassignment` |
| F02 | Kết quả recovery phụ thuộc thứ tự detection | `_maybe_spawn` greedily update LOST track ngoài assignment | đưa relaxed recovery vào một bipartite assignment duy nhất mỗi frame | một track/detection chỉ xuất hiện trong tối đa một recovery pair |
| F03 | Một GID có nhiều track cùng camera | owner history bị coi là active binding; track kế nhiệm và track cũ cùng tái xuất | tách historical owner khỏi active binding; superseded track bị quarantine, không mint/transfer | 0 `same_camera_duplicate_owner` |
| F04 | GID bị đứng/mất sau outlier | measurement bất khả thi chỉ tạo defer lặp lại; không có chuỗi hypothesis/outlier diagnosis | giữ GID, không mint; tích lũy recovery hypothesis theo thời gian và chỉ commit khi chuỗi đo vật lý nhất quán | không mint burst sau defer; recovery có evidence chain |
| F04a | Mint/retire một GID mỗi frame | template recovery bị relabel thành detection và Projector nhận timestamp cũ | lưu measurement source thật; template tạo synthetic observation ở timestamp hiện tại; tombstone ngăn Local ID tái sử dụng GID | 0 same-frame mint-retire; 0 owner reuse |
| F05 | Detector tách một xe thành nhiều blob | Local Tracker quản lý từng blob, chưa có split/alias hypothesis lâu dài | tạo split group: parent track + child blobs; đóng băng gallery; resolve joint khi tách/nhập ổn định | không tạo hai active local tracks cho cùng parent |
| F06 | Hai xe giao/cắt bị đổi local identity | quyết định từng frame thiếu permutation hypothesis qua occlusion | giữ N hypothesis qua group; joint assignment khi ra khỏi group dùng motion trước merge + appearance sạch + ordering/topology | crossing regression `IDSW=0` |
| F07 | Handoff camera sai/duplicate | ownership, overlap fusion và handoff đang là các quyết định gần như độc lập | handoff transaction: source-exit evidence → pending window → target-entry evidence → atomic active-binding switch | 0 invalid handoff; 1 GID xuyên camera |
| F08 | Provisional ID chiếm slot | slot stage từng dùng mọi live identity | chỉ ACTIVE/PARKED có quyền claim slot | 0 slot owner là provisional/non-live |
| F09 | Occupancy báo xe từ frame đầu | học “empty reference” từ cảnh đã có xe và dùng texture heuristic như production evidence | fail-closed khi chưa có commissioned empty reference; sau đó so sánh 25 biến thể gamma/CLAHE giữa current và reference đã đăng ký | 0 anonymous false occupied khi uncommissioned |
| F10 | Slot giữ occupied sau owner biến mất | release phụ thuộc tracker owner; thiếu channel xác nhận vật thể rời slot | tách occupancy truth khỏi identity ownership; release cần visual-empty commissioned hoặc exit/departure transaction | 0 orphaned slot-owner episode |
| F11 | Audit phóng đại lỗi kéo dài | đếm mỗi frame thay vì episode | report episode start/end/duration và ví dụ đầu tiên | số lỗi phản ánh incident, không phản ánh FPS |
| F12 | Không thể chứng minh physical ID | vd17/vd18 thiếu dense identity GT | annotation ở anchor/checkpoint + occlusion/handoff spans, independent review | evaluator tính được IDF1/IDSW thật |
| F13 | Motion artifact được coi là xe | camera/global jitter, chai nước, mép ROI, bóng và đường viền xe đỗ sinh contour | global-motion compensation trước background; contour consolidation; temporal rigid-body/world-size eligibility trước tracker/admission | artifact không được promote/mint |
| F14 | GID có thể sinh giữa bãi | replay config chưa có external entry gate được commissioning; `require_entry_gate` không thể bật an toàn | vẽ và đo entry/exit gate thật, kiểm tra hướng crossing; mid-lot candidate ở UNKNOWN, không cấp GID | mọi MINT có gate-crossing evidence |
| F15 | Overlay tự làm ID nhảy | renderer từng gán box vào marker gần nhất trong bán kính 120 px | renderer chỉ đọc exact Registry binding; unbound track hiện `L#` | UI không có identity heuristic |

## 4. Các bất biến bắt buộc

1. Một `(camera_id, local_track_id)` không bao giờ được chuyển sang GID khác trong
   cùng runtime.
2. Một GID chỉ có tối đa một active Local ID trên mỗi camera; Local ID cũ chỉ còn
   historical/superseded.
3. Một observation chỉ cập nhật tối đa một local track và một GID trong một frame.
4. Một GID chỉ nhận tối đa một fused observation trong một synchronized pair.
5. Measurement vi phạm physics/topology bị defer hoặc quarantine; không mint để né lỗi.
6. GID chỉ sinh tại evidence chain hợp lệ và phải qua provisional maturity.
7. Provisional identity không được claim slot.
8. Vision occupancy không được bật nếu thiếu empty-reference commissioning.
9. Parked slot ownership chỉ trỏ tới GID live/published; occupancy có thể anonymous
   chỉ khi vision channel đã commissioned và báo rõ là anonymous.
10. Handoff là chuyển binding nguyên tử, không phải hai match độc lập.

## 5. Kế hoạch triển khai theo phase

### Phase A — Observability và baseline — DONE

- `predictions.jsonl` lưu detection, local observation, world/fused observation,
  association decision/cost, ingest result, owner binding và tracker decision.
- `techgar.identity_audit` kiểm tra hard invariant và proxy.
- `techgar.demo --batch` chạy deterministic, lưu output rồi thoát, không cần port.
- Audit tự sinh `identity_audit.json` trong mỗi run.

Exit: có thể truy từ box nhảy trên video về đúng decision và cost gây ra nó.

### Phase B — Ownership safety boundary — DONE ON VD18 STRUCTURAL AUDIT

- Owner constraint được áp trước Hungarian assignment.
- Owner-bound GID được reserve, observation khác không được hijack row đó.
- Registry quarantine nếu caller cố chuyển historical Local ID sang GID khác.
- Active binding tách khỏi historical ownership; superseded observation không quay
  lại global association.
- Provisional identity không được claim slot.
- Demo overlay chỉ render Registry binding, không nearest-neighbour đoán GID.
- Template recovery giữ source/timestamp hiện tại; historical Local ID tồn tại như
  tombstone sau khi owner retire.
- Active binding có same-camera grace, nên fragment mới không thể làm GID đổi local
  anchor từng frame.

Exit: full vd18 có 0 owner reassignment, 0 same-camera duplicate active owner,
0 slot owner non-live.

### Phase C — Local split/merge/crossing hypothesis — OPEN

- Thêm camera/global-motion compensation bằng feature/RANSAC trên vùng nền tin cậy;
  nếu transform bất ổn thì frame là `environment_unstable` và cấm spawn/mint.
- Motion mask chỉ tạo proposal; proposal phải qua contour consolidation, calibrated
  world-size và temporal rigid-body eligibility mới được thành vehicle hypothesis.
- Thay mọi recovery side effect bằng one-to-one assignment — phần này đã làm.
- Bổ sung persistent split/merge group, parent/child relation và group resolution.
- Không học appearance từ merged, partial, shadow hoặc low-quality blob.
- Khi group tách, giải permutation joint qua nhiều frame; nếu ambiguous thì giữ
  identity pending thay vì đổi ID.

Exit: synthetic crossing, merge, split, stop-and-go và vd18 không phát sinh local
fragment successor gần track vừa mất.

### Phase D — Global recovery và handoff transaction — OPEN

- Tạo recovery hypothesis buffer cho các chuỗi defer liên tiếp.
- Score cả chuỗi theo reachable tube, direction, clean appearance và topology;
  commit một lần sau minimum evidence, không match từng frame độc lập.
- Handoff state machine `SOURCE_ACTIVE → EXIT_PENDING → TARGET_CANDIDATE → COMMITTED`;
  timeout quay lại missing, không mint GID mới.
- Commission external entry/exit gate; mọi detection giữa bãi không có gate evidence
  mang state UNKNOWN và không được Registry mint.

Exit: 0 invalid handoff, 0 duplicate GID qua overlap, 0 mint trong pending handoff.

### Phase E — Slot truth và ownership transaction — PARTIAL

- Provisional claim đã bị cấm; uncommissioned vision đã fail-closed.
- Cần công cụ capture empty reference cho từng slot/camera, lưu hash + exposure range.
- Current/reference phải được đăng ký hình học; camera drift làm channel `unknown`,
  không tự đổi occupied/empty.
- 25 gamma/CLAHE variants tạo consensus occupancy; tracker chỉ cấp owner.
- Arrival/park/departure/exit là transaction có audit event và rollback.

Exit: M04/vd datasets đạt slot F1/ownership gate bằng GT, không orphan/flicker.

### Phase F — Ground truth và evaluator thật — OPEN

- Annotate `ground_truth_identity.csv`: physical vehicle, frame, camera, anchor,
  required checkpoint, occlusion/handoff phase.
- Annotator thứ hai review các checkpoint và disagreement.
- Tính IDF1, IDSW, fragmentation, handoff accuracy, false mint, slot F1,
  ownership và false vacancy.

Exit: không còn metric “proxy only”; report có coverage và uncertainty.

### Phase G — Acceptance và soak — OPEN

- Chạy full vd17, vd18, M04 và nuisance/ablation.
- Soak camera stall/reconnect, DroidCam jitter, duplicate frame, delayed frame.
- So sánh run manifest/hash để bảo đảm reproducible.

Hard gate: `IDSW=0`, invalid handoff `=0`, ownership `>=0.98`, slot F1 `>=0.97`,
không hard-invariant error. Chỉ ký khi calibration và GT đã accepted.

## 6. Lệnh kiểm chứng

```powershell
# Toàn bộ regression suite
.venv\Scripts\python -m pytest

# Replay offline có lưu trace, không dùng port
.venv\Scripts\python -m techgar.demo --dataset droidcam_shared_vd_18 --batch --output-root outputs

# Audit lại một run đã có
.venv\Scripts\python -m techgar.identity_audit outputs\droidcam_shared_vd_18\<RUN_ID>

# Demo web để xem bằng mắt
.venv\Scripts\python -m techgar.demo --dataset droidcam_shared_vd_18 --speed 1.0 --port 8011
```

## 7. Quy tắc quyết định tiếp theo

- Nếu hard invariant fail: sửa logic trước, không tune.
- Nếu invariant pass nhưng video vẫn “nhảy box”: lỗi Local Tracker/detector, xử lý
  split/merge hypothesis; không đổ cho Global ID.
- Nếu Local ID ổn nhưng GID mất khi đổi camera: sửa handoff transaction/topology.
- Nếu GID ổn nhưng slot sai: sửa occupancy reference/slot transaction độc lập.
- Nếu mọi proxy pass nhưng cảm quan sai: bổ sung GT đúng đoạn đó rồi mới tối ưu.
