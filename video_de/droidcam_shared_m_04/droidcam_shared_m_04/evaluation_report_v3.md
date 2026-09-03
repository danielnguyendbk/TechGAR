# TechGAR Practical System Report — droidcam_shared_m_04

- Kết luận: **FAIL**
- Practical System Score: **40.85/100**
- Điểm trước giới hạn critical: 40.85/100
- Critical errors: **1**

## Điểm thành phần

| Thành phần | Trọng số | Điểm |
|---|---:|---:|
| Identity continuity + handoff | 35% | 31.39 |
| Đúng chủ sở hữu ô | 30% | 0.00 |
| Departure recovery / ReID | 15% | 84.00 |
| Occupied / free | 15% | 97.43 |
| Delay + stability | 5% | 52.97 |

## Occupancy

| Metric | Raw | Practical |
|---|---:|---:|
| Occupied F1 | 0.9729 | 0.9741 |
| Balanced accuracy | 0.9735 | 0.9746 |
| False-free rate | 0.0528 | 0.0504 |
| False-occupied rate | 0.0003 | 0.0003 |

## Critical errors

| Code | Frame(s) | Physical vehicle(s) | Camera(s) | Slot | Expected GID | Predicted GID(s) | Duration | Details |
|---|---:|---|---|---|---:|---:|---:|---|
| gid_shared_between_vehicles | 300–650 | M04_V1, M04_V4 | cam1, cam2 | n/a | n/a | 3 | n/a | Canonical GID 3 is observed for multiple physical vehicles |

## Non-critical misses / delays

| Code | Frame(s) | Physical vehicle(s) | Camera(s) | Slot | Expected GID | Predicted GID(s) | Duration | Details |
|---|---:|---|---|---|---:|---:|---:|---|
| long_unbound_gid | 53–913 | n/a | n/a | n/a | n/a | 3 | 302 frames | Canonical GID 3 remained visible for 302 frames without ever owning or reserving a slot |
| long_unbound_gid | 238–327 | n/a | n/a | n/a | n/a | 4 | 88 frames | Canonical GID 4 remained visible for 88 frames without ever owning or reserving a slot |
| long_unbound_gid | 582–848 | n/a | n/a | n/a | n/a | 7 | 226 frames | Canonical GID 7 remained visible for 226 frames without ever owning or reserving a slot |
| long_unbound_gid | 619–724 | n/a | n/a | n/a | n/a | 8 | 46 frames | Canonical GID 8 remained visible for 46 frames without ever owning or reserving a slot |
| long_unbound_gid | 803–881 | n/a | n/a | n/a | n/a | 9 | 44 frames | Canonical GID 9 remained visible for 44 frames without ever owning or reserving a slot |
| unmapped_physical_vehicle | n/a | M04_V4 | n/a | n/a | n/a | n/a | n/a | M04_V4 has no required independent checkpoint matched to a valid GID |
| missing_gid_at_checkpoint | 40 | M04_V1 | cam2 | B05 | 3 | n/a | n/a | M04_V1 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 80 | M04_V1 | cam2 | n/a | 3 | n/a | n/a | M04_V1 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 200 | M04_V1 | cam2 | n/a | 3 | n/a | n/a | M04_V1 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 350 | M04_V1 | cam1 | n/a | 3 | n/a | n/a | M04_V1 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 450 | M04_V1 | cam1 | F02 | 3 | n/a | n/a | M04_V1 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 520 | M04_V1 | cam1 | F02 | 3 | n/a | n/a | M04_V1 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 584 | M04_V1 | cam1 | n/a | 3 | n/a | n/a | M04_V1 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 100 | M04_V2 | cam2 | B04 | 4 | n/a | n/a | M04_V2 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 200 | M04_V2 | cam2 | B04 | 4 | n/a | n/a | M04_V2 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 280 | M04_V2 | cam2 | n/a | 4 | n/a | n/a | M04_V2 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 300 | M04_V2 | cam2 | n/a | 4 | n/a | n/a | M04_V2 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 350 | M04_V2 | cam1 | n/a | 4 | n/a | n/a | M04_V2 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 700 | M04_V2 | cam1 | n/a | 4 | n/a | n/a | M04_V2 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 100 | M04_V3 | cam1 | F01 | 7 | n/a | n/a | M04_V3 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 545 | M04_V3 | cam1 | n/a | 7 | n/a | n/a | M04_V3 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 560 | M04_V3 | cam1 | n/a | 7 | n/a | n/a | M04_V3 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 100 | M04_V4 | cam1 | F03 | n/a | n/a | n/a | M04_V4 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 610 | M04_V4 | cam1 | n/a | n/a | n/a | n/a | M04_V4 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 650 | M04_V4 | cam1 | n/a | n/a | n/a | n/a | M04_V4 has no valid Global ID at a required checkpoint |
| missing_gid_at_checkpoint | 700 | M04_V4 | cam1 | n/a | n/a | n/a | n/a | M04_V4 has no valid Global ID at a required checkpoint |
| missed_slot_binding | 1 | M04_V3 | cam1 | F01 | 7 | n/a | n/a | cam1/F01 did not bind canonical GID 7 within 75 frames |
| missed_slot_binding | 421 | M04_V1 | cam1 | F02 | 3 | n/a | n/a | cam1/F02 did not bind canonical GID 3 within 75 frames |
| missed_slot_binding | 1 | M04_V4 | cam1 | F03 | n/a | n/a | n/a | cam1/F03 did not bind canonical GID None within 75 frames |
| missed_slot_binding | 1 | M04_V2 | cam2 | B04 | 4 | n/a | n/a | cam2/B04 did not bind canonical GID 4 within 75 frames |
| missed_slot_binding | 1 | M04_V1 | cam2 | B05 | 3 | n/a | n/a | cam2/B05 did not bind canonical GID 3 within 75 frames |
| missed_departure_recovery | 598 | M04_V4 | cam1 | F03 | n/a | n/a | n/a | M04_V4 did not recover canonical GID None within 125 frames |

## Quy tắc kết luận

Một critical error làm phiên FAIL và giới hạn điểm tối đa 49. PASS yêu cầu tổng điểm ≥85 và cả identity, slot ownership, recovery đều ≥95.
