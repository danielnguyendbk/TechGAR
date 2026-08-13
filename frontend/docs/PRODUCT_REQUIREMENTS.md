# Product Requirements — Smart Parking Driver Frontend

## Product goal
Help a driver understand parking availability and optionally find a stable empty spot near one of three broad mall needs: Shopping, Dịch vụ, or Giải trí.

## Primary users
Drivers entering a shopping-mall parking lot from a QR code at the gate.

## Core jobs
1. Quickly see which spots are empty.
2. Optionally receive a suggestion matching a broad destination need.
3. Confirm a spot before navigation begins.
4. Follow a lane-valid route to the spot.
5. Recover safely when the chosen spot changes status.

## Page modes
### Entry
Initial bottom sheet over the map:
- `Nhận đề xuất vị trí đỗ xe`
- `Chỉ xem các ô đang trống`
- `Bỏ qua`

### Browse
- Full map visible.
- User may toggle empty-only or all statuses.
- User may inspect any spot.
- Only green spots expose `Chỉ đường đến {spotId}`.
- Persistent button: `Tìm chỗ phù hợp`.

### Recommendation
- Three choices only: Shopping, Dịch vụ, Giải trí.
- No subcategory step.
- Result card includes:
  - chosen need;
  - recommended spot ID;
  - zone;
  - estimated walking time;
  - approximate distance;
  - short reason;
  - two alternatives;
  - explicit confirmation button;
  - abandon-recommendation action;
  - non-reservation disclaimer.

### Navigation
- Route appears only after confirmation.
- Header/status bar states selected spot and zone.
- User can cancel route.
- If selected spot becomes invalid, route pauses and warning sheet appears.

## Summary information
Show compact cards:
- Còn trống
- Đã có xe
- Đang chuyển tiếp
- Không xác định
- Camera 2/2 online or degraded state

## Required Vietnamese copy
- `Bạn muốn tìm chỗ đỗ theo cách nào?`
- `Nhận đề xuất vị trí đỗ xe`
- `Chỉ xem các ô đang trống`
- `Bỏ qua`
- `Bạn muốn đến khu vực nào?`
- `Shopping`
- `Dịch vụ`
- `Giải trí`
- `Đề xuất tốt nhất`
- `Phương án khác`
- `Chọn {spotId} và chỉ đường`
- `Bỏ gợi ý và xem toàn bộ bãi`
- `Vị trí không được giữ trước và có thể thay đổi theo tình trạng thực tế.`
- `Tìm chỗ phù hợp`
- `Chỉ hiện ô trống`
- `Hiện tất cả trạng thái`

## Accessibility
- Status must have text and aria labels, not color alone.
- Selected spot must have `aria-current` or equivalent semantic indicator.
- Bottom sheet traps focus and restores it on close.
- All actionable elements have at least 44×44 px hit target.
