# TECHGAR — PLAN 5: FRONTEND ALGORITHMIC FORMULATIONS, MATHEMATICAL MODELS & MECHANISM SPECIFICATIONS

> **Loại tài liệu**: Toán học, hình học, logic hiển thị thuần — KHÔNG có code
> **Liên kết**: PLAN 4 (workflow frontend) · PLAN 6 (benchmark frontend) · PLAN 1/2 (backend: nguồn dữ liệu)
> **Ký hiệu**: $\mathbf{w}=(X,Y)$ world-cm; $\mathbf{s}=(x,y)$ SVG-px; $g$ Global ID; $t$ thời gian publish

---

## 1. World→SVG Affine Projection

### 1.1. Mô hình affine 2D

Mọi điểm world $\mathbf{w}_i$ chiếu lên SVG $\mathbf{s}_i$ qua phép affine:

\[
\mathbf{s}_i
=
A\mathbf{w}_i+\mathbf{b},
\qquad
A=
\begin{bmatrix}
a_{11}&a_{12}\\
a_{21}&a_{22}
\end{bmatrix},
\quad
\mathbf{b}=
\begin{bmatrix}
b_1\\b_2
\end{bmatrix}
\]

Dạng thuần nhất:

\[
\begin{bmatrix}
x_i\\y_i
\end{bmatrix}
=
\underbrace{\begin{bmatrix}
X_i&Y_i&1\\
\end{bmatrix}}_{\mathbf{m}_i^\top}
\underbrace{\begin{bmatrix}
a_{11}\\a_{12}\\b_1
\end{bmatrix}},
\qquad
y_i =
\mathbf{m}_i^\top
\begin{bmatrix}
a_{21}\\a_{22}\\b_2
\end{bmatrix}
\]

Hai trục giải độc lập — mỗi trục là một bài toán linear least squares 3 tham số.

### 1.2. Least-squares fit từ cặp tâm slot

Cho $N$ cặp tương ứng (tâm slot world $\mathbf{w}_k$ ↔ tâm slot SVG $\mathbf{s}_k$), $N\ge 3$:

\[
A^\*,\mathbf{b}^\*
=
\arg\min_{A,\mathbf{b}}
\sum_{k=1}^{N}
\left\|
A\mathbf{w}_k+\mathbf{b}-\mathbf{s}_k
\right\|^2
\]

Dạng ma trận (mỗi trục):

\[
\hat{\boldsymbol\theta}
=
\left(M^\top M\right)^{-1}M^\top\mathbf{y}
\qquad
M\in\mathbb{R}^{N\times3},\;
\mathbf{y}\in\mathbb{R}^{N}
\]

với hàng $k$ của $M$ là $[X_k,\;Y_k,\;1]$.

### 1.3. Điều kiện khả dụng & fallback

Phép fit khả dụng khi:

\[
\operatorname{rank}(M)=3
\quad\land\quad
\operatorname{cond}\left(M^\top M\right)<\kappa_{max}
\]

Nếu violated (slot degenerate, layout thiếu) → **identity fallback**: $\mathbf{s}=\operatorname{clamp}(\mathbf{w})$ vào canvas + cờ cảnh báo — map vẫn render slot tĩnh, không crash.

### 1.4. Nghịch đảo (svgToWorld)

\[
\mathbf{w}
=
A^{-1}(\mathbf{s}-\mathbf{b})
\quad\text{khi}\quad
\left|\det A\right|>\epsilon
\]

Nghịch đảo dùng cho click cấu hình cổng (PLAN 4 Stage F8). $\det A\approx 0$ → trả `null`, chặn thao tác.

### 1.5. Sai số chuẩn chiếu

Residual fit:

\[
r_{fit}
=
\sqrt{
\frac{1}{N}
\sum_k
\left\|
A^\*\mathbf{w}_k+\mathbf{b}^\*-\mathbf{s}_k
\right\|^2
}
\]

**Chuẩn chấp nhận:** $r_{fit} \le 2$ px. $r_{fit}$ vượt ngưỡng → cảnh báo cấu hình (không âm thầm sai lệch).

**Pass**: tâm D08 world (100, 200)cm → SVG nằm trong 2 px của tâm hình chữ nhật D08 chuẩn.

**Fail**: chiếu ngược chiều (dùng $A^{-1}$ thay $A$) — marker hiện ở góc đối xứng của map.

---

## 2. Marker Motion Smoothing & Teleport Guard

### 2.1. Nội suy hiển thị giữa hai snapshot

Snapshot đến chu kỳ $T\approx 1$ s; marker không nhảy giật mà chuyển động mượt:

\[
\mathbf{s}_{render}(\tau)
=
\mathbf{s}_{prev}
+
\left(\mathbf{s}_{new}-\mathbf{s}_{prev}\right)\cdot
\min\left(1,\;\frac{\tau}{T_{anim}}\right)
\]

với $T_{anim}=350$ ms (CSS transform transition), $\tau$ = thời gian từ snapshot.

### 2.2. Teleport guard (chặn bay xuyên bãi)

Phát hiện dịch chuyển phi vật lý giữa hai snapshot liên tiếp của CÙNG Global ID:

\[
d_{jump}
=
\left\|
\mathbf{w}_{new}-\mathbf{w}_{prev}
\right\|
\]

Ngưỡng chuyển snap:

\[
d_{snap}
=
v_{max}^{display}\cdot(T_{poll}+\epsilon)
+\rho_{seam}^{display}
\]

với $v_{max}^{display}$ = tốc tốc hiển thị tối đa (lấy theo vận tốc tối đa backend + margin) và $\rho_{seam}^{display}$ = allowance tái xuất hiện sau mất dấu.

\[
\begin{cases}
d_{jump}\le d_{snap} &\Rightarrow \text{transition mượt (§2.1)}\\
d_{jump}>d_{snap} &\Rightarrow \text{snap tức thì — KHÔNG animate}
\end{cases}
\]

**Lý do**: GID bị mất dấu 3 giây rồi tái xuất ở đầu kia (re-acquisition hợp lệ) — nếu animate, marker bay xuyên toàn bãi gây hiểu nhầm "xe chạy siêu tốc", tệ hơn là che mất đúng đích.

### 2.3. Chuyển động trạng thái parked

\[
\text{state}=\text{parked}
\Rightarrow
\mathbf{s}_{render}
\equiv
\mathbf{s}_{slot\_center}
\quad(\text{tĩnh tuyệt đối, không transition})
\]

Xe đỗ vẽ tại tâm slot (từ polygon chiếu §1), bất kể vị trí last-seen — đúng intent "fallback tâm slot".

**Pass**: GID 17 đỗ D07 → marker đứng yên tại tâm D07 qua 100 snapshot.

**Fail**: marker đỗ vẫn khẽ rung theo last-seen position mỗi snapshot (dùng vị trí thay vì tâm slot).

---

## 3. Display-Hold State Machine (formal hóa)

### 3.1. Hàm chuyển trạng thái hiển thị

Với mỗi xe $v$ trong snapshot tại thời điểm render $t$:

\[
\text{visible}(v,t)
=
\begin{cases}
\text{true}, & v.observed\\
\text{true}, & v.parked\_slot\_id\neq\text{null}\\
\text{true}, & v.stale\_seconds\le v.display\_hold\_seconds\\
\text{false}, & \text{ngược lại}
\end{cases}
\]

\[
\text{style}(v)
=
\begin{cases}
\text{moving}, & v.observed\land v.state\neq\text{parked}\\
\text{parked}, & v.parked\_slot\_id\neq\text{null}\\
\text{missing}, & \neg v.observed\land v.parked\_slot\_id=\text{null}\land v.stale\le hold
\end{cases}
\]

Ba trạng thái visual tách bạch: **moving** (đỏ + halo động) · **parked** (🔒 xanh, tĩnh) · **missing** (mờ + nhãn "tạm mất dấu", giữ vị trí last-seen).

### 3.2. Bất biến nguồn dữ liệu

Mọi tham số của §3.1 là **thuộc tính snapshot**:

\[
stale\_seconds,\;display\_hold\_seconds,\;observed,\;parked\_slot\_id
\;\in\;
\text{RuntimeSnapshot}
\]

Frontend **không được** tự tính stale bằng local clock trừ `lastPublishedAt` trừ khi snapshot thiếu trường (fallback có đánh dấu `deprecated`).

### 3.3. Bất biến không nhấp nháy

Số lần marker đổi giữa `visible ↔ hidden` trong cửa sổ $W_{flicker}=3$ s:

\[
N_{flip}(v,W_{flicker})\le 1
\]

vượt ngưỡng → frontend đang tự quyết hiển thị sai (backend đã hysteresis; frontend vi phạm bảng §3.1).

**Pass**: 10 snapshot liên tiếp: observed→missing→observed→missing (hold đập đúng) → marker không biến mất lần nào.

**Fail**: marker biến mất ngay snapshot missing đầu tiên (hold bị bỏ qua).

---

## 4. Lane-Graph Routing & Guidance

### 4.1. Đồ thị làn đường

\[
G=(V,E),\qquad
E\subseteq V\times V
\]

- $V$: nút làn (giao làn, đầu ô, cổng) — tọa độ $\mathbf{w}\in$ world
- $E$: đoạn làn chạy được — polygon hành lang không cắt slot/tường

Ràng buộc hình học: $\forall e\in E$: đoạn thẳng/đường đi của $e$ KHÔNG cắt bất kỳ slot polygon nào (ngoài điểm cuối chạm đầu slot).

### 4.2. Hàm chi phí cạnh

\[
c(e)
=
\left\|e\right\|_{2}
\cdot
\left(1+\lambda_{turn}\,\mathbf{1}[e\text{ rẽ}]\right)
\]

với $\lambda_{turn}$ phạt rẽ (ước lượng độ khó lái). Trọng số rẽ làm tuyến thiên về đường thẳng — dễ lái hơn tuyến zigzag ngắn hơn tí.

### 4.3. Tuyến ngắn nhất (Dijkstra)

Từ vị trí xe $\mathbf{w}_v$ (nút gần nhất — nearest node):

\[
u_0
=
\arg\min_{u\in V}
\left\|\mathbf{w}_v-\mathbf{w}_u\right\|
\]

\[
P^\*
=
\operatorname{Dijkstra}(G,\;c,\;u_0,\;u_{slot})
\]

Điểm đích $u_{slot}$ = nút đầu vào của slot đích. Tuyến $P^\*$ chiếu SVG bằng phép §1 — **cùng phép chiếu duy nhất**, không phép riêng cho route.

### 4.4. Off-route detection

Khoảng cách điểm-tới-tuyến của vị trí xe hiện tại:

\[
d_{route}(t)
=
\min_{\mathbf{p}\in P^\*}
\left\|
\mathbf{w}_v(t)-\mathbf{p}
\right\|
\]

Cảnh báo khi thỏa CẢ HAI (chống false-positive khi snapshot hold giữ vị trí cũ):

\[
d_{route}(t)>d_{off}
\quad\land\quad
v.observed=\text{true}
\]

Hành vi cảnh báo: giọng nói khẩn cấp + banner đỏ — **KHÔNG âm thầm vẽ lại tuyến** (người lái chủ động).

### 4.5. Sinh chỉ dẫn từ hình học tuyến

Từ chuỗi đoạn $P^\*=(p_0,p_1,\ldots,p_n)$, tại mỗi đổi hướng tính góc:

\[
\theta_i
=
\operatorname{atan2}
\left(
\mathbf{d}_{i+1}\times\mathbf{d}_i,\;
\mathbf{d}_{i+1}\cdot\mathbf{d}_i
\right),
\qquad
\mathbf{d}_i=p_i-p_{i-1}
\]

Phân loại chỉ dẫn:

\[
\text{instruction}(\theta_i)
=
\begin{cases}
\text{đi thẳng}, & |\theta_i|<30^\circ\\
\text{rẽ trái/phải}, & 30^\circ\le|\theta_i|<150^\circ\\
\text{quay ngược}, & |\theta_i|\ge150^\circ
\end{cases}
\]

Kích hoạt thoại khi xe tới gần nút đổi hướng ($\|\mathbf{w}_v-p_i\|<r_{speak}$) và chỉ dẫn chưa phát.

**Pass**: tuyến có 1 rẽ phải → đúng 1 lệnh "Phía trước rẽ phải vào làn đỗ" phát đúng lúc gần nút, không phát lặp.

**Fail**: phát cả ba lệnh cùng lúc lúc bắt đầu điều hướng (không có trigger theo vị trí).

---

## 5. Session–Parking Resolution (trang tài xế)

### 5.1. Vị trí hiển thị xe của phiên

\[
\mathbf{w}_{display}
=
\begin{cases}
\mathbf{w}_v, & \exists v:\;v.global\_id=session.gid\;\land\;v\text{ visible}\\
\mathbf{c}(slot_{parked}), & session.parked\_slot\_id\neq\text{null}\\
\mathbf{c}(slot_{target}), & \text{mất dấu}\;\land\;\text{đang điều hướng tới }slot_{target}\\
\varnothing\;(\text{trang kết thúc}), & session=404
\end{cases}
\]

($\mathbf{c}(\cdot)$ = tâm slot polygon.) Quy tắc ưu tiên: observation thật > slot đang đỗ > slot đích — không bao giờ render "không có gì" khi phiên còn sống.

### 5.2. Bộ lọc xe của phiên

\[
V_{driver}
=
\left\{
v\in V_{display}:\;
v.global\_id=session.global\_vehicle\_id
\right\}
\]

— đúng MỘT xe hoặc rỗng (rỗng → dùng §5.1 fallback, KHÔNG hiện xe người khác).

**Pass**: 3 xe chạy trên map monitor, trang tài xế chỉ hiện 1 xe (của session).

**Fail**: trang tài xế hiện đủ 3 xe (filter bị bỏ).

---

## 6. Polling & Backoff (uống lỗi mạng)

### 6.1. Exponential backoff

\[
\Delta_{retry}^{(n)}
=
\min\left(
\Delta_{max},\;
\Delta_{0}\cdot 2^{\,n}
\right),
\qquad
\Delta_0=1\,\text{s},\;\Delta_{max}=5\,\text{s}
\]

Thành công → reset $n=0$. Kết nối state:

\[
state
=
\begin{cases}
\text{live}, & \text{thành công}\land \text{staleness}\le 2\,\text{s}\\
\text{stale}, & \text{thành công}\land \text{staleness}> 2\,\text{s}\\
\text{connecting}, & \text{chưa có snapshot}\\
\text{error}, & \text{lỗi liên tục}
\end{cases}
\]

### 6.2. Coalesce request

Tại mọi thời điểm, số request snapshot đang bay $\le 1$:

\[
N_{inflight}(t)\le 1
\]

request mới trong lúc một request đang chờ → chia sẻ promise cũ (test: 100 lần render liên tiếp tạo đúng 1 request).

---

## 7. QR Kiosk & Deep-Link

### 7.1. Liên kết phiên

\[
\text{URL}_{driver}
=
\text{base}
+
\text{/?session=}
+
\text{sessionId}
\]

QR encode URL này. Trang tài xế parse `sessionId` → claim MỘT lần (guard idempotent — ref-lock, không claim lại khi re-render/StrictMode double-invoke).

### 7.2. Kiosk polling phiên chờ

\[
V_{waiting}
=
\text{GET /api/sessions/waiting}
\]

Hiển thị QR cho mỗi phiên chờ; phiên được claim (bởi ai đó quét) → biến khỏi danh sách — kiosk không giữ stale.

---

## 8. Bảng tham chiếu ký hiệu frontend

| Ký hiệu | Định nghĩa |
|---|---|
| $A,\mathbf{b}$ | affine world→SVG (fit least squares) |
| $r_{fit}$ | residual chiếu (chuẩn ≤ 2 px) |
| $T_{anim}$ | thời gian transition marker (350 ms) |
| $d_{jump},d_{snap}$ | bước nhảy vị trí / ngưỡng snap |
| $visible(v,t)$, $style(v)$ | hàm trạng thái hiển thị (bảng chân lý) |
| $N_{flip}$ | số lần visible↔hidden trong cửa sổ anti-flicker |
| $G=(V,E)$, $c(e)$ | đồ thị làn / chi phí cạnh |
| $P^\*$, $d_{route}(t)$ | tuyến Dijkstra / khoảng cách lệch tuyến |
| $\theta_i$ | góc đổi hướng → chỉ dẫn thoại |
| $\Delta_{retry}^{(n)}$ | chu kỳ backoff |
