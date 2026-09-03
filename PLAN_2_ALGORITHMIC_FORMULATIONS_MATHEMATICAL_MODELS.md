# TECHGAR — PLAN 2: ALGORITHMIC FORMULATIONS, MATHEMATICAL MODELS & MECHANISM SPECIFICATIONS

> **Loại tài liệu**: Pure math, geometry, logic — KHÔNG có code
> **Liên kết**: PLAN 1 (pipeline & roadmap) · PLAN 3 (benchmark & rubric)
> **Ký hiệu chuẩn**: $I_t$ = frame tại thời điểm $t$; $\mathbf{w}$ = world position; $g$ = Global ID; $H$ = homography

---

## 1. Motion Segmentation & Adaptive Noise Rejection

### 1.1. Background-subtraction evidence

Frame grayscale hiện tại $I_t(x,y)$; nền ước lượng $B_t(x,y)$:

\[
M^{bg}_t(x,y)
=
\mathbf{1}
\left(
\left|I_t(x,y)-B_t(x,y)\right|>\tau^{bg}_t(x,y)
\right)
\]

với $\tau^{bg}_t(x,y)$ là **ngưỡng thích ứng** (định nghĩa ở §1.4).

### 1.2. Temporal-difference evidence

Với frame tham chiếu $I_{t-\Delta}$:

\[
M^{diff}_{t,\Delta}(x,y)
=
\mathbf{1}
\left(
\left|
\widetilde I_t(x,y)-\widetilde I_{t-\Delta}(x,y)
\right|
>
\tau^{diff}_{t,\Delta}
\right)
\]

Frame chuẩn hóa $\widetilde I_t$ bù trôi brightness toàn cục. **Hiệu chỉnh brightness robust**:

\[
\delta_t
=
\operatorname{median}_{x,y}
\left(
I_t(x,y)-I_{t-\Delta}(x,y)
\right)
\]

\[
\widetilde I_{t-\Delta}(x,y)
=
\operatorname{clip}
\left(
I_{t-\Delta}(x,y)+\delta_t,\;0,\;255
\right)
\]

### 1.3. Dual-stage foreground evidence (cổng AND)

\[
M_t(x,y)
=
M^{bg}_t(x,y)
\;\land\;
\left(
\bigvee_{\Delta\in\mathcal D}
M^{diff}_{t,\Delta}(x,y)
\right)
\]

với họ tham chiếu:

\[
\mathcal D
=
\{\Delta_{short},\;\Delta_{long},\;\Delta_{lag}\}
\]

- $\Delta_{short}$: bắt xe nhanh
- $\Delta_{long}$: tích lũy dịch chuyển cho xe chậm
- $\Delta_{lag}$: **chỉ dùng khi gap timestamp trong khoảng tối đa hợp lệ** — không nối qua stream pause (nối sẽ tạo full-frame flash)

**Ý nghĩa của cổng AND**: background subtraction một mình đánh dấu cả xe đứng yên khi exposure đổi; frame-difference một mình bỏ sót xe chậm. Cổng AND yêu cầu bằng chứng từ cả hai nguồn độc lập.

### 1.4. Adaptive threshold engine

\[
\tau_t
=
\tau_0
+
\alpha\,\sigma^{noise}_t
+
\beta\,\left|L_t-L_{t-1}\right|
+
\gamma\,E^{illumination}_t
\]

với:

| Thành phần | Ý nghĩa |
|---|---|
| $\tau_0$ | ngưỡng nền |
| $\sigma^{noise}_t$ | nhiễu thời gian cục bộ ước lượng |
| $L_t$ | luminance toàn cục |
| $E^{illumination}_t$ | điểm chuyển tiếp chiếu sáng đã phát hiện |
| $\alpha,\beta,\gamma$ | hệ số hiệu chuẩn |

Nhiễu cục bộ cho vùng $R$:

\[
\sigma^{noise}_t(R)
=
\operatorname{MAD}
\left(
I_t(R)-I_{t-1}(R)
\right)
\]

(MAD = median absolute deviation — robust với outlier, khác std bị kéo bởi chính chuyển động xe).

### 1.5. Shadow rejection

Vùng được classify là bóng chỉ khi **đồng thời**:

**Điều kiện 1 — attenuation trong dải bóng:**

\[
a_{min}
<
\frac{Y_t}{Y_B}
<
a_{max}
\]

($Y_t$, $Y_B$: luminance hiện tại / nền; dải điển hình $a_{min}\approx0.32$, $a_{max}\approx0.62$ — bóng làm tối có giới hạn, vật thật che hẳn)

**Điều kiện 2 — chromaticity ổn định:**

\[
\left\|
\frac{\mathbf{c}_t}{\|\mathbf{c}_t\|}
-
\frac{\mathbf{c}_B}{\|\mathbf{c}_B\|}
\right\|_1
<
\epsilon_c
\]

(bóng chỉ nhân luminance, không đổi sắc độ)

**Điều kiện 3 — texture nền vẫn nhìn thấy** qua vùng.

**Điều kiện 4 (fail-open)**: vùng bóng-like KHÔNG bị loại nếu detector vật hoặc biên độc lập hỗ trợ — chống dập nhầm xe tối màu.

### 1.6. Chứng minh ngưỡng tĩnh thất bại trong môi trường thật

Ngưỡng cố định $\tau$ giả định nhiễu dừng:

\[
P\left(|I_t-I_{t-1}|>\tau\right)
=
\text{constant}
\]

Giả định này sai khi: exposure camera đổi · đèn LED nhấp nháy · nén JPEG biến thiên · bóng di chuyển · gain camera đổi · xe rất nhỏ · xe gần đứng yên.

**Bất đẳng thức thất bại kép** — với cùng một $\tau$ cố định:

\[
\underbrace{\sigma^{noise}_{lit}(t)}_{\text{đèn nhấp nháy}} > \tau
\quad\Rightarrow\quad
FP \uparrow \;(\text{floor thành xe})
\]

\[
\underbrace{\Delta_{vehicle}(t)}_{\text{xe chậm, 12 px}} < \tau
\quad\Rightarrow\quad
FN \uparrow \;(\text{xe bị bỏ})
\]

Cùng một ngưỡng tạo FP ở chế độ nhiễu cao VÀ FN ở chế độ tín hiệu thấp — không có $\tau$ cố định nào thắng đồng thời cả hai trạng thái. Đây là lý do tồn tại của $\tau_t$ thích ứng theo §1.4.

**Pass**: $\tau$ tăng trong chuyển tiếp brightness toàn cục → dập whole-frame change, giữ biên xe cục bộ.

**Fail**: dịch 10-px luminance do đèn thành "xe", trong khi xe chậm dịch 12-px bị loại.

---

## 2. Single-Camera Kinematic Tracking

### 2.1. State vector

\[
\mathbf{x}_t
=
\begin{bmatrix}
X_t\\
Y_t\\
V^X_t\\
V^Y_t\\
W_t\\
H_t\\
R_t
\end{bmatrix}
\in\mathbb{R}^7
\]

| Thành phần | Ý nghĩa |
|---|---|
| $X_t, Y_t$ | vị trí (image hoặc world) |
| $V^X_t, V^Y_t$ | vận tốc |
| $W_t, H_t$ | kích thước bbox |
| $R_t = W_t / H_t$ | aspect ratio |

### 2.2. Timestamp-based transition

\[
\mathbf{x}_{t|t-1}
=
F(\Delta t)\,\mathbf{x}_{t-1},
\qquad
\Delta t = t_t - t_{t-1}
\]

**Model constant-velocity:**

\[
F_{CV}(\Delta t)
=
\begin{bmatrix}
1&0&\Delta t&0&0&0&0\\
0&1&0&\Delta t&0&0&0\\
0&0&1&0&0&0&0\\
0&0&0&1&0&0&0\\
0&0&0&0&1&0&0\\
0&0&0&0&0&1&0\\
0&0&0&0&0&0&1
\end{bmatrix}
\]

**Model constant-acceleration:**

\[
X_t
=
X_{t-1}
+
V^X_{t-1}\Delta t
+
\tfrac{1}{2}A^X_{t-1}\Delta t^2
\]

\[
Y_t
=
Y_{t-1}
+
V^Y_{t-1}\Delta t
+
\tfrac{1}{2}A^Y_{t-1}\Delta t^2
\]

**Virtual interpolation khi lag (VR1)**: khi $\Delta t > \Delta t_{thresh}$, chèn $n=\lceil\Delta t / h\rceil$ bước dự đoán nhỏ ($h\approx100$ ms) kèm damping vận tốc mỗi bước:

\[
\mathbf{v}^{(k+1)}
=
\lambda_d\,\mathbf{v}^{(k)},
\qquad
\lambda_d \approx 0.5
\]

Damping nghiêng prediction về hypothesis "vật đã dừng" — lag ở bãi xe xảy ra đúng lúc xe đang vào ô đỗ, giữ nguyên quán tính đầy đủ sẽ đẩy prediction bay khỏi measurement thật.

### 2.3. Process covariance (lag-aware)

\[
P_{t|t-1}
=
F\,P_{t-1}\,F^\top+Q(\Delta t)
\]

\[
Q(\Delta t)
=
q
\begin{bmatrix}
\frac{\Delta t^4}{4}&0&\frac{\Delta t^3}{2}&0\\
0&\frac{\Delta t^4}{4}&0&\frac{\Delta t^3}{2}\\
\frac{\Delta t^3}{2}&0&\Delta t^2&0\\
0&\frac{\Delta t^3}{2}&0&\Delta t^2
\end{bmatrix}
\]

(không gian con position-velocity). Covariance **nở theo $\Delta t$ thật** — gate association mở đúng tốc độ khi FPS tụt, không phải nhảy vọt theo đếm frame.

### 2.4. Measurement model

\[
\mathbf{z}_t
=
H\,\mathbf{x}_t+\mathbf{v}_t,
\qquad
H
=
\begin{bmatrix}
1&0&0&0&0&0&0\\
0&1&0&0&0&0&0
\end{bmatrix}
\]

Covariance measurement phụ thuộc chất lượng detection:

\[
R_t
=
R_0
\left(
1+
\lambda(1-c_t)
+
\mu\,o_t
+
\nu\,s_t
\right)
\]

| Thành phần | Ý nghĩa |
|---|---|
| $c_t$ | confidence detection |
| $o_t$ | điểm occlusion |
| $s_t$ | uncertainty seam / phối cảnh |

### 2.5. Missed-observation state (thời gian, không phải frame)

Bộ đếm missed: $m_t = m_{t-1}+1$; reset về 0 khi measurement hợp lệ trở lại. Quyết định lifecycle dùng thời gian trôi:

\[
T_{miss}=t_t-t_{last\_observed}
\]

**Pass**: 12 FPS, xe mất 3 frame = 250 ms → track giữ active/re-acquiring.

**Fail**: 1 frame missed retire track → detection kế tiếp nhận identity mới.

---

## 3. Planar Homography & Topological Transition Zones

### 3.1. Phép biến đổi homography

Tọa độ pixel $\mathbf{p}=[u,v,1]^\top$; tọa độ world-plane $\mathbf{q}=[X,Y,1]^\top$:

\[
\mathbf{q}
\sim
H\,\mathbf{p},
\qquad
H
=
\begin{bmatrix}
h_{11}&h_{12}&h_{13}\\
h_{21}&h_{22}&h_{23}\\
h_{31}&h_{32}&h_{33}
\end{bmatrix}
\]

tức là:

\[
X
=
\frac{h_{11}u+h_{12}v+h_{13}}
{h_{31}u+h_{32}v+h_{33}}
\qquad
Y
=
\frac{h_{21}u+h_{22}v+h_{23}}
{h_{31}u+h_{32}v+h_{33}}
\]

### 3.2. Lan truyền uncertainty

Với covariance pixel $\Sigma_p$:

\[
\Sigma_w
=
J_H\,\Sigma_p\,J_H^\top
+
\Sigma_{calib}
+
\Sigma_{parallax}
\]

với $J_H$ = Jacobian của phép chiếu homography tại điểm đo (viết tường minh, không xấp xỉ hữu hạn). Gần seam:

\[
\Sigma_{seam}
=
\Sigma_w+\rho_{seam}^2\,I
\]

$\rho_{seam}$ **phải đo thực nghiệm** bằng quan sát đồng thời cùng vật ở 2 camera (chính xe ở đúng chiều cao xe — vật tiêu nhân tạo từng chiều cao là thay thế kém hơn).

**Cảnh báo hiệu chuẩn 4-điểm**: homography 8-DOF với đúng 4 cặp điểm cho residual $\equiv 0$ (nghiệm chính xác, overfit) — residual KHÔNG phản ánh sai số thật. Kiểm định phải dùng $n>4$ điểm và báo cáo bậc tự do dư.

### 3.3. Vùng overlap

\[
\Omega_{12}
=
\text{vùng world quan sát được bởi cả 2 camera}
\]

Detection chỉ được xét association cross-camera khi:

\[
\mathbf{w}\in\Omega_{12}^{expanded}
\]

với mức mở rộng do **uncertainty** quyết định, không phải bán kính toàn cục tùy ý.

### 3.4. Exit/entry topology

Định nghĩa:
- $E_1$: exit polygon của Camera 1
- $A_2$: entry polygon của Camera 2
- $G_{12}$: cung có hướng C1→C2 trong đồ topology $\mathcal{G}$

Handoff ứng viên $i\rightarrow j$ **topology-valid** chỉ khi:

\[
\mathbf{w}_i(t_i)\in E_1
\quad\land\quad
\mathbf{w}_j(t_j)\in A_2
\quad\land\quad
G_{12}\in\mathcal{G}
\]

**Ràng buộc thời gian:**

\[
\Delta t_{min}
\le
\Delta t=t_j-t_i
\le
\Delta t_{max}
\]

**Ràng buộc displacement:**

\[
\left\|
\mathbf{w}_j-\mathbf{w}_i
\right\|
\le
v_{max}\,\Delta t+\rho_{seam}
\]

### 3.5. Cấm tìm kiếm toàn cục mù quáng

Camera 2 chỉ được xét tập ứng viên:

\[
\mathcal C_2
=
\left\{
g:\;
lastCamera(g)=1,\;
\mathbf{w}_{last}(g)\in E_1,\;
G_{12}\text{ valid},\;
\Delta t\text{ feasible}
\right\}
\]

**Pass**: xe thoát C1 qua exit polygon hiệu chuẩn, vào C2 qua entry polygon tương ứng → identity đủ điều kiện handoff.

**Fail**: xe xuất hiện vùng C2 không liên quan nhưng nhận identity C1 vì màu giống.

---

## 4. Multi-Camera Cost Association Matrix (Cross-Camera Re-ID)

Với source identity $i$ và target observation $j$:

\[
C_{ij}
=
w_d\,C_{distance}
+
w_\theta\,C_{direction}
+
w_g\,C_{geometry}
+
w_a\,C_{appearance}
+
C_{topology}
+
C_{time}
\]

Ma trận được giải dưới **ràng buộc one-to-one** (assignment toàn cục, không greedy tuần tự).

### 4.1. Spatial cost (Mahalanobis bình phương)

\[
\mathbf{r}_{ij}
=
\mathbf{w}_j-
\widehat{\mathbf{w}}_i(t_j)
\qquad
S_{ij}
=
P_i(t_j)+R_j
\]

\[
C_{distance}
=
\mathbf{r}_{ij}^{\top}
S_{ij}^{-1}
\mathbf{r}_{ij}
\]

$P_i(t_j)$ = covariance prediction của identity $i$ tại thời điểm $j$ (đã nở theo $\Delta t$). Gate hợp lệ: $C_{distance}<\chi^2(2;0.99)=9.21$.

### 4.2. Direction cost

Vận tốc source/target: $\mathbf{v}_i,\mathbf{v}_j$:

\[
\cos\theta_{ij}
=
\frac{\mathbf{v}_i^\top\mathbf{v}_j}
{\|\mathbf{v}_i\|\,\|\mathbf{v}_j\|}
\qquad
C_{direction}
=
\frac{1-\cos\theta_{ij}}{2}
\in[0,1]
\]

Khi vector không đáng tin (occlusion / gap dài), direction **phải được giảm trọng số** thay vì làm hard-reject — nếu không, xe quay đầu hợp lệ (hướng $\cos\theta\approx-1$) sẽ bị chặn dù mọi bằng chứng khác hoàn hảo.

### 4.3. Geometry cost

\[
C_{geometry}
=
\left|
\log\frac{A_j}{\widehat A_i}
\right|
+
\eta
\left|
\log\frac{r_j}{\widehat r_i}
\right|
\]

($A$: diện tích footprint; $r$: aspect ratio; mũ $\widehat{}$: giá trị dự đoán của source)

### 4.4. Appearance cost

Embedding chuẩn hóa $\mathbf{e}_i,\mathbf{e}_j$:

\[
C_{appearance}
=
1-
\frac{\mathbf{e}_i^\top\mathbf{e}_j}
{\|\mathbf{e}_i\|\,\|\mathbf{e}_j\|}
\]

Với gallery nhiều mẫu:

\[
C_{appearance}(g,j)
=
\min_{k\in Gallery(g)}
d(\mathbf{e}_k,\mathbf{e}_j)
\]

Bổ sung thống kê nhất quán để một mẫu tình cờ không độc chiếm:

\[
C^{robust}_{appearance}
=
\alpha\,d_{min}
+
(1-\alpha)\,\operatorname{median}(d_{nearest})
\]

### 4.5. Topology cost

\[
C_{topology}
=
\begin{cases}
0,& \text{handoff có hướng hợp lệ}\\
+\infty,& \text{handoff không hợp lệ}
\end{cases}
\]

Ứng viên topology sai **không phải chỉ đắt — phải bị loại khỏi ma trận** trước khi giải.

### 4.6. Time cost

Dạng penalty khoảng thời gian giới hạn:

\[
C_{time}
=
\begin{cases}
0,& \Delta t_{min}\le\Delta t\le\Delta t_{max}\\
+\infty,& \text{otherwise}
\end{cases}
\]

hoặc dạng log-mềm:

\[
C_{time}
=
\left|
\log
\frac{\Delta t}{\widehat{\Delta t}}
\right|
\]

### 4.7. Assignment margin

Selected cost $C_1$; best competing feasible $C_2$:

\[
M=C_2-C_1
\qquad
\text{chấp nhận chỉ khi}\quad
M\ge M_{min}
\]

Nếu $M<M_{min}$: target **defer** (giữ Global ID cũ trong grace period) — không ép chọn tùy tiện.

### 4.8. Catastrophic failure khi $w_d$ quá cao

Giả sử:

\[
\begin{aligned}
\text{Xe A (source)} &: (100,100)\\
\text{Xe B (source)} &: (100,130)\\
\text{lỗi chiếu camera transition} &: 25\text{ units}\\
\text{Target A} &: (125,100)\\
\text{Target B} &: (100,130)
\end{aligned}
\]

Nếu $w_d$ áp đảo mọi hạng tử khác, hệ thống gán source A → target B vì target B *gần hơn* dưới phép chiếu méo:

\[
d(A_{src},B_{tgt})=30
<
d(A_{src},A_{tgt})=25+\rho_{seam}
\quad(\text{với } \rho_{seam}\text{ lớn})
\]

Kết quả là **ID switch kép**:

\[
\text{Xe A}\to\text{identity của B},\qquad
\text{Xe B}\to\text{identity của A}
\]

dù appearance và direction đều bác bỏ cặp này. Bài học: khoảng cách là tín hiệu *điều kiện* (có covariance mới đáng tin), không phải chân lý tuyệt đối.

### 4.9. Catastrophic failure khi bỏ $w_\theta$

Hai xe gần nhau, ngược chiều:

\[
\mathbf{v}_A=(1,0)
\qquad
\mathbf{v}_B=(-1,0)
\]

Khoảng cách không gian nhỏ; appearance có thể giống (cùng mẫu xe). Không có direction cost, identity sai được chọn trong sự kiện giao nhau/overlap — đặc biệt chết người ở seam vì đây là nơi hai xe qua lại nhiều nhất.

Direction KHÔNG được là gate duy nhất (xe quay đầu hợp lệ), nhưng **loại hẳn nó tạo failure đối xứng đã biết**: hệ association phải chứa cả $C_{direction}$ (đúng trọng số) và appearance làm kênh xác nhận chéo.

---

## 5. Slot Occupancy State Engine (Dynamic Equalizer & Temporal Confirmation)

### 5.1. Overlap hình học

Footprint xe (world) = polygon $V_t$; slot = polygon $S_k$:

\[
IoU(V_t,S_k)
=
\frac{|V_t\cap S_k|}
{|V_t\cup S_k|}
\qquad
Coverage(V_t,S_k)
=
\frac{|V_t\cap S_k|}{|V_t|}
\]

Xe là ứng viên slot $k$ khi CẢ HAI:

\[
Coverage(V_t,S_k)\ge \tau_{coverage}
\qquad\land\qquad
IoU(V_t,S_k)\ge \tau_{IoU}
\]

(Coverage chống trường hợp bbox lớn đè nhiều slot; IoU chống bbox nhỏ nằm trọn trong slot nhưng xe thật chỉ chạm mép.)

### 5.2. Centroid condition

\[
D_{center}
=
\left\|c(V_t)-c(S_k)\right\|
\qquad
\text{centered khi}\quad
D_{center}\le\tau_{center}
\]

Centroid một mình KHÔNG đủ — xe overlap hai slot kề nhau có tâm nằm gần biên chung. Centroid là bằng chứng *hỗ trợ*, không phải điều kiện duy nhất.

### 5.3. Inward-motion condition

$d_{outside}$ = khoảng cách từ vị trí ứng viên hợp lệ đầu tiên tới tâm slot; $d_t$ = khoảng cách hiện tại:

\[
\Delta d
=
d_{outside}-d_t
\qquad
\text{arrival hướng trong khi}\quad
\Delta d\ge\tau_{inward}
\]

— phân biệt xe **đang vào** slot với xe chỉ **đi ngang qua**.

### 5.4. Temporal sliding window

Cửa sổ cho mỗi slot candidate:

\[
W_k(t)
=
\left\{
o_{t-n+1},\ldots,o_t
\right\},
\qquad
o_t=
\big(
IoU_t,\;Coverage_t,\;D_{center,t},\;\Delta d_t,\;v_t,\;q_t
\big)
\]

**Arrival confirm khi CẢ NĂM điều kiện:**

\[
\sum_{o_i\in W_k}
\mathbf{1}(IoU_i\ge\tau_{IoU})
\ge N_{IoU}
\]

\[
\sum_{o_i\in W_k}
\mathbf{1}(Coverage_i\ge\tau_{coverage})
\ge N_{coverage}
\]

\[
\max_{o_i\in W_k}\Delta d_i
\ge\tau_{inward}
\]

\[
\operatorname{Var}\left(c(V_i)\right)\le\sigma^2_{stable}
\qquad(\text{vị trí ổn định})
\]

\[
v_t \le v_{parked} \;(\text{trong phần cuối cửa sổ})
\]

### 5.5. Hysteresis (Dynamic Equalizer)

Hai ngưỡng tách biệt:

\[
\tau_{enter}>\tau_{release}
\]

- Slot thành occupied: chỉ khi $\tau_{enter}$ đạt (cộng temporal window §5.4).
- Slot rời occupied: chỉ khi bằng chứng rơi **dưới $\tau_{release}$** trong thời gian release duration.

Chống dao động:

\[
\text{occupied}\to\text{empty}\to\text{occupied}\to\text{empty}
\]

do một frame nhiễu.

### 5.6. Ví dụ số Pass/Fail

**Slot D08, xe P01, 5 frame:**

```text
Frame 101: IoU = 0.22, coverage = 0.35
Frame 102: IoU = 0.48, coverage = 0.71
Frame 103: IoU = 0.61, coverage = 0.83
Frame 104: IoU = 0.64, coverage = 0.86
Frame 105: IoU = 0.63, coverage = 0.85
```

Ngưỡng: $\tau_{IoU}=0.50$, $\tau_{coverage}=0.75$, $N=3$.

Đếm: IoU đạt ở frame 103-105 (3 frame ✓); Coverage đạt ở 103-105 (3 frame ✓) → **PASS — slot D08 assigned GID của P01**.

**Fail case**: một frame duy nhất `IoU = 0.95` (bbox lỏng) rồi không còn bằng chứng → **FAIL** — yêu cầu temporal chưa thỏa, không confirm parking.

### 5.7. Quản lý xe đỗ sau khi motion = 0

Xe đỗ xong → motion = 0 → frame-difference mất dấu → tracking có thể mất observation. Engine phải:

1. Giữ ownership qua slot evidence (vision occupancy + presence watchdog).
2. KHÔNG giải phóng slot vì track chuyển `temporarily_missing`.
3. Chỉ release khi bằng chứng departure (ra khỏi $\tau_{release}$ + outward motion + duration).

**Pass**: D08 giữ GID 17 toàn thời gian đỗ; vision flicker 2-3 frame không đổi ownership.

**Fail**: D08 thành empty sau 1 false-negative frame; xe kế nhận sai GID 17.

---

## 6. Identity Creation, Retention & Re-Identification Logic

### 6.1. Identity score tổng hợp

Với identity $g$ và observation $o$:

\[
S(g,o)
=
w_p\,S_p
+
w_a\,S_a
+
w_t\,S_t
+
w_z\,S_z
+
w_c\,S_c
\]

| Hạng | Bằng chứng |
|---|---|
| $S_p$ | position likelihood (Mahalanobis từ §4.1) |
| $S_a$ | appearance similarity (§4.4) |
| $S_t$ | temporal continuity (§4.6) |
| $S_z$ | topology validity (§4.5) |
| $S_c$ | geometric consistency (§4.3) |

### 6.2. Điều kiện Re-ID chấp nhận

Observation tái dùng identity $g$ khi:

\[
S(g,o)\ge\tau_{accept}
\qquad\text{VÀ}\qquad
S(g,o)-S(g_2,o)\ge\tau_{margin}
\]

($g_2$ = identity dự nhì — margin chống cướp ID khi hai xe gần giống nhau.)

### 6.3. Cửa sổ cấm tạo ID mới (anti-fragmentation)

ID Global mới KHÔNG thể được tạo khi:

\[
\exists g:\;
T_{missing}(g)<T_{grace}
\;\land\;
S(g,o)\ge\tau_{candidate}
\]

— đây là **quy tắc trung tâm chống phân mảnh danh tính**: một xe vừa mất dấu trong khoảng grace còn hypothesis hợp lệ thì observation mới phải gán lại cho nó, không mint ID mới.

### 6.4. Identity retention

Identity giữ khả năng khôi phục khi ÍT NHẤT MỘT điều đúng:

\[
T_{missing}<T_{max}
\quad\lor\quad
\text{slot ownership hợp lệ}
\quad\lor\quad
\text{handoff pending hợp lệ}
\quad\lor\quad
\text{appearance gallery match còn khả thi}
\]

### 6.5. Quy tắc chống va chạm danh tính

Tại mọi thời điểm $t$:

\[
\forall g,\quad
\left|
\left\{
o:\;
o.global\_id=g,\;
o\text{ vật lý tách biệt tại }t
\right\}
\right|
\le 1
\]

Vi phạm → **quarantine** các quan sát (không merge tự động) cho đến khi bằng chứng phân giải. Merge hai xe thành một ID là lỗi nghiêm trọng hơn tạm thời hai ID cho một xe.

---

## 7. Occlusion Group — chính sách tường minh

Khi một detection phủ nhiều track:

1. Cả hai track chuyển `occluded/merged`, **coast theo prediction**.
2. **Đóng băng appearance update** — blob hợp nhất là hỗn hợp 2 xe, đưa vào gallery sẽ nhiễm cả hai danh tính.
3. Khi tách: giải bằng joint-assignment có margin; margin không đủ → coast tiếp (KHÔNG mint ID mới).
4. ID mới chỉ sinh khi fragment trưởng thành + hết grace + không track nào còn hypothesis.

**Pass**: frame 500 hai xe riêng; 501-502 một merged detection; 503 tách — cả 2 ID gốc được khôi phục đúng.

**Fail**: 1 trong 2 ID bị xóa trong giai đoạn merge; hoặc ID thứ 3 xuất hiện ngay khi tách.

---

## 8. Bảng tham chiếu ký hiệu

| Ký hiệu | Định nghĩa |
|---|---|
| $I_t, B_t$ | frame hiện tại / nền ước lượng |
| $M_t$ | foreground mask hợp nhất |
| $\tau_t$ | ngưỡng thích ứng |
| $\delta_t$ | dịch brightness toàn cục (median) |
| $\mathbf{x}_t \in \mathbb{R}^7$ | state vector local track |
| $F(\Delta t), Q(\Delta t)$ | ma trận transition / covariance nhiễu quá trình |
| $H$ (context homography) | phép chiếu pixel→world |
| $J_H$ | Jacobian homography |
| $\Sigma_w, \rho_{seam}$ | covariance world / seam parallax budget |
| $E_1, A_2, \mathcal{G}$ | exit polygon / entry polygon / đồ topology |
| $C_{ij}$ | chi phí association tổng hợp |
| $M$ (context margin) | khoảng cách best − competing |
| $IoU, Coverage$ | chỉ số overlap footprint/slot |
| $S(g,o)$ | điểm identity tổng hợp |
