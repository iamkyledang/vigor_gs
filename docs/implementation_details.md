# VIGOR-GS: chi tiết triển khai

Tài liệu này mô tả code VIGOR-GS trên nhánh `viet_huynh`: mỗi file làm gì, công thức nào được cài ra sao và vì sao lại chọn cách đó. Kết quả chạy thử, các quyết định còn chờ nhóm xem và các vấn đề đang mở nằm ở [implementation_notes.md](implementation_notes.md). Tài liệu đó không được chép lại ở đây.

Đặc tả gốc là [VIGOR_GS_Academic_Framework (2).pdf](VIGOR_GS_Academic_Framework%20(2).pdf) và `reference.md`. Chỗ nào hai tài liệu mâu thuẫn, code theo `reference.md`.

---

## Mục lục

1. [Bức tranh chung](#1-bức-tranh-chung)
2. [Quy ước dùng xuyên suốt](#2-quy-ước-dùng-xuyên-suốt)
3. [Dữ liệu: reader, camera, màu](#3-dữ-liệu-reader-camera-màu)
4. [Tham số dòng lệnh](#4-tham-số-dòng-lệnh)
5. [Deformation MLP](#5-deformation-mlp)
6. [Lớp nền `GaussianField`](#6-lớp-nền-gaussianfield)
7. [Interface field `G_I` và renderer hai pass](#7-interface-field-g_i-và-renderer-hai-pass)
8. [Phase I: `train_interface.py`](#8-phase-i-train_interfacepy)
9. [Trích xuất glass plane: `extract_plane.py`](#9-trích-xuất-glass-plane-extract_planepy)
10. [Camera ảo](#10-camera-ảo)
11. [Reflection field `G_R` và rasterizer 3DGS](#11-reflection-field-g_r-và-rasterizer-3dgs)
12. [Trọng số Fresnel và phép ghép ảnh](#12-trọng-số-fresnel-và-phép-ghép-ảnh)
13. [Phase II: `train_reflection.py`](#13-phase-ii-train_reflectionpy)
14. [Inference: `render_vigor.py`](#14-inference-render_vigorpy)
15. [Lưu và nạp kết quả](#15-lưu-và-nạp-kết-quả)
16. [Các ablation được cài thế nào](#16-các-ablation-được-cài-thế-nào)
17. [Gradient đi đâu trong từng phase](#17-gradient-đi-đâu-trong-từng-phase)
18. [Bộ test kiểm chứng những gì](#18-bộ-test-kiểm-chứng-những-gì)

---

## 1. Bức tranh chung

Một scene chạy qua bốn bước, mỗi bước là một script. Bước sau chỉ đọc kết quả bước trước từ đĩa:

```
                ảnh clean I_clean, mask M, thời gian t, pose COLMAP
                                   │
  train_interface.py   ──────────► │  Phase I: G_I (2DGS) + D_φI
                                   ▼
                     <model>/interface/   (ply, deform.pth, cfg_args)
                                   │
  extract_plane.py     ──────────► │  RANSAC + SVD trên điểm của mặt kính
                                   ▼
                     <model>/glass_plane.json   (n, d)
                                   │
  train_reflection.py  ──────────► │  Phase II: G_R (3DGS) + D_φR (+ h_ψ)
                                   │  G_I, D_φI và plane bị đóng băng
                                   ▼
                     <model>/reflection/  (ply, deform.pth, blend.pth, cfg_args)
                                   │
  render_vigor.py      ──────────► │  B, R, α, Î cho camera train/test
                                   ▼
                     <model>/{train,test}/ours_<iter>/   →   metrics.py
```

Forward pass đầy đủ cho một camera `C_t` tại thời điểm `t`, đúng thứ tự code chạy:

```
offsets_I = D_φI(μ_I, t)                                    scene/deform_model.py
B_t, glass_map_t = R_2D(G_I + offsets_I, C_t)               gaussian_renderer/interface.py
C_t^v = V(C_t, Π)                                           utils/virtual_camera.py
offsets_R = D_φR(μ_R, t)
R_t = R_3D(G_R + offsets_R, C_t^v)                          gaussian_renderer/reflection.py
α_t = clip(F(θ_t) + h_ψ) · glass_map_t                      utils/fresnel.py
Î_t = (1 − α_t) · B_t + α_t · R_t                           utils/fresnel.py
```

`gaussian_renderer/vigor.py` gói hai dòng đầu thành `render_direct` và các dòng còn lại thành `render_composite`. Script train Phase II, script render và các test đều gọi qua hai hàm này, nên forward pass chỉ được viết ở một chỗ.

**Baseline RT-Splatting không bị đụng tới.** `train.py`, `render.py`, `scene/gaussian_model.py` và `gaussian_renderer/__init__.py` giữ nguyên. Code VIGOR-GS nằm trong file mới. Vài file dùng chung (reader, camera, arguments) chỉ được thêm tham số có giá trị mặc định, nên baseline vẫn chạy như cũ.

---

## 2. Quy ước dùng xuyên suốt

| Quy ước | Nội dung | Ở đâu |
|---|---|---|
| **View matrix lưu dạng chuyển vị** | Codebase (theo 3DGS gốc) dùng vector hàng: `x_cam = x_world @ world_view_transform`, tức `world_view_transform = W2Cᵀ`. Mọi phép nhân ma trận camera phải tính đến chuyện này. | `scene/cameras.py`, `utils/virtual_camera.py` |
| **Glass plane** | `Π: nᵀx + d = 0` với `‖n‖ = 1`. Pháp tuyến luôn quay về phía camera: `nᵀc + d > 0` với camera thật. | `utils/plane_utils.py` |
| **`G_R` ở toạ độ thế giới thật** | Vật được phản chiếu nằm ở phía camera của tấm kính và lưu đúng ở vị trí thật. Gaussian không bao giờ bị lật. Chỉ camera bị lật. | `scene/reflection_model.py` |
| **Linear RGB** | Ảnh đích được đổi sRGB → linear lúc load. Phép ghép `(1−α)B + αR` tính trong không gian linear. Ảnh chỉ đổi lại sRGB khi lưu ra đĩa. | `utils/camera_utils.py`, `render_vigor.py` |
| **Thời gian** | `t ∈ [0, 1]`, đọc từ `frames.json`, gắn vào `Camera.time`. `p`, `o` và màu không phụ thuộc thời gian. Chỉ hình học (`μ, q, s`) bị deform. | `scene/cameras.py`, `scene/deform_model.py` |
| **Thư mục theo phase** | Mỗi phase có `cfg_args`, `point_cloud/`, `deform/` riêng trong `<model>/interface/` và `<model>/reflection/`. | `utils/vigor_utils.py` |
| **Nền** | `B_t` render trên nền đen (hoặc trắng nếu `--white_background`). `R_t` luôn render trên nền đen: chỗ nào không có Gaussian thì không có phản chiếu. | `gaussian_renderer/vigor.py` |

---

## 3. Dữ liệu: reader, camera, màu

### 3.1. `readVigorSceneInfo` ([scene/dataset_readers.py](../scene/dataset_readers.py))

`Scene` ([scene/__init__.py](../scene/__init__.py)) chọn reader này khi thấy file `colmap/frames.json`. Điều kiện này được kiểm tra trước nhánh COLMAP thường, vì scene VIGOR cũng có thư mục `sparse`, chỉ là nằm sâu hơn một cấp.

Reader làm theo thứ tự sau:

1. Đọc `frames.json` (thành dict theo tên ảnh) và `split.json`.
2. Đọc `images.bin` và `cameras.bin` trong `colmap/sparse/0/`.
3. Với mỗi ảnh trong model COLMAP:
   - **Bỏ qua ảnh không có trong `split.json`.** Model COLMAP của Wayve chứa ảnh của mọi camera trên rig (khoảng 1000 ảnh), trong khi scene chỉ dùng một phần.
   - Lấy FoV từ intrinsics. Chỉ nhận `PINHOLE` và `SIMPLE_PINHOLE`; model khác báo lỗi ngay thay vì cho ra hình học sai mà không ai hay.
   - Tìm mask ở `colmap/masks/<stem>.png`, nếu không có thì thử đường dẫn `source_mask` trong `frames.json`. Không có mask nào thì báo lỗi. Mask lệch kích thước với ảnh thì được đếm lại, cuối cùng in một cảnh báo.
   - Tìm ảnh clean theo stem trong `<scene>/<clean_images>/` với các đuôi `.png/.jpg/.jpeg`. Thiếu thì ghi tên vào danh sách `missing_clean`.
4. Sau vòng lặp: nếu còn ảnh thiếu bản clean mà không bật `--clean_fallback_original` thì báo lỗi, kèm số ảnh thiếu và tên một ảnh làm ví dụ. Bật cờ thì chỉ in cảnh báo.
5. Chia train/test theo `split.json` (thay cho `llffhold` của reader COLMAP). Không có `--eval` thì mọi ảnh đều vào train.
6. Point cloud khởi tạo lấy từ `points3D.bin`, chuyển sang `.ply` ở lần đọc đầu tiên.

`--ignore_clean_images` bỏ qua thư mục clean hoàn toàn và không tính ảnh nào là thiếu. Đây là ablation "một field trên ảnh gốc" (mục 16).

`CameraInfo` có thêm hai trường có mặc định: `time: float = 0.0` và `clean_image: object = None`. Reader COLMAP và Blender của baseline không truyền hai trường này, nên vẫn chạy.

### 3.2. `Camera` và `loadCam` ([scene/cameras.py](../scene/cameras.py), [utils/camera_utils.py](../utils/camera_utils.py))

`loadCam` resize ảnh clean cùng độ phân giải với ảnh gốc. Nếu `args.linear_rgb` bật, nó đổi cả ảnh gốc lẫn ảnh clean sang linear; kênh alpha (nếu có) giữ nguyên.

`Camera` có thêm:

- `time`: thời điểm chuẩn hoá của frame.
- `has_clean_image`: `True` khi có ảnh clean thật.
- `clean_image`: ảnh clean, hoặc **chính ảnh gốc** khi không có bản clean. Nhờ vậy code Phase I luôn đọc `cam.clean_image` mà không phải rẽ nhánh. `train_interface.py` in cảnh báo nếu không camera nào có ảnh clean thật.

### 3.3. Sửa `srgb2linear` ([utils/color_utils.py](../utils/color_utils.py))

Công thức sRGB → linear cho đoạn phi tuyến là `((x + 0.055) / 1.055)^2.4`. Code gốc thiếu dấu ngoặc nên tính `(x + 0.055/1.055)^2.4`, cho giá trị lớn hơn 1 khi `x` gần 1. Baseline không gọi hàm này nên lỗi chưa từng lộ ra; test `test_srgb_linear_conversions_are_inverse` phát hiện nó. Code mới chỉ thêm dấu ngoặc.

---

## 4. Tham số dòng lệnh

Mọi thứ nằm trong [arguments/__init__.py](../arguments/__init__.py).

### 4.1. `ModelParams`: các trường mới

Các trường này được lưu vào `cfg_args` và đi theo model qua mọi bước sau:

| Nhóm | Trường | Mặc định | Ý nghĩa |
|---|---|---|---|
| Dữ liệu | `clean_images` | `clean` | Thư mục ảnh clean (tuyệt đối, hoặc tương đối so với scene) |
| | `clean_fallback_original` | `False` | Thiếu ảnh clean thì dùng ảnh gốc |
| | `linear_rgb` | `False` | Đổi ảnh sang linear khi load. `train_interface.py` tự đặt thành `True` trừ khi có `--srgb` |
| Deformation | `deform_depth`, `deform_width` | 8, 256 | Kích thước MLP (giống Deformable-3DGS) |
| | `deform_xyz_multires`, `deform_time_multires` | 10, 6 | Số tần số positional encoding |
| Fresnel | `eta_air`, `eta_glass` | 1.0, 1.5 | Chiết suất, cho ra `F0 = 0.04` |
| | `optical_correction` | `False` | Học thêm `h_ψ` |
| Ablation | `ignore_clean_images`, `ablate_factorization`, `ablate_virtual_camera`, `ablate_fresnel`, `shared_reflection_field` | `False` | Xem mục 16 |

Ablation được đặt trong `ModelParams` chứ không trong tham số tối ưu. Lý do: chúng thay đổi chính model chứ không chỉ cách train, nên phải được lưu vào `cfg_args` để `render_vigor.py` dựng lại đúng biến thể.

### 4.2. `InterfaceOptimizationParams` và `ReflectionOptimizationParams`

Hai lớp riêng cho hai phase. Learning rate và lịch densify lấy theo 3DGS/2DGS. Các trọng số loss là giá trị đặt tạm (danh sách cần xem lại ở mục 3.2 của `implementation_notes.md`).

Phase I:

| Tham số | Mặc định | Vai trò |
|---|---|---|
| `lambda_dssim` | 0.2 | Tỉ lệ D-SSIM trong `L_rgb`. Đặt 0 để có L1 thuần như PDF |
| `norm_loss_weight`, `norm_loss_from_iter` | 0.05, 700 | `L_surf`: độ nhất quán giữa normal render ra và normal suy từ depth (surface pass) |
| `dist_loss_weight` | 0.0 | `L_surf`: distortion của volume pass (tắt mặc định) |
| `mask_loss_weight` | 0.01 | BCE giữa glass map và mask `M_t` |
| `occupancy_decay_weight` | 0.001 | Kéo `p` của Gaussian nhìn thấy về 0, để bớt Gaussian thừa |
| `reg_from_iter` | 700 | Mask loss và occupancy decay chỉ bật từ đây, khi đã có cảnh thô |
| `occupancy_cull` | 0.05 | Prune Gaussian có `p` dưới ngưỡng này |
| `occupancy_reset_interval` | 3000 | Chu kỳ reset `p` về tối đa 0.01. `o` được reset lệch nửa chu kỳ |
| `warm_up` | 3000 | Số iteration train tĩnh trước khi bật deformation |
| `time_noise_scale`, `time_noise_until_iter` | 0.1, 20000 | Nhiễu thời gian (mục 5.3) |
| `lambda_deform_reg` | 0.0 | Phạt `mean|Δμ|` để giữ kính ổn định (tắt mặc định) |

Phase II có thêm:

| Tham số | Mặc định | Vai trò |
|---|---|---|
| `lambda_motion` | 0.01 | `L_motion = mean|Δμ_R|` |
| `lambda_opt`, `optical_lr` | 0.01, 0.001 | `L_opt` và learning rate cho tham số blend |
| `gt_glass_mask` | `False` | Dùng mask GT thay glass map render ra để giới hạn `α` khi train |
| `refl_init_points`, `refl_init_radius_scale`, `refl_init_color` | 100000, 3.0, 0.5 | Khởi tạo ngẫu nhiên `G_R` (mục 11.1) |
| `min_opacity`, `opacity_reset_interval` | 0.005, 3000 | Prune và reset theo opacity, như 3DGS |

### 4.3. Hai thay đổi trong cơ chế đọc cấu hình

- **`get_combined_args(parser, cfg_subdir='')`** có thêm `cfg_subdir`. Vì mỗi phase có `cfg_args` riêng, các script sau đọc `cfg_args` của phase trước: `extract_plane.py` và `train_reflection.py` đọc `interface/cfg_args`; `render_vigor.py` đọc `reflection/cfg_args`, hoặc `interface/cfg_args` khi chạy `--direct_only`. `cfg_args` của Phase II được ghi từ tham số đã gộp, nên chứa cả cấu hình Phase I lẫn các cờ ablation của Phase II.
- **`ParamGroup.extract` điền giá trị mặc định cho trường không có trong `cfg_args`.** Khi một tham số mới được thêm sau lúc model đã train, `cfg_args` cũ không nhắc đến nó, và trước đây việc này làm script sau crash với `AttributeError` (đã gặp với `eta_air`). Giờ trường thiếu lấy giá trị mặc định.

---

## 5. Deformation MLP

File: [scene/deform_model.py](../scene/deform_model.py). Chuyển thể từ Deformable 3D Gaussians (Yang et al., CVPR 2024). Một lớp duy nhất phục vụ cả hai field: `D_φI` với `scale_dim = 2` (đĩa 2DGS) và `D_φR` với `scale_dim = 3` (ellipsoid 3DGS).

### 5.1. Kiến trúc

```
γ(x) = [x, sin(2⁰x), cos(2⁰x), …, sin(2^(L−1)x), cos(2^(L−1)x)]       PositionalEncoding

input = [γ(μ), γ(t)]                        3·(1+2·10) + 1·(1+2·6) = 63 + 13 = 76 chiều
h = ReLU(Linear(input))
h = ReLU(Linear(h)) × (depth − 1)           ở lớp depth//2, nối lại input vào h (skip)
Δμ = Linear(h) ∈ ℝ³,  Δq = Linear(h) ∈ ℝ⁴,  Δs = Linear(h) ∈ ℝ^scale_dim
```

Các điểm khác với bản gốc và lý do:

- **Ba head đầu ra khởi tạo bằng 0** (cả weight lẫn bias). Mạng mới tạo vì vậy là phép đồng nhất: bật deformation sau warm-up không làm cảnh tĩnh đã học bị xáo trộn. Test `test_fresh_network_is_the_identity` kiểm tra điều này.
- **Vị trí skip tính theo `depth`**: `skips = [depth // 2]` nếu lớp đó không phải lớp cuối, ngược lại không có skip. Bản gốc cố định skip ở lớp 4 cho mạng 8 lớp. Với mạng nông (`depth ≤ 2`, dùng khi test trên GPU 4GB), skip rơi vào sau lớp cuối và gây lệch shape; test có tham số `test_any_depth_builds_a_consistent_network` bắt được lỗi này.

### 5.2. Áp offset: `apply_deformation`

```
μ' = μ + Δμ
q' = normalize(q + Δq)
s' = exp(log_s + Δs)
```

Offset scale được cộng trong không gian log, nên scale sau deform luôn dương. Hàm nhận `pc._scaling` (giá trị thô trước `exp`) chứ không nhận `get_scaling`.

### 5.3. `DeformModel.step(xyz, time, time_noise)`

- Tâm Gaussian được **`detach()` trước khi đưa vào mạng.** Gradient từ loss vẫn về `μ` qua phép cộng `μ + Δμ`, nhưng không đi ngược qua đầu vào của MLP. Bản gốc cũng làm vậy: nếu không, mạng có thể "dời" field canonical chỉ qua đường encoding, khiến canonical và deformation khó tách nhau. Test `test_training_reaches_every_layer_but_not_the_canonical_centers` kiểm tra rằng `xyz.grad` là `None`.
- `time` là một số; hàm tự broadcast thành tensor `(N, 1)`.
- `time_noise`: cộng nhiễu Gauss vào `t` ở mỗi lần gọi. `train_interface.time_noise_std` tính độ lệch chuẩn là `time_noise_scale / num_frames · max(0, 1 − iteration / time_noise_until_iter)`: một phần khoảng cách giữa hai frame, giảm tuyến tính về 0. Mục đích là làm trơn deformation theo thời gian trên dữ liệu thật, giống bản gốc. `num_frames` là số camera train, dùng làm xấp xỉ số frame.

### 5.4. Tối ưu và lưu

- Learning rate `position_lr_init · cameras_extent`, giảm theo hàm mũ đến `position_lr_final` sau `deform_lr_max_steps` bước. Trong vòng lặp train, optimizer của deformation chỉ `step()` khi đã qua warm-up, nhưng `zero_grad` được gọi ở mọi iteration.
- `freeze()` tắt gradient, chuyển sang `eval()` và bỏ optimizer. Phase II dùng hàm này cho `D_φI`.
- Trọng số lưu ở `<phase>/deform/iteration_N/deform.pth`. `load_weights(iteration=-1)` lấy iteration lớn nhất.
- `from_args` dùng `getattr` với giá trị mặc định, nên đọc được cả `cfg_args` cũ chưa có trường `deform_*`.

---

## 6. Lớp nền `GaussianField`

File: [scene/gaussian_field.py](../scene/gaussian_field.py). Hai field có rất nhiều phần giống nhau: lưu tham số, optimizer, các thao tác optimizer khi clone/split/prune, lưu/đọc PLY. `GaussianField` gom các phần này lại, viết lại từ `GaussianModel` của 3DGS/2DGS.

### 6.1. Cơ chế `PARAMS`

Mỗi lớp con khai báo một tuple tên tham số, ví dụ `('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')`. Quy ước:

- Tham số tên `name` được lưu ở `self._<name>`.
- Group trong optimizer cũng tên `name`.
- PLY dùng tên cột suy từ `name` (`scaling` → `scale_0, scale_1, …`; `rotation` → `rot_0..3`; tham số 1 chiều như `opacity` → `opacity`).

Nhờ vậy khởi tạo, `training_setup`, `prune_points`, `densify_and_clone`, `densify_and_split`, `save_ply`, `load_ply`, `capture`, `restore` và `freeze` đều chỉ cần lặp qua `PARAMS`. Thêm một tham số chỉ cần ba việc: thêm tên vào `PARAMS`, trả giá trị khởi tạo trong `_initial_extra_params` và learning rate trong `_learning_rates`. `InterfaceGaussianModel` thêm `occupancy` theo đúng cách này.

### 6.2. Các hằng số lớp con ghi đè

| Hằng số | `GaussianField` | Interface | Reflection |
|---|---|---|---|
| `SCALE_DIM` | 3 | 2 | 3 |
| `INIT_OPACITY` | 0.1 | 0.5 | 0.1 |
| `INIT_SCALE_FACTOR` | 1.0 | 0.1 | 1.0 |

Scale khởi tạo là `log(sqrt(dist² · INIT_SCALE_FACTOR))`, với `dist²` là khoảng cách bình phương trung bình đến 3 điểm gần nhất (`distCUDA2`).

### 6.3. Densification

Logic clone/split giống 3DGS, với một khác biệt cho đĩa 2D: khi split, điểm mới được lấy mẫu theo `N(0, diag(s₁, s₂, 0))` trong hệ toạ độ của đĩa, nghĩa là **không lệch theo pháp tuyến**. Đĩa không có bề dày, nên điểm con phải nằm trên mặt phẳng của đĩa cha.

`densify(max_grad, extent)` chỉ làm clone + split. Prune là việc của lớp con, vì mỗi field prune theo một đại lượng khác nhau (occupancy hay opacity).

`_reset_activated(name, …, ceiling=0.01)` kẹp giá trị đã kích hoạt (sau sigmoid) xuống tối đa `ceiling`, rồi thay tensor trong optimizer và xoá moment Adam. Trước khi kẹp, nó đặt về 0 các hàng mà `xyz`, `scaling`, `rotation` hoặc chính tham số đó có NaN (giữ hành vi của RT-Splatting). `reset_opacity` và `reset_occupancy` đều gọi hàm này.

---

## 7. Interface field `G_I` và renderer hai pass

### 7.1. `InterfaceGaussianModel` ([scene/interface_model.py](../scene/interface_model.py))

Mỗi đĩa có `(μ, q, s⁽²⁾, p, o, SH)`. So với `GaussianModel` của RT-Splatting, lớp này bỏ transmissivity, reflectance, roughness, feature, `SphMipEncoding`, `light_mlp` và `env_scope`. Phần phản chiếu giờ là việc của `G_R`, nên Phase I không cần `nvdiffrast`.

- **Occupancy `p`**: đĩa có tồn tại về mặt hình học hay không. Kính có `p` cao.
- **Opacity `o`**: khi đã tồn tại, đĩa chắn sáng bao nhiêu. Kính có `o` thấp vì ánh sáng đi xuyên qua.
- `get_effective_opacity = p · o` là opacity của volume pass.
- Khởi tạo `p = 0.1`, `o = 0.5`, quaternion ngẫu nhiên (như RT-Splatting, để các đĩa không cùng một hướng ngay từ đầu).
- `densify_and_prune(max_grad, min_occupancy, extent, max_screen_size, last_reset_iter)`:
  1. clone + split;
  2. prune đĩa có `p < min_occupancy`; khi có `max_screen_size` (sau lần reset occupancy đầu tiên), prune thêm đĩa quá lớn trên màn hình (`max_radii2D > max_screen_size`) và đĩa quá lớn trong không gian (`max(s) > 0.1 · extent`);
  3. **không prune đĩa chưa được nhìn thấy lần nào kể từ lần reset occupancy gần nhất** (`last_update ≤ last_reset_iter`). Sau một lần reset, mọi `p` đều ≤ 0.01 < ngưỡng cull. Nếu không có điều kiện này, mọi đĩa chưa kịp được gradient kéo lên sẽ bị xoá sạch. RT-Splatting cũng làm vậy.
- `factorized=False` là ablation bỏ factorization (mục 16).

### 7.2. `render_interface` ([gaussian_renderer/interface.py](../gaussian_renderer/interface.py))

Cùng một tập đĩa (đã deform nếu có `offsets`) được rasterize **hai lần** bằng `diff_surfel_anych`:

| Pass | `opacities` | `extras` | Kết quả dùng |
|---|---|---|---|
| Volume | `p · o` | không có | `direct` = `B_t`: những gì nhìn thấy xuyên qua kính. Kèm alpha, depth, normal, distortion |
| Surface | `p` | `o` | Hình học của mặt đầu tiên, tính cả kính: `surface_alpha`, `surface_depth`, `surface_normal`. Kênh extra cho `surface_opacity` = `Σ wᵢ · oᵢ`, trung bình opacity của các đĩa trên mặt đầu tiên, có trọng số |

Từ đó:

```
glass_map = clip(1 − surface_opacity, 0, 1)
```

Ý nghĩa: ở đâu mặt đầu tiên tồn tại (`p` cao) nhưng cho ánh sáng đi qua (`o` thấp), ở đó là kính. Mask loss giám sát chính đại lượng này (mục 8.2). Lưu ý: chỗ không có đĩa nào thì `surface_opacity ≈ 0`, nên `glass_map ≈ 1`. Glass map vì vậy cũng cao ở vùng trống, không chỉ trên kính. Đây là một nguyên nhân khiến glass map "tràn" ra ngoài kính, đã ghi trong `implementation_notes.md`.

`unpack_allmap` tách buffer hình học của rasterizer: kênh 0 là depth·alpha, 1 là alpha, 2–4 là normal (trong hệ camera, được đổi sang hệ thế giới), 5 là median depth, 6 là distortion. Depth trả về là `(1 − depth_ratio) · expected + depth_ratio · median`, giống 2DGS. Normal suy từ depth (`depth_to_normal_sobel`) được nhân với alpha đã detach.

Hai pass dùng chung tensor `screenspace_points`, nên gradient 2D dùng để densify là tổng của cả hai pass.

---

## 8. Phase I: `train_interface.py`

File: [train_interface.py](../train_interface.py).

### 8.1. Vòng lặp

```
for iteration in 1..N:
    cập nhật lr của G_I và D_φI; cứ 1000 iteration tăng bậc SH một lần
    chọn ngẫu nhiên một camera train
    nếu iteration > warm_up: offsets = D_φI.step(μ, t, nhiễu thời gian)   ngược lại: offsets = None
    out = render_interface(cam, G_I, offsets)
    loss = Σ interface_losses(out, …)
    loss.backward()
    log, test, lưu
    densify / prune / reset (nếu iteration < densify_until_iter)
    optimizer.step() cho G_I; cho D_φI chỉ khi đã qua warm-up
```

Như baseline, iteration cuối không gọi `optimizer.step()`, vì model đã được lưu trước đó trong cùng iteration.

Lịch densify (khi `iteration < densify_until_iter`):

- Mỗi iteration: cộng dồn chuẩn gradient 2D và `max_radii2D` cho các đĩa nhìn thấy.
- Mỗi `densification_interval` iteration sau `densify_from_iter`: `densify_and_prune`. Giới hạn kích thước trên màn hình (20 px) chỉ bật sau lần reset occupancy đầu tiên.
- Mỗi `occupancy_reset_interval` iteration: `reset_occupancy()`, đồng thời ghi lại `last_reset_iter`.
- Lệch nửa chu kỳ so với đó: `reset_opacity()`.

### 8.2. Các thành phần loss: `interface_losses`

```
L_I = L_rgb + L_norm + L_dist + L_mask + L_occupancy + L_deform
```

| Thành phần | Công thức trong code | Khi nào bật |
|---|---|---|
| `rgb` | `(1 − λ)·L1(B, I_clean) + λ·(1 − SSIM(B, I_clean))` | luôn luôn |
| `norm` | `w · mean((1 − ⟨n_surface, n_from_depth⟩) · surface_alpha.detach())` | `iteration ≥ norm_loss_from_iter` |
| `dist` | `w · mean(volume_dist)` | trọng số > 0 (mặc định tắt) |
| `mask` | `w · BCE(glass_map, M_t)` | `iteration ≥ reg_from_iter` |
| `occupancy` | `w · mean(p)` trên các đĩa nhìn thấy | `iteration ≥ reg_from_iter`, chỉ khi factorized |
| `deform` | `w · mean|Δμ|` | có offsets và trọng số > 0 (mặc định tắt) |

Vì sao mask loss dùng `BCE(1 − surface_opacity, M_t)` (công thức của RT-Splatting) mà không dùng `M̂ = 1 − Π(1 − p·g)` trong `guildance.md`: công thức kia ép `p → 0` ở mọi chỗ ngoài vùng kính, và vì opacity hiệu dụng là `p·o`, điều đó cũng xoá luôn hình học của vật đục. Với công thức đang dùng, mask chỉ đẩy `o` xuống trong vùng kính và kéo `o` lên ngoài vùng kính, còn `p` vẫn tự do. Test `test_mask_loss_pushes_opacity_down_inside_the_glass_mask` kiểm tra đúng chiều gradient này.

### 8.3. Đánh giá trong lúc train

`evaluate_interface` tính trên tập test và 5 camera train:

- PSNR của `B_t` so với `I_clean`, sau khi kẹp về [0, 1]. Vì train trên linear RGB, đây là PSNR linear, khác PSNR sRGB của `metrics.py`.
- IoU giữa `glass_map > 0.5` và mask GT.

Kết quả được in ra, ghi vào TensorBoard và trả về dưới dạng `history` (test dùng giá trị này).

---

## 9. Trích xuất glass plane: `extract_plane.py`

Files: [extract_plane.py](../extract_plane.py), [utils/plane_utils.py](../utils/plane_utils.py).

### 9.1. Gom điểm: `collect_glass_points`

Với mỗi camera train, script render surface pass với `depth_ratio = 1` (median depth, để điểm nằm trên một mặt chứ không lơ lửng giữa kính và vật phía sau). Pixel được giữ khi thoả cả bốn điều kiện:

```
M_t(u) = 1                  mask GT nói là kính
surface_alpha(u) > 0.5      có mặt phẳng rắn ở đó
glass_map(u) > 0.5          chính field cũng coi đó là kính
surface_depth(u) > 0
```

Điều kiện thứ ba loại những pixel tuy nằm trong mask nhưng mặt đầu tiên là vật đục, ví dụ khung cửa hay vật đặt sát kính. Test `test_collected_points_lie_on_the_glass_not_the_wall` dựng một bức tường đục phía sau tấm kính để kiểm tra việc loại này.

Pixel được giữ lại được chiếu ngược sang 3D bằng `depths_to_points`. Mỗi view lấy ngẫu nhiên tối đa `max_points_per_view` điểm (mặc định 2000, seed cố định) để view gần kính không lấn át các view khác.

### 9.2. Fit: `fit_plane_ransac`

1. **RANSAC theo lô**: mỗi lô 256 bộ ba điểm ngẫu nhiên. Pháp tuyến là `normalize((p₁ − p₀) × (p₂ − p₀))`. Bộ ba suy biến (thẳng hàng hoặc trùng điểm) bị loại. Số inlier của cả lô được tính bằng một phép nhân ma trận `points @ normals.T`, nhanh hơn lặp từng giả thuyết. Giữ lại giả thuyết có nhiều inlier nhất qua `iterations` (mặc định 2000) giả thuyết.
2. **Tinh chỉnh**: lặp 3 lần: fit SVD trên inlier (pháp tuyến là vector suy biến nhỏ nhất của các điểm đã trừ trọng tâm), rồi tính lại inlier.
3. Trả về `normal`, `offset`, mask inlier, `inlier_ratio` và RMSE trên inlier.

Ngưỡng inlier mặc định là `0.01 · cameras_extent`, tức 1% bán kính của cụm camera, để không phụ thuộc vào đơn vị của scene. Generator có seed cố định, nên chạy lại cho cùng một plane.

### 9.3. Định hướng và lưu

`orient_plane` đảo dấu `(n, d)` nếu trung bình khoảng cách có dấu từ camera đến plane âm. Sau bước này `n` luôn hướng về phía camera, và `ReflectionGaussianModel.create_random` dựa vào quy ước đó.

`glass_plane.json` lưu `normal`, `offset` và thông tin chẩn đoán: `source` (`ransac`/`manual`), iteration, số điểm, số view, ngưỡng, `inlier_ratio`, `rmse`, `cameras_in_front` (tỉ lệ camera ở phía dương) và `mean_camera_distance`. Script in cảnh báo khi `inlier_ratio < 0.5`: kính có thể không phẳng, có nhiều mặt, hoặc hình học chưa hội tụ.

`glass_points.ply` lưu các điểm đã gom, inlier màu xanh và outlier màu đỏ. Mở file này cạnh point cloud interface để xem bằng mắt plane có khớp không.

`--glass_plane NX NY NZ D` bỏ qua bước fit: plane nhập tay được chuẩn hoá về `‖n‖ = 1` và định hướng như trên.

---

## 10. Camera ảo

File: [utils/virtual_camera.py](../utils/virtual_camera.py).

### 10.1. Toán

Phép phản xạ qua `Π` ở toạ độ thuần nhất:

```
M = [ I − 2nnᵀ   −2d·n ]
    [    0         1   ]          M·[x; 1] = [x − 2(nᵀx + d)·n; 1]
```

`M` là phép đối hợp (`M·M = I`) và có định thức −1.

Một vật ở `X` (phía camera) được thấy qua kính tại ảnh gương `M·X`. Camera thật chiếu ảnh gương đó thành

```
P · W2C · (M·X) = P · (W2C·M) · X
```

Vậy render `G_R` (vẫn ở toạ độ thật) với view matrix `W2C·M` cho ra đúng ảnh phản chiếu mà camera thật nhìn thấy, **khớp từng pixel với `B_t`**, không cần lật ảnh và không cần lật Gaussian.

Vì codebase lưu `world_view_transform = W2Cᵀ`, view matrix mới là `(W2C·M)ᵀ = Mᵀ · W2Cᵀ`. Trong code đó là `mirror.T @ camera.world_view_transform`.

### 10.2. `reflect_camera(camera, normal, offset)`

Hàm trả về một `VirtualCamera` chỉ chứa những thuộc tính rasterizer cần đọc:

- `world_view_transform = Mᵀ @ world_view_transform` và `full_proj_transform = world_view_transform @ projection_matrix`;
- `camera_center = c − 2(nᵀc + d)·n`, tâm camera đã phản chiếu;
- kích thước ảnh, FoV, `znear`/`zfar`, `projection_matrix` và `time` giữ nguyên.

**Vì sao `camera_center` phải là tâm đã phản chiếu:** rasterizer dùng `campos` để tính hướng nhìn khi đánh giá màu SH, theo hướng `μ − campos`. Ánh sáng phản xạ đi từ `X` tới điểm `g` trên kính rồi tới `c`. Vì `c'` là ảnh gương của `c`, ba điểm `c'`, `g`, `X` thẳng hàng. Hướng `X − c'` vì vậy chính là hướng tia sáng rời `X` (đảo dấu), và SH của `G_R` được đánh giá đúng theo hướng vật thực sự phát sáng về phía kính.

**Định thức −1 không làm hỏng rasterizer:** 3DGS chiếu covariance thành `J·W·Σ·Wᵀ·Jᵀ`. Biểu thức này có `W` xuất hiện hai lần nên không đổi khi `W` đổi dấu định thức, và phép kiểm tra frustum chỉ xét độ sâu `z` của `M·X`, vốn nằm trước camera. Với rasterizer 2DGS (dùng trong ablation shared field), test `test_surfel_rasterizer_handles_the_mirrored_view` kiểm tra đĩa hiện đúng tại pixel của ảnh gương.

---

## 11. Reflection field `G_R` và rasterizer 3DGS

### 11.1. `ReflectionGaussianModel` ([scene/reflection_model.py](../scene/reflection_model.py))

Đây là 3DGS chuẩn `(μ, q, s⁽³⁾, o, SH)`, không chia sẻ tham số nào với `G_I`.

**Khởi tạo `create_random`:** cảnh phản chiếu không có điểm SfM (COLMAP chỉ thấy cảnh thật), nên `G_R` bắt đầu từ điểm ngẫu nhiên:

1. Lấy mẫu đều trong khối cầu bán kính `refl_init_radius_scale · cameras_extent`, tâm là trọng tâm các camera. Bán kính lấy mẫu theo `r·u^(1/3)` để phân bố đều theo thể tích.
2. Chỉ giữ điểm có `nᵀx + d > 0`, tức ở phía camera của tấm kính, nơi vật phản chiếu có thể tồn tại.
3. Lặp lại cho đến khi đủ `num_points` điểm. Mọi điểm có màu xám `refl_init_color`, opacity 0.1, scale theo khoảng cách đến láng giềng gần nhất.

`spatial_lr_scale` của `G_R` là `cameras_extent`. PDF không nói cách khởi tạo `G_R`. Cách trên là lựa chọn của người triển khai và vẫn là điểm nghiên cứu còn mở (xem `implementation_notes.md`).

**`densify_and_prune(max_grad, min_opacity, extent, max_screen_size)`**: clone + split, rồi prune theo opacity và kích thước, như 3DGS.

### 11.2. Rasterizer 3DGS được vendor

Repo gốc chỉ có rasterizer 2DGS (`diff-surfel-anych`). Code mới vendor `diff-gaussian-rasterization` của Inria tại commit `59f5f77` vào [submodules/diff-gaussian-rasterization/](../submodules/diff-gaussian-rasterization/), kèm hai chỉnh sửa:

- thêm `#include <cstdint>` vào `cuda_rasterizer/rasterizer_impl.h`, vì gcc 13 không còn include gián tiếp header này;
- `setup.py` trỏ include GLM sang `../diff-surfel-anych/third_party/glm`, để không phải vendor thêm một bản GLM.

### 11.3. `render_reflection` ([gaussian_renderer/reflection.py](../gaussian_renderer/reflection.py))

Hàm nhận `virtual_camera` từ `reflect_camera`, áp offset của `D_φR` (nếu có) bằng `apply_deformation`, rồi gọi rasterizer 3DGS. Kết quả là `reflection` (`R_t`), `viewspace_points`, `visibility_filter` và `radii`. Ba giá trị cuối phục vụ densification.

---

## 12. Trọng số Fresnel và phép ghép ảnh

File: [utils/fresnel.py](../utils/fresnel.py).

### 12.1. Góc tới: `glass_geometry`

PDF định nghĩa `cos θ_t(u) = |nᵀ(c_t − x_t(u))| / ‖c_t − x_t(u)‖`, với `x_t(u)` là giao điểm của tia qua pixel `u` với kính. Vì `x_t(u)` nằm trên chính tia đó, `c_t − x_t(u)` song song với hướng tia `r(u)`. Công thức vì vậy rút gọn thành

```
cos θ(u) = |nᵀ r(u)|,     r(u) = hướng tia đã chuẩn hoá (lấy từ camera_rays)
```

Hàm cũng trả về `x_t(u)` vì `h_ψ` dùng nó làm đầu vào. Tia không bao giờ chạm plane (khoảng cách âm) được kẹp về khoảng cách 0, tức nhận chính tâm camera; glass map sẽ loại các pixel này. Test `test_cosine_matches_the_definition_through_the_glass_point` đối chiếu dạng rút gọn với định nghĩa đầy đủ.

### 12.2. Schlick

```
F0 = ((η_g − η_a) / (η_g + η_a))²          = 0.04 với 1.0 / 1.5
F(θ) = F0 + (1 − F0)·(1 − cos θ)⁵
```

Nhìn vuông góc cho `F = F0`; nhìn sượt cho `F → 1`.

### 12.3. `fresnel_weight` và `BlendModel`

```
α = clip(F(θ) + h, 0, 1) · glass_map
```

- `glass_map` lấy từ `G_I` đã đóng băng, nên dùng được cả lúc inference khi không có mask GT. `--gt_glass_mask` thay nó bằng mask GT.
- `h` là `h_ψ` khi bật `--optical_correction`, ngược lại là `None`.
- `constant` (ablation no-Fresnel) thay toàn bộ số hạng Schlick bằng một hằng số.

`BlendModel` gói mọi thứ quyết định `α` vào một `nn.Module`, để lưu/nạp bằng một file `blend.pth`:

| Cấu hình | Tham số học | `α` |
|---|---|---|
| mặc định | không có | `F(θ) · glass_map` |
| `--optical_correction` | `h_ψ` | `clip(F(θ) + h_ψ) · glass_map` |
| `--ablate_fresnel` | `constant_logit` | `sigmoid(constant_logit) · glass_map` |

Hằng số khởi tạo tại `logit(F0)`, nên biến thể ablation bắt đầu từ cùng mức pha trộn với Fresnel lúc nhìn vuông góc. `is_trainable` cho script train biết có cần tạo optimizer cho blend hay không.

### 12.4. `OpticalCorrection` (`h_ψ`)

MLP nhỏ (3 lớp, rộng 64) nhận `[γ(x_t(u)), γ(t), cos θ]` với `γ` có 6 tần số cho toạ độ và 4 cho thời gian. Đầu ra là một số cho mỗi pixel. Head khởi tạo bằng 0, nên `h_ψ` mới tạo trùng với Fresnel thuần. `L_opt = mean(h²)` giữ nó nhỏ trong lúc train.

Đầu vào của `h_ψ` không được PDF nêu rõ; đây là lựa chọn của người triển khai.

### 12.5. `composite`

```
Î = (1 − α)·B + α·R
```

Tính trong linear RGB. Ngoài vùng kính `α = 0`, nên `Î = B`.

---

## 13. Phase II: `train_reflection.py`

File: [train_reflection.py](../train_reflection.py).

### 13.1. Chuẩn bị

1. `load_interface` nạp `G_I` và `D_φI` (iteration mới nhất, hoặc `--interface_iteration`), rồi `freeze()` cả hai.
2. `load_plane` đọc `glass_plane.json`.
3. **`DirectCache`**: render `B_t` và glass map cho mọi camera train **một lần duy nhất**, lưu theo `cam.uid`. Phase I không đổi trong suốt Phase II, nên không cần rasterize `G_I` lại ở mỗi iteration. Mỗi iteration vì vậy bỏ được hai pass 2DGS cùng bộ nhớ trung gian của chúng. Khi bật `--gt_glass_mask`, cache lưu mask GT thay cho glass map.
4. Tạo `BlendModel.from_args`, và tạo optimizer Adam (`optical_lr`) chỉ khi blend có tham số học.
5. Tạo `G_R` bằng `create_random` và `D_φR` (`scale_dim = 3`). Trong ablation shared field thì bỏ bước này (mục 16).

### 13.2. Vòng lặp

```
for iteration in 1..N:
    cập nhật lr của G_R và D_φR; cứ 1000 iteration tăng bậc SH một lần
    chọn ngẫu nhiên một camera; lấy B_t và glass_map từ cache
    offsets = D_φR.step(μ_R, t, nhiễu) nếu iteration > warm_up, ngược lại None
    out = render_composite(cam, B_t, glass_map, G_R, Π, offsets, blend)
    loss = Σ reflection_losses(out, …)
    loss.backward()
    log, test, lưu
    densify / prune / reset opacity của G_R
    optimizer.step() cho G_R, D_φR (sau warm-up) và blend (nếu có)
```

### 13.3. Loss: `reflection_losses`

| Thành phần | Công thức | Khi nào bật |
|---|---|---|
| `rgb` | `(1 − λ)·L1(Î, I) + λ·(1 − SSIM(Î, I))`, với `I` là **ảnh gốc** | luôn luôn |
| `motion` | `λ_m · mean|Δμ_R|` | có offsets (đã qua warm-up) |
| `opt` | `λ_o · mean(h²)` | bật `--optical_correction` |

### 13.4. Đánh giá: `evaluate_composite`

Hàm tính bốn PSNR so với ảnh gốc: composite và direct-only, mỗi loại trên toàn ảnh và trong vùng kính. Hiệu giữa `psnr` và `psnr_direct` là phần reflection giải thích thêm được. PSNR vùng kính được tính trên ảnh đã nhân với mask (pixel ngoài mask bằng 0 ở cả hai ảnh), nên con số cao hơn PSNR thật trong vùng kính; nó chỉ dùng để so sánh tương đối giữa composite và direct.

### 13.5. Số iteration thực tế

- Shared field kết hợp Fresnel thuần không có gì để train, nên `num_iterations = 0`: script chỉ đánh giá và lưu ở iteration 0.
- Iteration cuối luôn được thêm vào danh sách test và lưu, nên luôn có kết quả để `render_vigor.py` nạp.

---

## 14. Inference: `render_vigor.py`

File: [render_vigor.py](../render_vigor.py).

`load_model` nạp `G_I` + `D_φI`, rồi (trừ khi `--direct_only`) nạp `G_R` + `D_φR` + blend và plane. Kết quả gói vào một dict `model`. Nhãn thư mục ra là `ours_<iteration Phase II>`, hoặc `direct_<iteration Phase I>` khi chạy direct-only.

`render_set` render mỗi camera bằng `render_direct` và `render_composite` (cùng hai hàm Phase II dùng) rồi ghi ra:

| Thư mục | Nội dung |
|---|---|
| `renders/` | `Î` (hoặc `B` khi direct-only), đổi về sRGB |
| `gt/` | ảnh gốc, đổi về sRGB |
| `transparent_masks/` | mask GT |
| `vis/` | `direct_*`, `reflection_*`, `reflection_contribution_*` (`α·R`), `alpha_*`, `glass_map_*` |

Ba thư mục đầu đúng layout mà `metrics.py` sẵn có đang đọc, nên `metrics.py` dùng lại nguyên vẹn, kể cả phần metric trong vùng kính. Ảnh được ghi bằng một thread pool 8 luồng để GPU không phải chờ ghi đĩa.

Khi `linear_rgb` bật (mặc định của VIGOR-GS), ảnh 3 kênh được đổi linear → sRGB trước khi lưu. Mask và `α` (1 kênh) giữ nguyên.

---

## 15. Lưu và nạp kết quả

File: [utils/vigor_utils.py](../utils/vigor_utils.py).

- `phase_args(dataset, phase_dir)` trả về bản sao tham số với `model_path` trỏ vào thư mục của phase. `Scene` vẫn đọc/ghi `point_cloud/iteration_N/` như cũ, chỉ là trong thư mục con. Nhờ vậy không cần sửa `Scene`.
- `load_interface(dataset, iteration, shuffle)` trả về `(scene, G_I, D_φI)`. Cờ `ablate_factorization` được đọc từ `cfg_args`, nên model non-factorized được dựng lại đúng biến thể.
- `save_reflection(model_path, iteration, blend, gaussians, deform)` luôn lưu `blend.pth`, kể cả khi blend gần như rỗng. Field và deformation chỉ được lưu khi có `G_R` riêng.
- `load_reflection(dataset, iteration)` trả về `(G_R, D_φR, blend, iteration)`. Với shared field, `G_R` và `D_φR` là `None` và nơi gọi tự thay bằng `G_I`, `D_φI`. Iteration mặc định là iteration lớn nhất trong `reflection/blend/`, vì thư mục đó tồn tại ở mọi biến thể.

---

## 16. Các ablation được cài thế nào

| Ablation (`reference.md`) | Cờ | Script | Cơ chế trong code |
|---|---|---|---|
| Không factorization `p·o` | `--ablate_factorization` | `train_interface.py` | `InterfaceGaussianModel(factorized=False)`: `get_occupancy` trả về toàn 1, **cả hai pass dùng `o`**, prune theo `o`, occupancy decay tắt. Tham số `_occupancy` vẫn tồn tại để file PLY có cùng layout, nhưng không ảnh hưởng gì |
| Không tách trường phản chiếu độc lập (cách 1) | `--ignore_clean_images` rồi `render_vigor.py --direct_only` | Phase I + render | Phase I fit ảnh gốc, nên phản chiếu bị "nướng" vào `G_I`. Không chạy Phase II; ảnh cuối là `B` |
| Không tách trường phản chiếu độc lập (cách 2) | `--shared_reflection_field` | `train_reflection.py` | Không có `G_R`: `render_composite` nhận `G_I` và render nó qua camera ảo bằng `render_interface` (nhánh `isinstance`), lấy volume pass làm `R`. `G_I` vẫn đóng băng; chỉ blend được train nếu có tham số |
| Không camera ảo | `--ablate_virtual_camera` | `train_reflection.py` | `render_composite(use_virtual_camera=False)` render `G_R` bằng camera thật. Cờ được lưu vào `reflection/cfg_args` nên `render_vigor.py` cũng làm theo |
| Không Fresnel | `--ablate_fresnel` | `train_reflection.py` | `BlendModel(learn_constant=True)`: `α = sigmoid(c) · glass_map`, một hằng số `c` học được cho mỗi scene, khởi tạo tại `logit(F0)` |

Mỗi biến thể cần thư mục `-m` riêng. Cờ nằm trong `ModelParams`, được ghi vào `cfg_args` và đi theo model đến bước render.

---

## 17. Gradient đi đâu trong từng phase

| | Phase I | Phase II (mặc định) | Phase II, shared field |
|---|---|---|---|
| `G_I` | train | đóng băng (`render_direct` chạy trong `no_grad`, kết quả được cache) | đóng băng, nhưng được render lại qua camera ảo |
| `D_φI` | train sau warm-up | đóng băng | đóng băng |
| Plane `Π` | — | cố định (đọc từ JSON) | cố định |
| `G_R` | — | train | không tồn tại |
| `D_φR` | — | train sau warm-up | không tồn tại |
| Blend (`h_ψ` hoặc hằng số) | — | train nếu bật cờ | train nếu bật cờ |

Hệ quả cần biết: trong Phase II, gradient về `G_R` bị nhân với `α`. Với Fresnel thuần, `α ≈ 0.04` ở góc nhìn gần vuông góc, nên gradient của `G_R` nhỏ khoảng 25 lần so với khi fit thẳng ảnh, và ngưỡng densify mặc định (0.0002) hiếm khi đạt tới. Ảnh hưởng đến kết quả đã được ghi trong `implementation_notes.md`.

Test `test_short_training_run_keeps_phase_one_frozen_and_saves` băm toàn bộ file của `interface/` trước và sau Phase II để chắc chắn không bit nào thay đổi.

---

## 18. Bộ test kiểm chứng những gì

Chạy bằng `python -m pytest` trong WSL (lệnh đầy đủ ở `CLAUDE.md`). Có 13 file và 100 test, tất cả đều pass lần chạy gần nhất (khoảng 7 phút trên RTX 3050).

- Marker `gpu`: test cần CUDA.
- Marker `data`: test cần scene `3D-FRONT-T/scene_1`. Đổi scene bằng biến môi trường `VIGOR_TEST_SCENE`.
- `tests/conftest.py` có hai fixture dùng chung cho cả phiên test: `trained_interface` (Phase I 400 iteration ở `-r 8`, MLP 64×4) và `trained_reflection` (Phase II 300 iteration). Các test end-to-end dùng chung hai kết quả này thay vì mỗi test tự train.
- `tests/helpers.py` có `look_at_camera` (tạo camera tổng hợp) và `project_points` (chiếu điểm bằng ma trận của camera, dùng làm đáp án độc lập).

| File | Kiểm chứng chính |
|---|---|
| `test_env.py` | CUDA và các extension import được; rasterizer 2DGS chạy |
| `test_data.py` | Split theo `split.json`; `time` chuẩn hoá và tăng dần; mask khớp kích thước; thiếu ảnh clean thì báo lỗi, có cờ fallback thì dùng ảnh gốc; `linear_rgb` đổi cả hai ảnh đích; sRGB↔linear là nghịch đảo của nhau; `Scene` nhận ra layout VIGOR |
| `test_deform.py` | Layout positional encoding; shape đầu ra với `scale_dim` 2 và 3; mọi `depth` đều dựng được mạng; mạng mới là phép đồng nhất; quaternion vẫn chuẩn hoá và scale vẫn dương sau deform; đầu ra phụ thuộc cả `t` lẫn `x`; gradient tới mọi lớp nhưng không tới tâm canonical; nhiễu thời gian; lr giảm dần; save/load; freeze |
| `test_interface.py` | Đủ key và shape; gradient tới mọi tham số và tới MLP; `p` và `o` đóng vai trò riêng; offset 0 trùng render canonical; offset tịnh tiến bằng dời Gaussian; PLY round-trip; densify/prune giữ tham số và optimizer đồng bộ; prune theo `p` và tha đĩa chưa được nhìn thấy; reset kẹp đúng giá trị; capture/restore; freeze |
| `test_train_interface.py` | Nhiễu thời gian giảm về 0; các thành phần loss bật đúng lịch; `L_rgb` so với ảnh clean; mask loss đẩy `o` đúng chiều; một lần train ngắn làm PSNR tăng và lưu đủ file |
| `test_plane.py` | SVD khôi phục đúng plane; RANSAC chịu được nhiễu và outlier; chọn plane trội nhất khi có hai plane; kết quả tất định; định hướng về phía camera; chuẩn hoá và save/load; điểm gom được nằm trên kính chứ không trên tường; khôi phục được tấm kính nghiêng; không lấy điểm ngoài mask; plane nhập tay; trích xuất từ interface đã train và báo chất lượng |
| `test_virtual_camera.py` | Công thức phản xạ trên ví dụ trong PDF (camera `(1,0,2)` qua `z = 0` thành `(1,0,−2)`); ma trận khớp với phản xạ điểm và là phép đối hợp; camera ảo thấy vật ở đúng pixel camera thật thấy ảnh gương; điểm trên kính chiếu ra cùng pixel ở cả hai camera; camera thật di chuyển thì camera ảo di chuyển đối xứng; định thức −1 và intrinsics giữ nguyên; phản chiếu hai lần trả lại camera gốc |
| `test_reflection.py` | Đủ key; vật hiện ở đúng pixel của ảnh gương; view ảo bằng view thật của cảnh đã lật; gradient tới mọi tham số và tới MLP; offset tịnh tiến bằng dời Gaussian; khởi tạo ngẫu nhiên chỉ ở phía camera; densify/prune theo opacity; PLY round-trip |
| `test_fresnel.py` | Giới hạn Schlick (`F0` khi nhìn vuông góc, 1 khi nhìn sượt); cos rút gọn khớp định nghĩa qua điểm trên kính; glass map giới hạn phản chiếu; composite là phép trộn tuyến tính; `h_ψ` mới tạo là Fresnel thuần và train được; `α` nằm trong [0, 1] dù correction lớn |
| `test_train_reflection.py` | Các thành phần loss; cache giữ đúng `B_t`; `G_R` học được một phản chiếu tổng hợp trong khi interface đứng yên; một lần train ngắn giữ Phase I không đổi bit nào và lưu đủ; kết quả lưu nạp lại được |
| `test_render_vigor.py` | Layout thư mục đúng thứ `metrics.py` đọc; ảnh nhất quán với nhau; `metrics.py` chạy được trên bộ ảnh vừa render |
| `test_arguments.py` | Tham số thiếu trong `cfg_args` cũ nhận giá trị mặc định |
| `test_ablations.py` | Non-factorized: một opacity điều khiển cả hai pass, prune theo opacity, cờ đọc từ `ModelParams`; không camera ảo: reflection là view không lật; hằng số học thay số hạng Fresnel; blend dựng từ tham số và save/load được; rasterizer 2DGS chạy đúng với view phản chiếu; shared field render interface qua camera ảo; `ignore_clean_images` train trên ảnh gốc; `--direct_only` chỉ cần Phase I; shared field + Fresnel thuần không train gì; shared field + hằng số chỉ train blend; Phase II không camera ảo chạy trọn |

Những gì **chưa** có test chạy thật (chỉ có test đơn vị) được liệt kê ở mục 7 của `implementation_notes.md`.
