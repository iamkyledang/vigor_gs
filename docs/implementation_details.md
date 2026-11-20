# VIGOR-GS: chi tiết triển khai (nhánh `f_refactor`)

Tài liệu này mô tả code hiện tại theo từng bước của framework (`docs/framework.pdf`, nguồn LaTeX `docs/framework.tex`). Mỗi mục nêu: phương trình được triển khai ở đâu, quy ước dữ liệu, và những chỗ code phải chọn một cách hiểu hoặc thêm vào so với PDF. Danh sách gom các chỗ đó nằm ở mục 12. Các ablation có tài liệu riêng ở `docs/ablations.md`. Những gì thay đổi so với nhánh `quan_dang` nằm ở `docs/changes_from_quan_dang.md`.

## 0. Thứ tự đọc code

| Bước | File | Nội dung |
|---|---|---|
| Dữ liệu | `scene/dataset_readers.py`, `scene/cameras.py`, `undistort_sources.py` | Layout đã căn chỉnh, camera, linear RGB, vùng hợp lệ |
| Trường Gaussian | `scene/gaussian_model.py`, `scene/deformation.py` | `G_I^0`, `G_R^0` (Sec. 1), `D_I`, `D_R` (Eq. 1) |
| Phase I | `gaussian_renderer/interface.py`, `utils/losses.py`, `train_interface.py` | Eq. 2–4, 3a |
| Mặt kính | `utils/glass_geometry.py`, `utils/patch_extraction.py`, `extract_glass.py` | `P`, Eq. 9, 9a, 9b, 10 |
| Phase II | `gaussian_renderer/reflection.py`, `gaussian_renderer/composition.py`, `train_reflection.py` | Eq. 5–8, 11 |
| Suy luận, đánh giá | `render.py`, `metrics.py` | Eq. 12–13 |
| Toàn chuỗi | `train.py`, `run_ablations.py` | Chạy các bước theo thứ tự, bộ ablation |
| Dùng chung | `arguments/__init__.py`, `utils/model_io.py`, `utils/checkpoint_utils.py`, `utils/run_utils.py` | Tham số, nạp model, checkpoint, log |
| CUDA | `submodules/diff-surfel-anych` | Rasterizer 2DGS và renderer tia phản xạ `render_reflected_rays` |

## 1. Đối chiếu phương trình và code

| PDF | Ý nghĩa | Code |
|---|---|---|
| Sec. 1, `G_I^0`, `G_R^0` | Đĩa 2D `(μ, q, s⁽²⁾, p, o, a)`, ellipsoid 3D `(μ, q, s⁽³⁾, o, a)` | `InterfaceField`, `ReflectionField` trong `scene/gaussian_model.py` |
| Eq. 1 | `G(t) = D_φ(G^0, t)`, p và o không phụ thuộc t | `Deformation.forward` (`scene/deformation.py`) |
| Sec. 1, `P = {(n_k, d_k, S_k)}` | Patch kính cố định, hữu hạn | `validate_patches`, `glass_geometry.json` |
| Eq. 2 | `B_t = R_2D(G_I(t), C_t)` với opacity `p·o` | `render_interface` (direct pass) |
| Eq. 3 | `L_I = (1/|Ω|) Σ ‖B − I^clean‖₁ + λ_s L_surf` | `pixel_l1`, `interface_losses` |
| Eq. 3a | `L_surf = (1/|Ω_s|) Σ ‖N − N^depth‖₁` | `surface_loss`, `out['surface_valid']` |
| Eq. 4 | Tối ưu `{G_I^0, φ_I}` | `train_interface.training` |
| Eq. 5 | Camera ảo `V_dyn` và renderer `R_3D^refl` | `dynamic_virtual_camera`, `ray_bundle`, `render_reflection` |
| Eq. 6, 13 | `Î = (1 − α) B + α R` (linear RGB) | `render_composite` |
| Eq. 7 | `L_R = (1/|Ω|) Σ ‖Î − I‖₁ + λ_m L_motion + λ_o L_opt` | `reflection_losses` |
| Eq. 7a–7b | So độ thay đổi dự đoán và quan sát giữa hai frame liền kề | `motion_loss`, `previous_frames` |
| Eq. 7c | `L_opt = (1/|Ω|) Σ h_ψ²` | `optical_loss` |
| Eq. 8 | Tối ưu `{G_R^0, φ_R}` và ψ, Θ_I cố định | `train_reflection.training` |
| Eq. 9 | `Π_t(u) = FirstHit(...)`, `x_t(u)` | `first_hit` (có kiểm tra che khuất, mục 6.3) |
| Eq. 9a | `r = d − 2(nᵀd) n` | `reflect`, `first_hit()['reflected']` |
| Eq. 9b | `c_v = x(u_ref) + ε r(u_ref)`, `Q_v = LookAt(r, h)` | `dynamic_virtual_camera`, `look_at` |
| Eq. 10 | `cos θ = |nᵀd|`, `F0`, Schlick | `first_hit()['cosine']`, `schlick` |
| Eq. 11 | `α = clip(F + h_ψ)` trên kính, 0 ngoài kính | `reflection_weight` |
| Eq. 12 | Suy luận với tham số cố định | `render.render_sets` |

## 2. Dữ liệu

### 2.1. Layout của một scene

```
<scene>/colmap/images/<name>          ảnh gốc I_t, đã undistort về camera PINHOLE của COLMAP
<scene>/colmap/sparse/0/              model COLMAP (bin hoặc txt): camera, pose, điểm SfM
<scene>/colmap/frames.json            mỗi frame: time (chuẩn hoá về [0, 1]), split, source_clip (sequence)
<scene>/colmap/split.json             tuỳ chọn, dùng khi frame không có split
<scene>/colmap/masks/<stem>.png       mask kính M_t, đã căn với colmap/images
<scene>/colmap/<clean_images>/        ảnh clean I_t^clean đã căn: <split>/images/<stem>.* hoặc <stem>.*
<scene>/colmap/source_warp.json       tuỳ chọn: mô hình méo của ảnh nguồn (do undistort_sources.py ghi)
```

Mask và ảnh clean được tạo trên ảnh nguồn còn méo, nên phải được warp sang lưới COLMAP **một lần cho mỗi scene** bằng `undistort_sources.py` (lấy từ `viet_huynh`). Reader không warp gì cả, chỉ resize theo `-r`. Cả VSGD 0758, WayveScene101 scene_001 và 3D-FRONT-T scene_1 trong `data/` đều đã ở layout này.

- Frame train thiếu ảnh clean là lỗi, trừ khi bật `--ignore_clean_images` (ablation no-separation). Frame test thiếu ảnh clean thì được phép: nó chỉ được chấm với ảnh gốc.
- Ảnh trong model COLMAP mà không có trong `frames.json` (ví dụ các camera khác của một rig) bị bỏ qua và in ra số lượng.
- Chỉ nhận camera PINHOLE / SIMPLE_PINHOLE. Camera méo bị từ chối kèm hướng dẫn chạy `undistort_sources.py`.
- `center`, `extent` tính như `getNerfppNorm` của 3DGS: tâm là trung bình các tâm camera train, `extent` bằng 1.1 lần khoảng cách lớn nhất.

### 2.2. Vùng hợp lệ Ω

Undistort một ảnh góc rộng để lại những pixel không có pixel nguồn nào ánh xạ tới (viền đen). Khi có `source_warp.json`, reader dựng lại map của `cv2.remap` (`utils/undistort_utils.build_remap`). Pixel có toạ độ nguồn nằm ngoài ảnh nguồn được đánh dấu không hợp lệ. Cả hai định dạng của file này đều đọc được: bản cũ có `model` ở cấp ngoài cùng và một tiêu cự `f`, bản mới có `model` cho từng camera kèm `fx`, `fy`. Khi giảm độ phân giải, một pixel chỉ hợp lệ nếu toàn bộ diện tích của nó hợp lệ. Ω của mọi loss và metric là tập pixel hợp lệ này. Với 3 scene hiện có, Ω là toàn ảnh.

### 2.3. Camera và màu

`scene/cameras.py` giải mã pixel một lần, đúng độ phân giải train. Ảnh lưu dạng `uint8`, mask dạng `bool`, trên `data_device` (mặc định là RAM). Các property trả tensor CUDA:

- `original_linear`, `clean_linear`: ảnh ở **linear RGB**, vì mọi phép khớp và blend của framework đều ở linear RGB (Eq. 3, 6). Giải mã sRGB theo chuẩn trong `utils/color_utils.srgb2linear`.
- `original_srgb`: ảnh để lưu GT và tính metric.
- `glass_mask` (M_t), `valid_mask` (Ω).

Intrinsics được scale theo `-r`, principal point theo quy ước COLMAP (tâm pixel tại +0.5). Ma trận chiếu có principal point lệch tâm: `P[0,2] = 2cx/W − 1`. Test `test_camera_scales_intrinsics_and_decodes_linear_rgb` kiểm tra rằng rasterizer và tia trong `utils/point_utils.camera_rays` dùng cùng một quy ước tâm pixel.

## 3. Hai trường Gaussian

`GaussianField` (`scene/gaussian_model.py`) chứa phần không phụ thuộc loại primitive: lưu tham số, phẫu thuật optimizer khi clone/split/prune (moment Adam của điểm cũ được giữ, điểm mới bắt đầu từ 0), lịch densify của 3DGS, `save_ply`, `capture`/`restore`. Tham số theo từng Gaussian được liệt kê một lần trong `PARAMS`.

| | `InterfaceField` (G_I) | `ReflectionField` (G_R) |
|---|---|---|
| Primitive | Đĩa 2D (`SCALE_DIM = 2`) | Ellipsoid 3D (`SCALE_DIM = 3`) |
| Tham số | xyz, SH, opacity o, scaling, rotation, **occupancy p** | xyz, SH, opacity o, scaling, rotation |
| Opacity khi render | `p · o` (direct pass) | `o` |
| Khởi tạo | Điểm SfM, màu đổi sang linear; p = 0.1, o = 0.5, quaternion ngẫu nhiên, scale = 0.1 × khoảng cách láng giềng | Các điểm dọc theo tia phản xạ (mục 7.6) |
| "Presence" để prune/reset | p (o nếu không factorize) | o |
| Giới hạn điểm | `--max_interface_points` 500k | `--max_reflection_points` 250k |

Occupancy, opacity và màu là tham số canonical, nên không phụ thuộc thời gian (Sec. 1). `factorized=False` (`--ablate_factorization`) bỏ hẳn tham số p (p ≡ 1).

**Lịch densify** (theo 3DGS/2DGS):

- Từ `densify_from_iter` đến `densify_until_iter`, cứ mỗi `densification_interval` vòng: clone các Gaussian nhỏ, split các Gaussian lớn có gradient trung bình vượt ngưỡng. Khi vượt giới hạn điểm, chỉ những Gaussian có gradient mạnh nhất được nhân lên.
- Prune khi presence < `min_presence` (0.005).
- Mỗi `presence_reset_interval` (3000) vòng, presence bị kẹp về tối đa 0.01, như opacity reset của 3DGS.
- Sau lần reset đầu tiên, prune Gaussian lớn hơn 10% extent, và với G_I prune thêm đĩa có bán kính trên màn hình lớn hơn `max_screen_size` (20 px). Không có bước này, vài Gaussian khổng lồ trong cảnh rộng (Wayve) phủ mọi tile và làm buffer của rasterizer phình ra.

Gradient dùng để densify:
- G_I: chuẩn của gradient toạ độ màn hình (`viewspace_points.grad[:, :2]`) như 2DGS.
- G_R: renderer theo tia không có toạ độ màn hình, nên dùng chuẩn gradient 3D của tâm đã deform nhân với extent (để ngưỡng không phụ thuộc thang đo scene), ngưỡng mặc định 2e-5.

## 4. Deformation (Eq. 1)

`scene/deformation.py`, theo Yang et al. (Deformable 3D Gaussians):

```
(Δμ, Δq, Δs) = MLP(γ(μ̂), γ(t))
μ' = μ + extent · Δμ,   q' = normalize(q + Δq),   s' = exp(log s + Δs)
```

- `μ̂ = (μ − center) / extent`. γ là positional encoding sin/cos với 2^k·π: `deform_xyz_freqs` = 6, `deform_time_freqs` = 6.
- MLP: `deform_depth` lớp × `deform_width` (mặc định 4 × 128), ReLU. Lớp đầu ra khởi tạo bằng 0, nên mạng mới là hàm đồng nhất.
- Tâm canonical được detach ở đầu vào, như Yang et al.: gradient vẫn tới μ qua `μ + Δμ` nhưng không đi ngược qua encoding, nhờ vậy field canonical và deformation tách được. Test `test_deformation_starts_as_identity_and_static_never_moves` kiểm tra điều này.
- Hai trường có hai instance riêng (chung class, không chung trọng số). Chỉ hình học thay đổi theo t.
- `warm_up` (1000 vòng) đầu mỗi phase chỉ dùng field canonical.
- Learning rate giảm mũ tới 1% sau cả phase.
- `static=True` (`--static_interface`, `--static_reflection`) bỏ mạng: field đứng yên theo thời gian.

## 5. Phase I (Eq. 2–4)

### 5.1. Renderer (`gaussian_renderer/interface.py`)

Rasterizer 2DGS `diff_surfel_anych` được build in-place trong submodule của nhánh này. `gaussian_renderer/__init__.py` đưa thư mục đó lên đầu `sys.path`, nên bản `diff_surfel_anych` đã cài trong môi trường (của RT-Splatting) không bị thay.

- **Direct pass:** đĩa render với `p·o`, cho ra `B_t`, depth (expected, hoặc median theo `depth_ratio`), alpha, pháp tuyến render `N_t` (đổi sang hệ thế giới) và pháp tuyến suy từ depth `N_t^depth` (tích chéo của sai phân hàng và cột của điểm 3D chiếu ngược). `Ω_s` của Eq. 3a là các pixel có alpha > 0.1, cả lân cận 3×3 cũng được phủ, và cả hai pháp tuyến đều xác định.
- **Surface pass** (chỉ khi cần): cùng các đĩa, render với opacity = p, mang o như một kênh phụ. `glass_map = surface_alpha − Σ w_i o_i = Σ w_i (1 − o_i)`: cao ở nơi có bề mặt (p cao) nhưng cho ánh sáng đi qua (o thấp). Chỉ dùng cho prior mask và việc trích patch; không bao giờ dùng để blend (Sec. 1: "There is no learned image-space plane mask").
- **`DUAL_VISIABLE 1`** (`cuda_rasterizer/auxiliary.h`): pháp tuyến của mỗi đĩa được lật về phía camera, như 2DGS gốc [1]. RT-Splatting để giá trị 0. Khi đó pháp tuyến render chỉ quay về camera ở khoảng 41% pixel, trong khi pháp tuyến từ depth luôn quay về camera, và L_surf dạng L1 (Eq. 3a) phạt mỗi đĩa quay lưng khoảng 2 đơn vị.

### 5.2. Loss (`utils/losses.py`)

Chuẩn hoá theo pixel như mục "Reading the losses" của PDF: `‖·‖₁` cộng ba kênh của một pixel, rồi chia cho số pixel hợp lệ `|Ω|`.

```
L_I = pixel_l1(B, I^clean, Ω) + λ_s · L_surf(N, N^depth, Ω_s ∩ Ω) [+ λ_glass · BCE(glass_map, M_t)] [+ λ_dssim · D-SSIM]
```

| Thành phần | Mặc định | Ghi chú |
|---|---|---|
| `λ_s` (`--lambda_surface`) | 0.05, từ vòng `surface_from_iter` = 1000 | Eq. 3a; Ω_s rỗng thì đóng góp 0 |
| `λ_glass` (`--lambda_glass`) | 0.05 | **Không có trong Eq. 3.** Xem mục 5.3 |
| `λ_dssim` | 0 | Tuỳ chọn; PDF bỏ SSIM |

### 5.3. Vì sao có prior mask kính

Với ảnh clean, kính không hiện trên ảnh, nên chỉ riêng Eq. 3 không có tín hiệu nào cho biết kính nằm ở đâu. Mục Scope của PDF cũng nói p và o không tách được nếu không có prior. Nếu không có hình học kính biết trước, `S_k` phải lấy "từ hình học Phase I" (Sec. 1). Prior BCE giữa `glass_map` và mask train làm cho G_I có một lớp đĩa "có mặt nhưng trong suốt" đúng chỗ kính, để `extract_glass.py` lấy mẫu được. Prior chỉ dùng mask **train**. Mask test không bao giờ được dùng, trừ khi chấm metric vùng kính. Khi có hình học kính biết trước thì có thể đặt `--lambda_glass 0`.

### 5.4. Vòng lặp và output

- View được lấy lần lượt đủ mọi view train trong mỗi epoch, thứ tự ngẫu nhiên mới mỗi epoch (`ViewSampler`).
- SH tăng một bậc mỗi 1000 vòng. Learning rate của xyz giảm mũ như 3DGS.
- Mỗi `test_interval` vòng: PSNR của B so với target (linear, trên Ω) và IoU của `glass_map > 0.5` với mask, trên tập test.
- Output: `<model>/config.json` (`ModelParams`, mọi bước sau đọc lại); `<model>/interface/{config.json, latest.pth, training.jsonl, point_cloud.ply}`.
- `--resume` chạy tiếp checkpoint. Không có `--resume` thì không ghi đè checkpoint đã có.

## 6. Mặt kính (Sec. 1, Eq. 9)

### 6.1. Định dạng

`<model>/glass_geometry.json`:

```json
{"patches": [{"normal": [nx, ny, nz], "offset": d, "vertices": [[x, y, z], ...]}],
 "settings": {"occlusion_tolerance": 0.02, "occlusion_alpha": 0.5, "ablate_occlusion": false, "glass_margin": 0.043},
 "quality": {"coverage": 0.93, "precision": 0.88, "views": 12, "samples": 9000},
 "source": "phase_i_iteration_30000",
 "accepted": true}
```

Mặt phẳng là `nᵀx + d = 0`. `vertices` là một đa giác lồi có thứ tự, nằm trên mặt phẳng. `validate_patches` chuẩn hoá n và từ chối: pháp tuyến bằng 0, đỉnh không đồng phẳng, đa giác có diện tích 0, đa giác không lồi. Hình học biết trước được truyền bằng `--glass_geometry <file>` (một danh sách patch, hoặc dict có khoá `patches`).

### 6.2. Trích từ Phase I (`utils/patch_extraction.py`)

1. **Lấy mẫu.** Trên khoảng 12 view train, surface pass của G_I đã đóng băng cho depth của bề mặt "có mặt" đầu tiên. Những pixel vừa nằm trong mask kính train, vừa có `surface_alpha > 0.3` và `glass_map > 0.1`, được chiếu ngược ra 3D (tối đa `plane_max_points` = 12000 điểm).
2. **Mặt phẳng.** Chạy RANSAC tuần tự (512 giả thuyết, ngưỡng `plane_threshold × extent`), tinh chỉnh bằng SVD trên các inlier, bỏ inlier, rồi lặp lại tới `max_patches` (4) mặt phẳng.
3. **Patch.** Các inlier đồng phẳng được tách thành thành phần liên thông: union-find trên đồ thị bán kính bằng 8 lần khoảng cách láng giềng trung vị, trong toạ độ mặt phẳng. Mỗi thành phần đủ lớn thành một convex hull. Nhờ vậy các ô kính rời nhau trên cùng một bức tường thành các patch riêng.
4. **Định hướng.** Pháp tuyến được lật về phía có đa số camera. Dấu này không ảnh hưởng tới phương trình (`cos θ` lấy trị tuyệt đối, phép phản xạ không đổi khi n đổi dấu), nhưng làm file dễ đọc hơn.
5. **Chất lượng.** Giao các patch với tia của các view mẫu (có kiểm tra che khuất), rồi so với mask: `coverage = |hit ∩ mask| / |mask|`, `precision = |hit ∩ mask| / |hit|`. Dưới `plane_min_coverage` hoặc `plane_min_precision` (0.5) thì file được ghi với `"accepted": false`, `extract_glass.py` kết thúc với lỗi, và Phase II từ chối chạy.

`<model>/glass_preview/<view>.png`: xanh lá là mask không bị patch nào phủ, đỏ là hit ngoài mask, vàng là hit nằm trong mask, xanh dương là hit bị che khuất.

Giới hạn: ảnh clean và silhouette không xác định duy nhất độ sâu của kính trong suốt (Scope của PDF). Vì vậy residual và độ khớp với mask chỉ là phép kiểm, không phải ground truth. Convex hull chỉ xấp xỉ biên. Kính có lỗ hoặc kính cong cần hình học biết trước.

### 6.3. FirstHit và kiểm tra che khuất (Eq. 9)

`first_hit(view, patches, epsilon, occlusion)` làm các bước sau cho mọi pixel:

1. Bắn tia camera (hướng đơn vị d), giao với từng mặt phẳng, chỉ giữ giao điểm có khoảng cách > ε nằm trong đa giác `S_k`. Chọn giao điểm gần nhất, tức "the nearest hit if projections overlap".
2. **Kiểm tra che khuất** (bạn đã chọn phương án A). Eq. 9 chỉ xét các patch. Một tấm kính nằm sau vật đục (người đứng trước cửa kính) vẫn được chọn, và phản xạ bị blend đè lên vật đó. Vì vậy một hit bị loại khi **lượt render che khuất** của G_I đã đóng băng **đục** ở pixel đó (`alpha ≥ occlusion_alpha` = 0.5) **và** depth của nó nằm **trước** kính: `depth < z_hit · (1 − occlusion_tolerance)`, với `occlusion_tolerance` = 0.02 tương đối.
   - **Lượt render che khuất** (`utils/model_io.DirectPass`) là direct pass của G_I sau khi **bỏ các đĩa nằm trên chính các patch kính**: những đĩa cách mặt phẳng một patch không quá `glass_margin` (= ngưỡng RANSAC `plane_threshold × extent`) và có hình chiếu nằm trong đa giác (`on_patches`). Các đĩa đó là lớp kính. Với ảnh clean còn sót phản chiếu (DAI), Phase I "vẽ" phần sót đó lên đúng lớp này, làm nó hơi đục. Nếu tính cả lớp này, chỉ cần sai số fit mặt phẳng là depth của nó đã nằm trước patch, và kính thật bị báo là bị che. Lần chạy thử trên 3D-FRONT-T scene_1 cho thấy đúng hiện tượng này (mục 13). B dùng để blend vẫn là direct pass đầy đủ.
   - Ở pixel kính không bị che, depth của lượt render che khuất là depth của thứ nằm **sau** kính, nên hit được giữ. Một bề mặt nằm ngay tại hoặc sau kính không bao giờ bị coi là vật che.
   - Đây chỉ là hình học đã dựng lại: không có mask học được, không có tham số mới.
   - `--ablate_occlusion` (chọn lúc `extract_glass.py`, lưu trong `settings`) trả về Eq. 9 nguyên văn.
3. Trả về các map (H, W): `patch` (int32, −1 là không có kính), `visible`, `occluded`, `distance`, `cosine = |nᵀd|` (Eq. 10), và các map (H, W, 3): `hits` x_t(u), `normals`, `real_directions` d, `reflected` r (Eq. 9a).

Luật này được dùng **giống hệt nhau** khi trích patch (đo quality), khi khởi tạo và train Phase II, và khi render.

## 7. Phase II (Eq. 5–8, 11)

### 7.1. Tia phản xạ và camera ảo động V_dyn

Với một pixel kính: `r = d − 2(nᵀd) n` (Eq. 9a). Tia đúng vật lý là `x_t(u) + s·r_t(u)`, `s ≥ 0` (Sec. 3, "Render").

`dynamic_virtual_camera` triển khai Eq. 9b cho từng patch nhìn thấy được:
- `u_ref` là pixel của vùng nằm gần vị trí pixel trung bình của vùng nhất.
- `c_v = x(u_ref) + ε r(u_ref)`.
- `Q_v = LookAt(r(u_ref), h_t)`: phép quay đúng (det = +1) với trục x phải, y xuống, z theo r. Hướng "up" lấy từ camera thật (−y của camera thật trong hệ thế giới).
- Hàm này là một và chỉ một: với mỗi vùng/patch, nó đặt lại pose cho cùng một camera, không lưu camera riêng cho từng patch.

Tính chất quan trọng: trong một patch phẳng, **mọi tia phản xạ đều đi qua tâm camera đối xứng** `c'_k = mirror_k(c)` khi kéo dài ngược lại (test `test_dynamic_virtual_camera_pose_and_mirrored_center`). Tức là tập tia đúng của một patch là một phép chiếu tâm (từ `c'_k`), chỉ có điều tia bắt đầu tại mặt kính. Union các tia qua nhiều patch thì không còn chung tâm, đúng như PDF nói ("non-central rays").

### 7.2. Ba chế độ tia (`ray_bundle`)

| Chế độ | Gốc tia | Hướng | Dùng cho |
|---|---|---|---|
| `exact` (mặc định) | `x_t(u) + ε r` | `r_t(u)` | Renderer của framework |
| `pinhole` (`--pinhole_virtual_camera`) | `c_v` của vùng (Eq. 9b) | `r_t(u)` | Xấp xỉ mà PDF nhắc tới: "using that pose alone in an ordinary pinhole rasterizer is generally only an approximation". Sai số thị sai lớn với vật ở gần kính |
| `unreflected` (`--ablate_virtual_camera`) | `x_t(u) + ε d` | `d_t(u)` | Không có camera ảo: G_R được nhìn thẳng xuyên qua kính |

Test `test_a_gaussian_in_front_of_the_glass_appears_at_its_mirror_image` đặt một Gaussian nhỏ trước kính. Với tia `exact`, pixel sáng nhất là hình chiếu của ảnh gương của nó, sai số ≤ 1 px. Với `pinhole`, nó lệch hơn 2 px.

### 7.3. Renderer theo tia (CUDA, từ `quan_dang`)

`diff_surfel_anych.render_reflected_rays` (`rasterize_points.cu`):
- Với mỗi pixel và mỗi Gaussian ứng viên của tile, tìm điểm trên tia gần tâm Gaussian nhất theo metric của Gaussian (`A = R / s`). Gaussian bị bỏ nếu điểm đó nằm sau gốc tia (`s ≤ 0`, đúng điều kiện `s ≥ 0` của PDF) hoặc ngoài bán kính `ray_cutoff` (3.5σ).
- `α = min(0.99, o · exp(−q/2))`, composite từ trước ra sau, dừng khi T < 1e-4.
- Backward tính gradient theo tâm, ma trận trục nghịch đảo (sau đó PyTorch đưa về rotation, scale và deformation), opacity và màu. Test so forward và gradient với một bản tham chiếu PyTorch viết độc lập.
- Xấp xỉ (giống 3DGS): thứ tự composite theo depth của tâm Gaussian trong tile (theo trục của camera đối xứng), đánh giá Gaussian tại điểm gần nhất thay vì tích phân thể tích. Không mô hình khúc xạ, phản xạ nhiều lần, hay nhiều lần giao với kính.

### 7.4. Danh sách ứng viên theo tile (`tile_candidates`)

Pixel mà tia của nó đi qua một điểm P chính là hình chiếu qua camera thật của:
- `mirror_k(P)` với chế độ `exact` (vì camera đối xứng chính là ảnh gương của camera thật);
- `mirror_k(P + c'_k − c_v)` với chế độ `pinhole` (tia pinhole là tia exact tịnh tiến đi `c_v − c'_k`);
- `P` với chế độ `unreflected`.

Mỗi Gaussian được bao bằng một hình cầu bán kính `ray_cutoff × max scale`. Hình chiếu bảo thủ của nó cho dải tile, chỉ giữ những tile mà patch đó thật sự xuất hiện. Các entry `k·N + i` được sắp theo (tile, depth) bằng một khoá 64 bit. Tổng số cặp (Gaussian, tile) vượt `--max_tile_entries` là lỗi rõ ràng, không lặng lẽ bỏ Gaussian. Test kiểm tra rằng với cả ba chế độ, mọi Gaussian bị tia cắt đều có trong danh sách của tile.

Màu là SH của từng Gaussian theo hướng tia tới nó, `normalize(μ − tâm tia của patch)`. Một Gaussian thấy qua hai patch vì vậy có hai màu phụ thuộc hướng nhìn. Bức xạ linear không âm và có thể vượt 1 trước khi blend.

### 7.5. Trọng số phản xạ và blend (`gaussian_renderer/composition.py`)

```
F0 = ((η_a − η_g)/(η_a + η_g))²,   F(θ) = F0 + (1 − F0)(1 − cos θ)⁵        (η_a = 1.0, η_g = 1.5, nên F0 = 0.04)
α = clip_[0,1](F(θ) + h_ψ) trên pixel kính được chọn, 0 ở các pixel khác
Î = (1 − α) B + α R
```

**h_ψ** (`OpticalCorrection`, bật mặc định theo Eq. 7–8; `--pure_fresnel` để tắt):
- Đầu vào: `γ(x̂_t(u))` (điểm kính bị tia của pixel cắt, chuẩn hoá theo center/extent, 4 tần số), `γ(t)` (2 tần số) và `cos θ`. MLP 2 lớp ẩn × 64, lớp ra khởi tạo bằng 0, nên ban đầu là Fresnel thuần. Chỉ tính trên pixel kính; ngoài kính h = 0.
- PDF viết `h_ψ(u, t)`. Khi đã biết camera thì `x_t(u)` là hàm của u, nên đây vẫn là `h_ψ(u, t)`. Khác với toạ độ pixel thô (`quan_dang` dùng (u, v) chuẩn hoá), `x_t(u)` chỉ đúng **cùng một chỗ trên kính** ở mọi view, nên hiệu chỉnh học ở view train áp được lên view test.
- `--optical_bound` mặc định 0: không chặn, chỉ có `clip[0,1]` của Eq. 11 và `L_opt`. Giá trị > 0 thì chặn bằng `bound · tanh(h / bound)`. `quan_dang` chặn cứng ±0.1.

`--ablate_fresnel`: thay F(θ) bằng một hằng số học được `σ(c)` cho mọi pixel kính, khởi tạo bằng F0. Nếu h_ψ đang bật thì nó cộng lên trên hằng số đó.

### 7.6. Khởi tạo G_R

Trên khoảng 12 view train, lấy ngẫu nhiên các pixel kính hợp lệ, đặt một điểm tại `gốc tia + s · hướng tia` với `s ∈ extent · [0.1, 3.1]` (theo chế độ tia đang dùng), màu `(I − (1 − F) B) / F`, kẹp trong [0, 20]. Đó là bức xạ có thể tự mình giải thích phần dư. Opacity 0.1, scale theo láng giềng gần nhất.

### 7.7. Loss và vòng lặp (`train_reflection.py`)

```
L_R = pixel_l1(Î, I, Ω) + λ_m · motion_loss(Î_k, Î_{k−1}, I_k, I_{k−1}, Ω_k ∩ Ω_{k−1}) + λ_o · optical_loss(h, Ω)
```

- **L_motion (Eq. 7a–7b):** frame k−1 là frame liền trước trong **cùng sequence (`source_clip`) và cùng camera (`sensor_id`)** theo thời gian (`previous_frames`). VSGD 0758 có hai clip, nên frame đầu của mỗi clip không có frame liền trước. Cả hai dự đoán đều được render và đều nhận gradient. Đây là ràng buộc trên độ thay đổi của ảnh, không phải phạt độ lớn deformation. Mặc định `λ_m` = 0.1.
- **L_opt (Eq. 7c):** trung bình h² trên Ω (h = 0 ngoài kính), `λ_o` = 0.01, chỉ khi có h_ψ.
- G_I, D_I và các patch hoàn toàn cố định. `DirectPass` render B_t và lượt render che khuất (depth, alpha) một lần cho mỗi view rồi cache trong RAM (LRU, `--cache_views` = 32), vì Phase II không bao giờ đổi chúng. Ảnh clean không được nạp ở Phase II.
- Optimizer: Adam cho G_R (lr theo 3DGS), cho D_R (`deform_lr`, giảm mũ), cho h_ψ / hằng số (`optical_lr` = 1e-3).
- Mỗi `test_interval` vòng, trên tập test: PSNR của composite và của direct, toàn ảnh và vùng kính (linear, Ω). Hiệu giữa hai PSNR là phần mà phản xạ giải thích được.
- Output: `<model>/reflection/{config.json, latest.pth, training.jsonl, point_cloud.ply}`.

## 8. Suy luận (`render.py`, Eq. 12–13)

- Nạp Phase I, Phase II (cấu hình từ `reflection/config.json`) và `glass_geometry.json`. Mỗi pixel chọn patch kính nhìn thấy đầu tiên theo đúng luật lúc train. **Mask test không bao giờ được dùng để blend.**
- Output `<model>/<split>/ours_<iter>/{renders, gt, glass_masks, valid_masks}`.
- `--components` thêm: `direct/`, `reflection/`, `alpha/`, `patches/` (màu theo patch được chọn), `clean/` (ảnh clean của view, cho metric tách lớp), `virtual_cameras.json` (pose V_dyn của mỗi patch nhìn thấy được ở mỗi frame).
- `--direct_only` (tự bật với ablation no-separation) render `direct_<iter>`.
- `comparison.mp4`: ours bên trái, GT bên phải, theo thứ tự sequence, camera, thời gian.
- `render_stats.json`: thời gian render mỗi frame đo bằng CUDA event, gồm direct pass, FirstHit, renderer tia và blend; không gồm nạp dữ liệu và ghi file.

## 9. Metric (`metrics.py`)

Chấm trên PNG sRGB, từng view một:

| Metric | Định nghĩa |
|---|---|
| PSNR, SSIM, LPIPS | Toàn ảnh so với ảnh gốc (LPIPS VGG, đầu vào [−1, 1]) |
| Valid PSNR | Chỉ trên Ω |
| Glass PSNR / Opaque PSNR | Pixel kính / không kính được annotate trong Ω, chia cho đúng số pixel đó |
| Masked PSNR/SSIM/LPIPS (RT) | Nhân cả hai ảnh với mask rồi tính trung bình toàn ảnh, theo giao thức RT-Splatting (nền 0 làm số bị thổi phồng) |
| LPIPS (RT input) | LPIPS với đầu vào [0, 1], quy ước của RT-Splatting |
| **Transmission PSNR**, Glass Transmission PSNR | B so với ảnh clean (cần `--components`): lớp không phản chiếu được khôi phục tốt đến đâu |
| FPS, Training Time | Từ `render_stats.json` và `<phase>/training.jsonl` (cộng qua các session resume) |

Transmission PSNR là thước đo cho chính việc tách lớp. PSNR so với ảnh gốc ưu tiên những phương pháp khớp thẳng ảnh gốc (ví dụ RT-Splatting, hoặc ablation `no_separation`), nên một mình nó không đủ để đánh giá VIGOR-GS.

## 10. Cấu hình, checkpoint, resume

- `arguments/__init__.py`: `ModelParams` (scene, độ phân giải, kiến trúc, ablation của Phase I), `PipelineParams`, `InterfaceOptimizationParams`, `GlassParams`, `ReflectionParams` (model Phase II và ablation), `ReflectionOptimizationParams`.
- Phase I ghi `ModelParams` vào `<model>/config.json`. Các bước sau parse `ModelParams` với `sentinel=True` và `load_model_args` chỉ ghi đè những gì được truyền tường minh. Đổi kiến trúc của Phase I (SH, deformation, ablation Phase I) ở bước sau là lỗi.
- Checkpoint (`utils/checkpoint_utils.py`, version 2) gồm: field, optimizer và moment của nó, deformation và optimizer của nó, module phụ (h_ψ, hằng số Fresnel) và optimizer của chúng, trạng thái RNG. File được ghi tạm rồi `os.replace`. Resume chạy tiếp đúng từng bit (`test_checkpoint_resume_continues_exactly`).
- Không có `--resume` thì một checkpoint đã có không bao giờ bị ghi đè.

## 11. Lệnh

Build extension (một lần, và sau mỗi lần sửa CUDA hoặc header):

```bash
cd submodules/diff-surfel-anych
rm -rf build && CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6 python setup.py build_ext --inplace
```

Phải `rm -rf build` khi chỉ sửa header (ví dụ `auxiliary.h`): ninja không tự build lại khi header đổi.

Chạy cả chuỗi:

```bash
python train.py -s data/datasets/datasets/3D-FRONT-T/scene_1 -m output/3dfront_full -r 2
python train.py -s <scene> -m <model> --glass_geometry patches.json        # hình học kính biết trước
python train.py -s <scene> -m <model> --reflection_args "--iterations 20000 --lambda_motion 0.05"
```

Từng bước:

```bash
python train_interface.py -s <scene> -m <model> -r 2
python extract_glass.py -m <model>                  # hoặc --glass_geometry patches.json
python train_reflection.py -m <model>
python render.py -m <model> --skip_train --components
python metrics.py -m <model>
```

Bộ ablation: `python run_ablations.py -s <scene> --root output/<scene>_ablations -r 2` (xem `docs/ablations.md`).

Test: `python -m pytest`, gồm 50 test với scene tổng hợp `tests/synthetic.py`. Test GPU bị skip khi không có CUDA. Đánh dấu `slow` cho các test có train ngắn.

## 12. Những chỗ code chọn hoặc thêm so với PDF

| Chỗ | PDF | Code | Lý do |
|---|---|---|---|
| Che khuất trong FirstHit | Eq. 9 chỉ xét các patch | Loại hit khi depth của G_I đã đóng băng (không tính lớp đĩa nằm trên kính) nằm trước kính | Bạn đã chọn phương án A; ablation `no_occlusion` |
| Prior mask kính ở Phase I | Eq. 3 chỉ có L1 + λ_s L_surf | Thêm λ_glass · BCE(glass_map, M_t) | Cần có hình học kính để trích `S_k` từ Phase I (Sec. 1, Scope). Tắt được khi có hình học biết trước |
| Nguồn của `S_k` | "known geometry or Phase I interface geometry" | Cả hai: `--glass_geometry` hoặc RANSAC + convex hull từ surface pass, có ngưỡng chất lượng | |
| Renderer `R_3D^refl` | Tia không chung tâm `x + s r` | Kernel CUDA theo tia (closest approach), thứ tự theo depth của tâm | Xấp xỉ như 3DGS, mục 7.3 |
| Vai trò của V_dyn (Eq. 9b) | Pose để mô tả hướng nhìn; dùng pose này trong pinhole chỉ là xấp xỉ | Tính theo từng vùng/frame, xuất ra `virtual_cameras.json`; là gốc tia trong ablation `pinhole_virtual_camera` | Render chính dùng tia đúng |
| Đầu vào của h_ψ(u, t) | Hàm của pixel và thời gian | `γ(x_t(u))`, `γ(t)`, `cos θ` | Áp đúng chỗ kính ở view mới |
| Chặn h_ψ | Không có, chỉ có clip[0,1] | Mặc định không chặn; `--optical_bound` để chặn | |
| D-SSIM | Bị bỏ | `--lambda_dssim` mặc định 0 | |
| Thứ tự trong L_motion | "neighboring timestamps", "ordered training sequence" | Frame liền trước cùng `source_clip` và `sensor_id` | Các clip khác nhau không liên tục |
| Pháp tuyến 2DGS | 2DGS [1] | `DUAL_VISIABLE 1` | RT-Splatting tắt cờ này, làm L1 của Eq. 3a phạt đĩa quay lưng |
| Densify, reset, prune | Không nói | Theo 3DGS/2DGS (mục 3) | Ổn định số điểm và bộ nhớ |
| Khởi tạo G_R | Không nói | Điểm dọc theo tia, màu theo phần dư (mục 7.6) | |
| Vùng hợp lệ Ω | "evaluated image pixels" | Pixel có pixel nguồn sau undistort | |

## 13. Giới hạn và việc còn mở

- Với scene chỉ có ảnh clean pseudo-GT (DAI), B bị buộc theo các ảnh đó, và G_I bị đóng băng ở Phase II đúng như framework yêu cầu. Phân tích trên VSGD 0758 trước đây cho thấy đây là nguồn chính khiến PSNR so với ảnh gốc thấp hơn RT-Splatting. Nên báo cáo kèm Transmission PSNR.
- Hình học kính trích từ Phase I phụ thuộc vào prior mask và chất lượng Phase I. Ngưỡng chất lượng 0.5/0.5 chỉ là để chặn hình học hỏng, không phải thước đo độ chính xác.
- **Hiệu năng.** Danh sách ứng viên tile được dựng bằng PyTorch (`tile_candidates`), còn kernel CUDA của từng pixel duyệt các entry của tile trực tiếp từ global memory (không dùng shared memory như 3DGS). Hai phần này chiếm phần lớn thời gian của Phase II và của render. Bộ nhớ tỉ lệ với số cặp (Gaussian, tile), bị giới hạn bởi `--max_tile_entries`. Hướng tối ưu: chuyển binning sang CUDA (duplicate keys + radix sort như 3DGS) và nạp Gaussian của tile vào shared memory.
- Không mô hình khúc xạ, phản xạ lồng nhau, hay nhiều lần giao với kính (Scope của PDF).
- Baseline RT-Splatting không nằm trong nhánh này (nhánh `quan_dang` đã thay `train.py` của RT). Hãy chạy nó từ `viet_huynh` hoặc `main`.

## 14. Lần chạy thử trên dữ liệu thật

3D-FRONT-T scene_1, `-r 4` (240×135), ảnh clean DAI, RTX 3050 Laptop 4 GB (GPU bị hạ xung do nhiệt, 240–712 MHz trong lúc chạy):

```bash
python train.py -s data/datasets/datasets/3D-FRONT-T/scene_1 -m output/smoke_3dfront_r4 -r 4 --iterations 1500 --skip_lpips     --reflection_args "--iterations 300 --test_interval 300 --checkpoint_interval 300 --densify_until_iter 250"
```

| Bước | Kết quả |
|---|---|
| Phase I, 1500 vòng | 2 phút, VRAM đỉnh 577 MB; PSNR test của B so với ảnh clean 26.7 dB (linear); IoU của glass map 0.75 |
| Trích kính | 4 patch cho hộp kính nhiều mặt, coverage 0.72, precision 0.62 (được chấp nhận) |
| Che khuất | Bỏ lớp đĩa trên kính khỏi lượt render che khuất làm tỉ lệ báo nhầm trong mask giảm từ 3.9% xuống 2.2% (mục 6.3) |
| Phase II, 300 vòng | 6.5 phút; PSNR test vùng kính (linear) 25.95 dB với direct, 27.63 dB với composite |
| Render | Khoảng 0.86 s/frame khi GPU bị hạ xung (mục 13) |

Đây chỉ là kiểm tra tích hợp (số vòng rất ít, độ phân giải thấp), không phải kết quả để báo cáo.

