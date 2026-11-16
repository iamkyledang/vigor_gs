# VIGOR-GS — Ghi chú triển khai và kết quả chạy thử

**Cập nhật:** 2026-11-16 · **Nhánh:** `viet_huynh` · **Commit mới nhất được mô tả:** `33d709d`

Tài liệu này ghi lại ba việc: repo đã thay đổi những gì so với baseline RT-Splatting, kết quả các lượt chạy thử, và những điều ghi nhận được trong quá trình triển khai. Đặc tả phương pháp nằm ở `VIGOR_GS_Academic_Framework (2).pdf` và `reference.md`; hướng dẫn chạy nằm ở `README.md`.

## 1. Tóm tắt

- Pipeline VIGOR-GS đã chạy trọn từ đầu đến cuối: Phase I → trích glass plane → Phase II → render → metrics. Bộ test có 100 test, đều qua.
- **Chưa có bằng chứng reflection field cải thiện kết quả.** Trên scene thử, composite thấp hơn ảnh direct một chút (PSNR 26.12 so với 26.60). Nguyên nhân nằm ở dữ liệu và điều kiện chạy, xem mục 5.
- Ba điểm cần giải quyết trước khi có số liệu dùng được: ảnh clean, giả định một glass plane, và máy đủ mạnh để train đủ dài.

## 2. Repo đã thay đổi những gì

Baseline RT-Splatting được giữ nguyên để so sánh: `train.py`, `render.py`, `scene/gaussian_model.py`, `gaussian_renderer/__init__.py`, `eval.sh`. Code VIGOR-GS nằm ở file mới.

### 2.1. Các commit

| Commit | Module | Nội dung |
|---|---|---|
| `5d25843` | 00 | Hạ tầng pytest (`tests/`, `pytest.ini`), đưa tài liệu đặc tả vào repo |
| `cc174d4` | 01 | Loader cho layout dữ liệu VIGOR-GS |
| `36e1c7c` | 02 | Deformation MLP dùng chung cho hai field |
| `1917854` | 03 | Interface field (2DGS, `p` và `o`) và renderer hai pass |
| `d1afa54` | 04 | `train_interface.py` — Phase I |
| `38e68d9` | 05 | `extract_plane.py` — trích glass plane |
| `af5ecea` | 06 | Camera ảo đối xứng qua plane |
| `bcb3a65` | 07 | Reflection field (3DGS), vendor rasterizer 3DGS |
| `a00b76a` | 08 | Fresnel (Schlick) và phép ghép ảnh |
| `80f8d25` | 09 | `train_reflection.py` — Phase II |
| `80852ac` | 10 | `render_vigor.py`, dùng lại `metrics.py` |
| `33d709d` | 11 | Năm công tắc ablation, cập nhật README và CLAUDE.md |

### 2.2. File mới

| File | Vai trò |
|---|---|
| `train_interface.py` | Phase I: train `G_I` và `D_φI` trên ảnh clean |
| `extract_plane.py` | Fit glass plane từ surface pass, ghi `glass_plane.json` |
| `train_reflection.py` | Phase II: train `G_R` và `D_φR` trên ảnh gốc, Phase I đóng băng |
| `render_vigor.py` | Inference, xuất ảnh theo layout của `metrics.py` |
| `scene/gaussian_field.py` | Lớp nền chung cho hai field: tham số, optimizer, clone/split/prune, PLY, checkpoint |
| `scene/interface_model.py` | `InterfaceGaussianModel`: đĩa 2D với occupancy `p` và opacity `o` |
| `scene/reflection_model.py` | `ReflectionGaussianModel`: ellipsoid 3D, khởi tạo ngẫu nhiên phía camera |
| `scene/deform_model.py` | `DeformNetwork`, `DeformModel`, `apply_deformation` |
| `gaussian_renderer/interface.py` | `render_interface`: volume pass (`p·o`) và surface pass (`p`) |
| `gaussian_renderer/reflection.py` | `render_reflection` bằng rasterizer 3DGS |
| `gaussian_renderer/vigor.py` | `render_direct`, `render_composite`: forward pass đầy đủ cho một camera |
| `utils/plane_utils.py` | RANSAC + SVD, định hướng plane, lưu/nạp plane |
| `utils/virtual_camera.py` | `reflect_camera`, ma trận phản xạ |
| `utils/fresnel.py` | Schlick, `OpticalCorrection` (`h_ψ`), `BlendModel`, `composite` |
| `utils/vigor_utils.py` | Bố cục thư mục kết quả, `load_interface`, `save_reflection`, `load_reflection` |
| `submodules/diff-gaussian-rasterization/` | Rasterizer 3DGS chính thức của Inria (commit `59f5f77`) |
| `tests/` | 13 file test, 100 test |

### 2.3. File baseline bị sửa

Các thay đổi dưới đây đều giữ nguyên hành vi của baseline, trừ khi ghi rõ.

| File | Thay đổi |
|---|---|
| `scene/dataset_readers.py` | Thêm `readVigorSceneInfo`; `CameraInfo` có thêm `time`, `clean_image` (có giá trị mặc định) |
| `scene/__init__.py` | Nhận diện layout VIGOR-GS khi có `colmap/frames.json` |
| `scene/cameras.py`, `utils/camera_utils.py` | `Camera` có thêm `time`, `clean_image`, `has_clean_image`; tuỳ chọn đổi sang linear RGB |
| `arguments/__init__.py` | Thêm tham số vào `ModelParams`; thêm `InterfaceOptimizationParams`, `ReflectionOptimizationParams`; `get_combined_args` nhận `cfg_subdir`; tham số thiếu trong `cfg_args` cũ lấy giá trị mặc định |
| `utils/color_utils.py` | **Sửa lỗi** công thức `srgb2linear` (xem mục 6.1) |
| `README.md`, `.gitignore` | Thêm hướng dẫn pipeline mới, bỏ qua `.pytest_cache` |

Rasterizer 3DGS được sửa hai chỗ so với bản gốc: thêm `#include <cstdint>` trong `cuda_rasterizer/rasterizer_impl.h` (cần cho gcc 13) và `setup.py` trỏ tới GLM có sẵn ở `submodules/diff-surfel-anych/third_party/glm`.

### 2.4. Bố cục kết quả

```
output/<scene>/
├── interface/            Phase I: cfg_args, point_cloud/iteration_N/, deform/iteration_N/
├── glass_plane.json      n, d và thống kê của phép fit
├── glass_points.ply      điểm kính dùng để fit (inlier xanh, outlier đỏ)
├── reflection/           Phase II: cfg_args, point_cloud/, deform/, blend/
├── test/ours_<iter>/     renders, gt, transparent_masks, vis
├── test/direct_<iter>/   kết quả của render_vigor.py --direct_only
└── results.json, per_view.json
```

## 3. Các quyết định thiết kế

### 3.1. Do nhóm chốt

| Vấn đề | Quyết định |
|---|---|
| PDF và `reference.md` mâu thuẫn về plane và mask | Theo `reference.md`: plane trích xuất từ `G_I`, có dùng glass mask |
| Nguồn ảnh clean | Sẽ cung cấp sau; tạm chạy với `--clean_fallback_original` |
| Nơi chạy | WSL Ubuntu trên máy này |
| Deformation MLP | Có ngay từ đầu cho cả hai field |
| Hộp kính nhiều mặt ở `scene_1` | Tạm để nguyên giả định một plane |
| Ablation "không tách trường phản chiếu" | Không dùng ảnh clean, một field học ảnh thường; thêm một ablation "không có `G_R` riêng" |
| Ablation "không Fresnel" | Một hằng số học được cho cả scene |

### 3.2. Do người triển khai chọn, cần nhóm xem lại

1. **Mask loss giữ công thức RT-Splatting**: `BCE(1 − surface_opacity, M_t)`. Công thức `M̂ = 1 − Π(1 − p·g)` trong `guildance.md` ép `p → 0` ngoài vùng kính, mà effective opacity là `p·o`, nên vật đục cũng biến mất.
2. **`G_R` nằm ở toạ độ thế giới thật**, render bằng view matrix `W2C · M`. Ảnh phản xạ khớp pixel với camera thật; không lật ảnh và không phản chiếu Gaussian.
3. **`G_R` khởi tạo ngẫu nhiên**: 100.000 điểm trong quả cầu bán kính `3 × cameras_extent` quanh camera, chỉ giữ phía camera của plane. Cảnh phản xạ không có điểm SfM.
4. **α được giới hạn bằng glass map render từ `G_I`** (`1 − surface_opacity`), dùng được cả lúc inference. Có tuỳ chọn `--gt_glass_mask`.
5. **Train trên linear RGB** (mặc định của `train_interface.py`, tắt bằng `--srgb`); ảnh lưu ra được đổi về sRGB.
6. **Scale được deform trong không gian log**, nên scale sau deform luôn dương. Deformable-3DGS gốc cộng trực tiếp vào scale.
7. **`h_ψ` nhận đầu vào là điểm trên kính, cos góc nhìn và thời gian**, không phải toạ độ pixel. PDF chỉ ghi `h_ψ(u, t)`.
8. **Trích plane dùng median depth** của surface pass, để điểm nằm trên một bề mặt chứ không nằm giữa kính và vật phía sau.
9. **Điểm dùng để fit plane được lọc ba lớp**: pixel phải thuộc mask kính `M_t`, có surface alpha > 0.5, và có glass map > 0.5 (field tự coi đó là kính). Mỗi view lấy tối đa 2000 điểm.
10. **Các trọng số loss lấy theo baseline** (`norm 0.05`, `mask 0.01`, `occupancy decay 0.001`, `λ_dssim 0.2`); `λ_motion = 0.01` và `λ_opt = 0.01` là giá trị đặt tạm, chưa tinh chỉnh.

## 4. Dữ liệu

Ba bộ trong `data/datasets/datasets/` cùng một layout (`colmap/`, `train/`, `test/`, `glint/diffrens/`).

| Bộ | Ảnh COLMAP | Mask | Ghi nhận |
|---|---|---|---|
| 3D-FRONT-T | 960×540, PINHOLE | 960×540, khớp ảnh | Dùng được ngay. `scene_1`: 137 train / 34 test |
| VSGD | 1927×1080 (đã undistort) | 1920×1080 (ảnh gốc) | Mask lệch so với ảnh COLMAP |
| WayveScene101 | 3271×1224 (đã undistort) | 1920×1080 | Mask là mask privacy/validity, không phải mask kính |

- **Không bộ nào có ảnh clean.** `glint/diffrens/` có `basecolor`, `diffuse_albedo`, `depth`, `normal` nhưng chỉ cho frame train, và đó là albedo chứ không phải ảnh đã loại reflection.
- Tên ảnh trong `images.bin` khớp với trường `name` của `frames.json` ở cả ba bộ. Thời gian chuẩn hoá `time ∈ [0, 1]` có sẵn trong `frames.json`.
- **Kính trong `3D-FRONT-T/scene_1` là một hộp kính** (tủ trưng bày chứa quả bóng), camera đi vòng quanh. Scene này vi phạm giả định một glass plane.

## 5. Kết quả chạy thử

Tất cả chạy trên `3D-FRONT-T/scene_1`, độ phân giải 1/4 (240×135), deformation MLP nhỏ (`--deform_width 64 --deform_depth 4`), ảnh gốc thay ảnh clean. Đây là lượt kiểm tra kỹ thuật, không phải số liệu của phương pháp.

### 5.1. Phase I — 2500 iteration, warm-up 1500

| Iteration | PSNR test | PSNR train | Glass IoU test | Glass IoU train |
|---|---|---|---|---|
| 500 | 22.85 | 24.08 | 0.088 | 0.119 |
| 1500 | 27.11 | 27.99 | 0.523 | 0.657 |
| 2500 | 28.14 | 29.84 | 0.649 | 0.797 |

PSNR ở bảng này tính trong linear RGB so với target của Phase I. Số Gaussian tăng từ 54.085 (điểm SfM) lên khoảng 138.000.

### 5.2. Glass plane

| Đại lượng | Giá trị |
|---|---|
| `n` | (−0.9666, 0.2549, −0.0259) |
| `d` | −1.4587 |
| Số điểm / số view | 259.607 / 137 |
| Ngưỡng inlier | 0.0428 (1% `cameras_extent`) |
| Tỉ lệ inlier | 0.405 |
| RMSE trên inlier | 0.0211 |
| Camera ở phía dương của plane | 86% |

Plane tìm được là một mặt của hộp kính. Chỉ 40% điểm kính nằm trên nó, và 14% camera nằm phía sau, nên với các camera đó phép phản xạ không có nghĩa vật lý.

### 5.3. Phase II — 600 iteration, warm-up 300, 20.000 điểm khởi tạo

PSNR trong linear RGB so với ảnh gốc:

| | Toàn ảnh | Vùng kính |
|---|---|---|
| Chỉ ảnh direct `B_t` | 28.14 | 35.71 |
| Composite, iteration 1 | 26.12 | 33.21 |
| Composite, iteration 600 | 27.53 | 35.86 |

Số liệu lấy từ hai lượt chạy cùng cấu hình (dòng iteration 1 từ lượt đầu, dòng 600 từ lượt sau; hai lượt lệch nhau dưới 0.02 dB). Khởi tạo ngẫu nhiên màu xám làm composite tụt 2 dB ở iteration đầu; sau 600 iteration phần lớn mức tụt này được lấy lại. Vùng kính nhỉnh hơn direct 0.15 dB, toàn ảnh vẫn thấp hơn 0.6 dB.

### 5.4. `metrics.py` trên tập test (34 ảnh, sRGB)

| Phương án | PSNR | SSIM | LPIPS | PSNR kính | SSIM kính | LPIPS kính |
|---|---|---|---|---|---|---|
| `direct_2500` (chỉ `G_I`) | 26.60 | 0.888 | 0.115 | 32.04 | 0.953 | 0.049 |
| `ours_600` (composite) | 26.12 | 0.879 | 0.122 | 31.93 | 0.949 | 0.052 |

Composite kém direct ở mọi chỉ số. Lưu ý PSNR ở đây thấp hơn mục 5.1 và 5.3 vì `metrics.py` đo trên ảnh sRGB 8-bit, còn lúc train đo trên linear RGB.

### 5.5. Vì sao composite chưa hơn direct

1. **Phase I đã học cả reflection.** Target của Phase I là ảnh gốc, nên `B_t` tái dựng luôn reflection. Phần còn lại cho `G_R` giải thích gần như bằng không, và mọi thứ `G_R` thêm vào đều là nhiễu.
2. **Plane sai với phần lớn kính.** Camera ảo chỉ đúng với một mặt hộp.
3. **Phase II quá ngắn.** 600 iteration chưa đủ để `G_R` hội tụ, nhất là khi gradient bị nhân với α nhỏ (mục 6.3).

Vì vậy lượt chạy này chỉ xác nhận pipeline đúng về mặt kỹ thuật. Bằng chứng duy nhất rằng Phase II học được reflection là test tổng hợp `test_reflection_field_learns_the_reflection_while_the_interface_stays_fixed`: trên cảnh có reflection biết trước, PSNR của composite tăng hơn 5 dB trong khi tham số interface không đổi.

### 5.6. Tốc độ và bộ nhớ (RTX 3050 Laptop 4GB, qua WSL)

| Cấu hình | Tốc độ |
|---|---|
| Phase I, tĩnh, ~54k Gaussian | 6–10 it/s |
| Phase I, deformation 64×4, ~130k Gaussian | 1.2–1.6 it/s |
| Phase I, deformation mặc định 256×8, ~54k Gaussian | khoảng 1 it/s (đo đúng lúc GPU đang chậm) |
| Bộ test đầy đủ (100 test) | khoảng 7 phút |

- VRAM khi train ở 1/4 độ phân giải: khoảng 1.4 GB.
- Tốc độ GPU dao động mạnh giữa các lần đo. Có lúc một phép nhân ma trận 54k×256×256 mất 30 ms (GPU báo ~10 W), chậm hơn mức bình thường của card này nhiều lần. Chưa xác định nguyên nhân; nghi do chế độ nguồn của laptop.
- Load 171 ảnh từ `/mnt/d` mất khoảng 30 giây mỗi lần khởi động.

## 6. Ghi nhận trong quá trình triển khai

### 6.1. Lỗi có sẵn trong baseline

| Lỗi | Trạng thái |
|---|---|
| `srgb2linear` tính `(x + 0.055/1.055)^2.4` thay vì `((x + 0.055)/1.055)^2.4`, cho giá trị > 1 khi `x` gần 1 | Đã sửa. Baseline không gọi hàm này |
| Loader Blender không gán `transparent_mask`, crash trong `loadCam` | Chưa sửa |
| `capture()`/`restore()` của `GaussianModel` bỏ sót `_opacity`, `_roughness`, `_reflectance`, feature, MLP và encoding, nên resume từ `.pth` không đầy đủ | Chưa sửa. Lớp `GaussianField` mới lưu đủ |

### 6.2. Lỗi phát sinh và đã sửa khi làm

- `DeformNetwork` với `depth ≤ 2` đặt skip connection sau lớp cuối, sai kích thước đầu vào của các head.
- `get_combined_args` bỏ các tuỳ chọn để ở `None`, nên script phải đọc cờ tuỳ chọn bằng `getattr`.
- `cfg_args` lưu trước khi thêm tham số mới làm thiếu thuộc tính khi nạp lại; đã sửa trong `ParamGroup.extract`.

### 6.3. Điểm cần lưu ý về phương pháp

1. **Vị trí kính trong Phase I chỉ được ràng buộc yếu.** Với ảnh clean, kính không hiện trong ảnh. Tín hiệu duy nhất cho hình học kính là mask loss (buộc có một lớp `p` cao, `o` thấp che vật phía sau trong surface pass) và normal consistency. Độ sâu của lớp đó chỉ được xác định qua việc biên mask phải nhất quán giữa các view. RT-Splatting định vị kính nhờ nhánh specular, thứ VIGOR-GS không có ở Phase I. Đây là rủi ro chính cho độ chính xác của plane.
2. **Schlick thuần cho α rất nhỏ**: khoảng 0.04 khi nhìn vuông góc với chiết suất 1.0/1.5. Hệ quả:
   - `R_t` phải sáng hơn nhiều lần giá trị [0, 1] thông thường mới tạo ra reflection nhìn thấy được.
   - Gradient về `G_R` bị nhân với α, nên ngưỡng densify `0.0002` của 3DGS hiếm khi đạt tới.
   - Hướng thử: bật `--optical_correction`, hạ `--densify_grad_threshold`, hoặc tăng learning rate của màu.
3. **Glass map rò ra ngoài vùng kính.** Glass map bắt đúng hộp kính nhưng có giá trị khác 0 ở cạnh vật và vùng tối, nên reflection bị cộng nhẹ ra ngoài kính. Ở vùng không có bề mặt nào, `surface_opacity = 0` nên glass map bằng 1.
4. **Khởi tạo `G_R` màu xám 0.5 làm giảm chất lượng lúc đầu.** Khởi tạo tối hơn có thể tốt hơn, nhưng màu đúng bằng 0 dễ bị kẹt ở ngưỡng clamp của SH.
5. **`D_φI` chưa được kiểm chứng trên cảnh động.** `scene_1` là cảnh tĩnh; deformation chỉ được test về mặt cơ chế.
6. **Rasterizer xử lý đúng view matrix có định thức −1.** Test so sánh "render qua camera ảo" với "render cảnh đã lấy đối xứng qua camera thật" khớp nhau cho cả rasterizer 3DGS và 2DGS.

## 7. Trạng thái kiểm thử

- 100 test trong `tests/`, chạy bằng `python -m pytest`. Test đánh dấu `gpu` hoặc `data` tự bỏ qua khi thiếu CUDA hoặc scene phát triển.
- Lượt train ngắn dùng chung cho các test: Phase I 400 iteration và Phase II 300 iteration ở 1/8 độ phân giải.
- Ba công tắc đã có lượt train ngắn trên dữ liệu thật: `--ablate_virtual_camera`, `--ablate_fresnel`, `--shared_reflection_field`.
- **Hai công tắc chưa được train thử**: `--ablate_factorization` và `--ignore_clean_images`. Chúng chỉ có test ở mức renderer, model và loader.
- Chưa có lượt chạy nào ở độ phân giải đầy đủ, với 30.000 iteration mặc định, hay với deformation MLP mặc định (256×8).
- VSGD và WayveScene101 chưa được chạy.

## 8. Việc tiếp theo

1. **Cung cấp ảnh clean** vào `<scene>/clean/`, cùng tên file với `colmap/images/`. Đây là điều kiện để đánh giá được Phase II.
2. **Chọn hướng cho scene nhiều mặt kính**: mở rộng sang nhiều plane, hoặc đánh giá trên scene có một tấm kính phẳng.
3. **Chạy đủ dài trên máy mạnh hơn**, ở độ phân giải đầy đủ, rồi chạy năm ablation, mỗi cái một thư mục `-m` riêng.
4. **Xử lý mask của VSGD** (undistort mask cho khớp ảnh COLMAP) và tìm mask kính cho WayveScene101.
5. **Tinh chỉnh Phase II** theo mục 6.3: mức α, ngưỡng densify, khởi tạo `G_R`, `λ_motion`, `λ_opt`.
