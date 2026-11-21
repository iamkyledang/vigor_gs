# Tổng hợp và phân tích kết quả: VIGOR so với các baseline

Cập nhật: 21/11/2026. Nguồn: ba thư mục kết quả trên Google Drive (Deformable 3DGS, SpectroMotion, VIGOR + RT-Splatting).

## Tóm tắt

- **Deformable 3DGS là baseline mạnh nhất.** Nó có PSNR và SSIM cao nhất trên Wayve 001 và 3D-FRONT 1. Trên VSGD 0758 nó chỉ đứng sau RT-Splatting. VIGOR bản đầy đủ thua Deformable 3DGS trên cả ba scene: −0.85 dB (VSGD), −1.52 dB (Wayve), −1.66 dB (3D-FRONT).
- **So với RT-Splatting, VIGOR chỉ thắng trên Wayve:** toàn ảnh +1.30 dB (thắng 30/40 view), vùng kính +0.61 dB. Trên 3D-FRONT VIGOR thua (−0.93 dB toàn ảnh, −1.12 dB vùng kính). Trên VSGD cũng thua (−1.69 dB toàn ảnh, −2.02 dB vùng kính).
- **Phase II đang làm giảm điểm.** Chỉ dùng Phase I (biến thể `no_separation`) cho điểm cao hơn bản đầy đủ trên VSGD (+1.35 dB) và 3D-FRONT (+0.36 dB).
- **Ablation:** factorization p·o và camera ảo đều có ích trên cả ba scene. **Fresnel thì không:** bỏ Fresnel làm điểm tăng trên VSGD (+1.03 dB) và 3D-FRONT (+0.32 dB).
- ⚠️ **Trên VSGD và Wayve, các so sánh với Deformable 3DGS và SpectroMotion chưa cùng độ phân giải:** hai method này chấm ở độ phân giải gốc, còn VIGOR và RT-Splatting ở `-r 2`. Xem mục 1.4; cần chấm lại bằng `notebooks/colab_masked_eval_baselines.ipynb`.
- **Ba chỗ cần xử lý trước khi đưa số vào paper:**
  - `no_separation` trên Wayve bị hỏng (5.84 dB).
  - SpectroMotion trên VSGD 0758 có thể đã chấm trên dữ liệu khác.
  - `summary.csv` của VSGD 0758 đã cũ.

---

## 1. Nguồn dữ liệu và cách tổng hợp

### 1.1 Các thư mục trên Drive

| Method | Đường dẫn trong thư mục Drive | File đã đọc |
|---|---|---|
| Deformable 3DGS | `output/vsgd/<id>/`, `output/wayvescene101/scene_001/`, `output/3d-front-t/scene_<n>/` | `results.json`, `per_view.json`, `dataset_run.json`, `cfg_args` |
| SpectroMotion | `output/vsgd/0758/`, `output/.../scene_001/`, `output/3d-front-t/scene_1/` | `results.json`, `per_view.json`, `cfg_args` |
| VIGOR + RT-Splatting | `VSGD_0758/<biến thể>/`, `WayveScene101_scene_001_dai/<biến thể>/`, `3D-FRONT-T_scene_1_dai/<biến thể>/` | `summary.csv`, `results.json`, `per_view.json` |

Trong thư mục VIGOR, biến thể `rtsplatting` là baseline RT-Splatting. Chỉ ba scene sau có kết quả của đủ bốn method: **VSGD 0758, WayveScene101 scene_001, 3D-FRONT-T scene_1.**

### 1.2 Kiểm tra các method có cùng dữ liệu test không

- **Deformable 3DGS:** `dataset_run.json` lưu hash của `split.json` và `frames.json`, cùng kích thước file ảnh và file COLMAP mà nó đã train.

  | Scene | Kết quả so với dữ liệu local đã căn chỉnh |
  |---|---|
  | VSGD 0758 | Hash `split.json` (`f0bfeb00…`) và `frames.json` (`647eb45e…`) khớp. `colmap/images/00000000.jpg` cùng 840,647 byte; `images.bin` cùng 10,564,186 byte |
  | Wayve 001 | Hash `split.json` (`e615cafe…`) và `frames.json` (`317191df…`) khớp. `colmap/images/00000000.jpeg` cùng 2,333,825 byte; `images.bin` cùng 36,694,008 byte |

  Như vậy Deformable 3DGS train và test trên đúng ảnh `colmap/` đã undistort và cùng split với VIGOR.
- **Số view test giống nhau ở cả bốn method:** 29 (VSGD 0758), 40 (Wayve 001), 34 (3D-FRONT 1).
- **SpectroMotion:** không có `dataset_run.json`. `cfg_args` cho thấy nó đọc `<scene>/images` với `resolution=1`, `eval=True`. Riêng VSGD 0758 cần lưu ý, xem mục 5.2.

### 1.3 Metric

- **PSNR, SSIM, LPIPS** tính trên toàn ảnh của test split.
- **Masked PSNR/SSIM/LPIPS** chỉ tính trong mask kính. Chỉ VIGOR và RT-Splatting có metric này, vì cả hai đều được chấm bằng pipeline của VIGOR.
- **So từng view (paired):** với mỗi cặp method, lấy PSNR của từng view chung, rồi tính chênh lệch trung bình và số view mà VIGOR cao hơn.

### 1.4 Số iteration và độ phân giải (không đồng đều giữa các method)

| Method | Checkpoint được chấm | Độ phân giải chấm |
|---|---|---|
| Deformable 3DGS | `ours_40000` | gốc (`resolution=1`): VSGD 1927×1080, Wayve 3271×1224, 3D-FRONT 960×540 |
| SpectroMotion | `ours_40000` | gốc (`resolution=1`) |
| VIGOR (Phase II) | `ours_30000` (biến thể `no_separation` dùng `direct_30000`) | `-r 2` cho VSGD (964×540) và Wayve (1636×612); `-r 1` cho 3D-FRONT |
| RT-Splatting | `ours_61000` | giống VIGOR |

> ⚠️ **Lệch độ phân giải:** trên VSGD 0758 và Wayve 001, Deformable 3DGS và SpectroMotion được chấm ở độ phân giải gốc, còn VIGOR và RT-Splatting ở `-r 2`. Vì vậy các so sánh toàn ảnh giữa hai nhóm này ở mục 2 **chưa cùng điều kiện**. Chỉ có 3D-FRONT 1 là cùng độ phân giải. Notebook `notebooks/colab_masked_eval_baselines.ipynb` (script `eval_masked_baselines.py`) chấm lại mọi method ở cùng độ phân giải và trên cùng ảnh tham chiếu `colmap/`, đồng thời tính các metric vùng kính.

---

## 2. So sánh chính

### 2.1 Toàn ảnh

| Scene | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| **VSGD 0758** | Deformable 3DGS | 33.94 | **0.9463** | 0.2593 |
| | SpectroMotion ⚠️ | 32.43 | 0.9377 | 0.2872 |
| | RT-Splatting | **34.77** | 0.9451 | **0.1671** |
| | VIGOR đầy đủ | 33.09 | 0.9390 | 0.2109 |
| | VIGOR chỉ Phase I (`no_separation`) | 34.44 | 0.9439 | 0.2001 |
| **Wayve 001** | Deformable 3DGS | **26.40** | **0.8911** | 0.1667 |
| | SpectroMotion | 26.02 | 0.8808 | 0.1916 |
| | RT-Splatting | 23.58 | 0.8689 | **0.1492** |
| | VIGOR đầy đủ | 24.88 | 0.8745 | 0.1666 |
| | VIGOR đầy đủ + MLP Fresnel | 24.97 | 0.8746 | 0.1665 |
| **3D-FRONT 1** | Deformable 3DGS | **30.95** | **0.8650** | **0.2197** |
| | SpectroMotion | 30.27 | 0.8579 | 0.2412 |
| | RT-Splatting | 30.22 | 0.8575 | 0.2295 |
| | VIGOR đầy đủ | 29.29 | 0.8371 | 0.2696 |
| | VIGOR đầy đủ + MLP Fresnel | 29.41 | 0.8386 | 0.2686 |
| | VIGOR chỉ Phase I (`no_separation`) | 29.65 | 0.8445 | 0.2649 |

### 2.2 Vùng kính (chỉ VIGOR và RT-Splatting)

| Scene | Method | Masked PSNR ↑ | Masked SSIM ↑ | Masked LPIPS ↓ |
|---|---|---|---|---|
| VSGD 0758 | RT-Splatting | **36.26** | **0.9725** | **0.0758** |
| | VIGOR đầy đủ | 34.24 | 0.9690 | 0.0961 |
| | VIGOR chỉ Phase I | 35.62 | 0.9697 | 0.0958 |
| Wayve 001 | RT-Splatting | 30.27 | **0.9553** | **0.0432** |
| | VIGOR đầy đủ | 30.89 | 0.9528 | 0.0527 |
| | VIGOR đầy đủ + MLP Fresnel | **31.15** | 0.9529 | 0.0529 |
| 3D-FRONT 1 | RT-Splatting | **36.06** | **0.9429** | **0.0750** |
| | VIGOR đầy đủ | 34.94 | 0.9320 | 0.0908 |
| | VIGOR chỉ Phase I | 35.21 | 0.9371 | 0.0873 |

Trên Wayve, VIGOR có PSNR vùng kính cao hơn RT-Splatting, nhưng LPIPS vùng kính vẫn kém hơn (0.053 so với 0.043).

### 2.3 So từng view: VIGOR đầy đủ so với từng method

Cột "Tương quan" là hệ số tương quan PSNR từng view giữa hai method. Hai method chấm trên cùng tập ảnh thì hệ số này thường cao. Nếu thấp bất thường, có thể ảnh test đã khác nhau.

| Scene | So với | ΔPSNR trung bình | ΔPSNR trung vị | VIGOR thắng | Tương quan |
|---|---|---|---|---|---|
| VSGD 0758 | Deformable 3DGS | −0.85 | −0.84 | 8/29 | 0.72 |
| | SpectroMotion | +0.66 | +0.71 | 17/29 | **0.47** ⚠️ |
| | RT-Splatting | −1.69 | – | – | – |
| Wayve 001 | Deformable 3DGS | −1.52 | −1.20 | 12/40 | 0.83 |
| | SpectroMotion | −1.15 | −0.64 | 6/40 | 0.84 |
| | RT-Splatting | **+1.30** | +1.01 | **30/40** | 0.88 |
| | RT-Splatting (vùng kính) | +0.61 | – | 28/40 | – |
| 3D-FRONT 1 | Deformable 3DGS | −1.66 | −1.44 | 0/34 | 0.95 |
| | SpectroMotion | −0.98 | −0.86 | 3/34 | 0.96 |
| | RT-Splatting | −0.93 | −0.82 | 4/34 | 0.94 |
| | RT-Splatting (vùng kính) | −1.12 | – | 1/34 | – |

Với VSGD 0758, kết quả RT-Splatting được lấy từ `VSGD_0758/rtsplatting/results.json`, chấm ngày 19/11 ở commit `cb086df`. Mình chưa tải `per_view.json` của lần chạy này, nên chỉ có chênh lệch trung bình.

---

## 3. Ablation của VIGOR

Δ = PSNR của bản đầy đủ trừ PSNR của biến thể. **Δ dương nghĩa là thành phần bị bỏ có ích.** Trong ngoặc là Δ của Masked PSNR.

### 3.1 Bảng tổng hợp

| Biến thể (bỏ đi) | VSGD 0758 | Wayve 001 | 3D-FRONT 1 | Kết luận |
|---|---|---|---|---|
| (1) Factorization p·o | +1.04 (+1.53) | +2.46 (+1.32) | +0.61 (+0.56) | **Có ích trên cả ba scene** |
| (2a) Tách trường phản chiếu (`no_separation`) | **−1.35** (−1.38) | hỏng, xem 5.1 | **−0.36** (−0.27) | **Phase II đang gây hại** |
| (2b) G_R riêng (`shared_field`, chấm ở `ours_0`) | +8.57 | +1.35 | +3.13 | Không phải ablation thật, xem 5.4 |
| (3) Camera ảo | +2.88 (+3.42) | +0.27 (+0.38) | +0.34 (+0.82) | **Có ích**, rõ nhất trên VSGD |
| (4) Fresnel | **−1.03** (−1.31) | +0.03 (−0.02) | **−0.32** (−0.19) | **Fresnel không giúp, thậm chí gây hại** |

### 3.2 Số liệu đầy đủ từng scene

**VSGD 0758** (chưa có các biến thể `*_mlp`):

| Biến thể | Checkpoint | PSNR | SSIM | LPIPS | Masked PSNR | Masked SSIM | Masked LPIPS |
|---|---|---|---|---|---|---|---|
| full | ours_30000 | 33.09 | 0.9390 | 0.2109 | 34.24 | 0.9690 | 0.0961 |
| no_factorization | ours_30000 | 32.05 | 0.9404 | 0.2041 | 32.71 | 0.9679 | 0.0967 |
| no_separation | direct_30000 | **34.44** | **0.9439** | **0.2001** | **35.62** | 0.9697 | 0.0958 |
| shared_field | ours_0 | 24.52 | 0.8928 | 0.2575 | 24.77 | 0.9231 | 0.1426 |
| no_virtual_camera | ours_30000 | 30.21 | 0.9213 | 0.2423 | 30.82 | 0.9513 | 0.1274 |
| no_fresnel | ours_30000 | 34.12 | 0.9406 | 0.2060 | 35.55 | **0.9706** | **0.0915** |

**Wayve 001:**

| Biến thể | PSNR | SSIM | LPIPS | Masked PSNR | Masked LPIPS | Δ so với RT | Thắng RT |
|---|---|---|---|---|---|---|---|
| full | 24.88 | 0.8745 | 0.1666 | 30.89 | 0.0527 | +1.30 | 30/40 |
| full_mlp | **24.97** | 0.8746 | 0.1665 | **31.15** | 0.0529 | +1.39 | 33/40 |
| no_factorization | 22.41 | 0.8253 | 0.2110 | 29.57 | 0.0642 | −1.17 | 14/40 |
| no_factorization_mlp | 22.37 | 0.8226 | 0.2122 | 29.59 | 0.0647 | −1.20 | 19/40 |
| no_separation | 5.84 | 0.0079 | 0.7212 | 12.37 | 0.2809 | −17.74 | 0/40 |
| shared_field (ours_0) | 23.53 | 0.8623 | 0.1740 | 28.44 | 0.0565 | −0.05 | 23/40 |
| shared_field_mlp | 24.75 | 0.8736 | 0.1667 | 30.60 | 0.0532 | +1.18 | 30/40 |
| no_virtual_camera | 24.60 | 0.8726 | 0.1672 | 30.51 | 0.0534 | +1.02 | 29/40 |
| no_virtual_camera_mlp | 24.94 | 0.8747 | 0.1664 | 31.02 | 0.0531 | +1.36 | 32/40 |
| no_fresnel | 24.85 | **0.8751** | 0.1659 | 30.91 | 0.0526 | +1.27 | 31/40 |
| no_fresnel_mlp | 24.94 | **0.8752** | **0.1657** | 31.10 | **0.0526** | +1.37 | 33/40 |

**3D-FRONT 1:**

| Biến thể | PSNR | SSIM | LPIPS | Masked PSNR | Masked LPIPS | Δ so với RT | Thắng RT |
|---|---|---|---|---|---|---|---|
| full | 29.29 | 0.8371 | 0.2696 | 34.94 | 0.0908 | −0.93 | 4/34 |
| full_mlp | 29.41 | 0.8386 | 0.2686 | 34.89 | 0.0903 | −0.81 | 6/34 |
| no_factorization | 28.68 | 0.8314 | 0.2765 | 34.39 | 0.0950 | −1.53 | 0/34 |
| no_factorization_mlp | 28.83 | 0.8332 | 0.2747 | 34.68 | 0.0940 | −1.39 | 0/34 |
| no_separation | **29.65** | **0.8445** | **0.2649** | **35.21** | **0.0873** | −0.57 | 6/34 |
| shared_field (ours_0) | 26.16 | 0.8211 | 0.2782 | 30.45 | 0.0977 | −4.06 | 0/34 |
| shared_field_mlp | 28.35 | 0.8349 | 0.2713 | 33.16 | 0.0925 | −1.87 | 6/34 |
| no_virtual_camera | 28.95 | 0.8353 | 0.2713 | 34.13 | 0.0920 | −1.27 | 5/34 |
| no_virtual_camera_mlp | 29.39 | 0.8384 | 0.2690 | 34.80 | 0.0905 | −0.83 | 6/34 |
| no_fresnel | 29.61 | 0.8399 | 0.2674 | 35.13 | 0.0894 | −0.61 | 6/34 |
| no_fresnel_mlp | 29.58 | 0.8398 | 0.2678 | 35.03 | 0.0899 | −0.64 | 6/34 |

### 3.3 Tác dụng của MLP Fresnel

ΔPSNR = (biến thể có MLP) − (biến thể không MLP). Cột "thắng" là số view mà bản có MLP cao hơn.

| Biến thể | Wayve 001 | 3D-FRONT 1 |
|---|---|---|
| full | +0.09 (thắng 28/40) | +0.12 (thắng 21/34) |
| no_factorization | −0.04 (12/40) | +0.14 (31/34) |
| shared_field | +1.23 (40/40) | +2.20 (34/34) |
| no_virtual_camera | +0.33 (40/40) | +0.44 (32/34) |
| no_fresnel | +0.10 (37/40) | −0.03 (19/34) |

**Nhận xét:** khi có MLP Fresnel, `no_virtual_camera_mlp` gần bằng `full_mlp`:

| Scene | `full_mlp` | `no_virtual_camera_mlp` |
|---|---|---|
| Wayve 001 | 24.97 | 24.94 |
| 3D-FRONT 1 | 29.41 | 29.39 |

Như vậy MLP Fresnel bù gần hết phần mà camera ảo đóng góp. Nếu dùng MLP Fresnel làm cấu hình chính, đóng góp riêng của camera ảo sẽ khó chứng minh.

---

## 4. Phân tích

### 4.1 Vì sao VIGOR chưa vượt baseline

1. **Phase II không cộng thêm mà còn trừ.** Trên VSGD và 3D-FRONT, chỉ dùng Phase I (trường giao diện G_I, render trực tiếp) cho điểm tốt hơn bản có thêm nhánh phản xạ G_R. Điều đó nghĩa là phần phản xạ đang thêm sai số vào ảnh tổng hợp, thay vì giải thích thêm được phần nào của ảnh. Hai nguyên nhân có thể có:
   - Ảnh clean dùng làm mục tiêu cho G_I chưa sạch, khiến phần dư mà G_R phải học bị nhiễu.
   - Hệ số trộn giữa truyền qua và phản xạ (Fresnel) chưa đúng. Kết quả "bỏ Fresnel thì tốt hơn" ủng hộ khả năng này.
2. **Fresnel Schlick + h_ψ chưa hoạt động như mong đợi.** Bỏ Fresnel làm điểm tăng trên VSGD (+1.03 dB) và 3D-FRONT (+0.32 dB), còn trên Wayve thì gần như không đổi. MLP Fresnel chỉ thêm khoảng 0.1 dB. Cần kiểm tra:
   - Giá trị F (hệ số Fresnel) thực tế mà model học được trên vùng kính.
   - Liệu h_ψ có đang bị bão hoà về 0 hoặc 1 không.
3. **Deformable 3DGS thắng ở vùng ngoài kính.** Chênh lệch toàn ảnh giữa VIGOR và Deformable 3DGS (−0.85 đến −1.66 dB) lớn hơn chênh lệch giữa VIGOR và RT-Splatting trên 3D-FRONT. Có thể phần lớn khoảng thua nằm ở vùng ngoài kính, nơi VIGOR chỉ dùng G_I. Cần tính Masked PSNR cho Deformable 3DGS để xác nhận (mục 6).

### 4.2 Điểm mạnh hiện có
- **Wayve 001:** VIGOR vượt RT-Splatting rõ ràng (+1.30 dB, thắng 30/40 view; vùng kính +0.61 dB). Đây là scene có chuyển động camera lớn và kính xe, đúng bối cảnh mà camera ảo động được thiết kế cho.
- **LPIPS trên VSGD:** VIGOR có LPIPS 0.211, tốt hơn Deformable 3DGS (0.259) và SpectroMotion (0.287), dù PSNR thấp hơn.
- **Factorization p·o và camera ảo** đều có đóng góp dương, nhất quán trên cả ba scene. Đây là hai ablation dùng được cho paper.

### 4.3 So sánh giữa các baseline
- Thứ hạng giữa các baseline ổn định: Deformable 3DGS > SpectroMotion trên cả ba scene, chênh 0.4–1.5 dB.
- RT-Splatting rất mạnh ở vùng kính trên dữ liệu tổng hợp (3D-FRONT: 36.06 dB) và VSGD (36.26 dB), nhưng kém trên Wayve (23.58 dB toàn ảnh).

---

## 5. Những điểm chưa đáng tin

### 5.1 `no_separation` trên Wayve bị hỏng
PSNR chỉ 5.84 dB và SSIM 0.008, tức ảnh render gần như trống. Đây là lần chạy thất bại, không phải kết quả thật. Cần xem `log.txt` của biến thể này và chạy lại. Hiện chưa kết luận được Phase II có gây hại trên Wayve hay không.

### 5.2 SpectroMotion trên VSGD 0758
- `results.json` được tạo ngày 09/11, sớm nhất trong tất cả các kết quả.
- Không có `dataset_run.json`, nên không xác minh được nó dùng ảnh nào và split nào.
- PSNR từng view của nó chỉ tương quan 0.47 với VIGOR, trong khi Deformable 3DGS là 0.72. Trên hai scene còn lại, SpectroMotion tương quan với VIGOR ở mức 0.84 và 0.96.

Như vậy nhiều khả năng ảnh test của lần chạy này khác. Nên chạy lại SpectroMotion trên `colmap/` của VSGD 0758 trước khi đưa số vào bảng.

### 5.3 `summary.csv` của VSGD 0758 đã cũ
File này thiếu:
- Dòng RT-Splatting (đã có `rtsplatting/results.json`, chấm ngày 19/11).
- Các biến thể `*_mlp`.
- Hai cột "Δ so với RT" và "thắng/số view".

Cần chạy lại bước tổng hợp, hoặc chạy thêm các biến thể `*_mlp`, để VSGD có cùng định dạng với hai scene kia.

### 5.4 `shared_field` chấm ở `ours_0`
Biến thể không MLP được chấm ở iteration 0, nghĩa là G_R chưa train bước nào. Con số này chỉ phản ánh Phase I cộng với một trường phản xạ khởi tạo, không phải ablation "không có G_R riêng". Nên dùng `shared_field_mlp` (`ours_30000`), hoặc sửa để biến thể không MLP cũng được train đủ.

### 5.5 Iteration, độ phân giải và số lần chạy
- Số iteration khác nhau giữa các method (40k / 30k / 61k), xem mục 1.4.
- Trên VSGD và Wayve, Deformable 3DGS và SpectroMotion được chấm ở độ phân giải gốc, còn VIGOR và RT-Splatting ở `-r 2`. Cần chấm lại bằng `notebooks/colab_masked_eval_baselines.ipynb` trước khi kết luận trên hai scene này.
- Mỗi cấu hình chỉ chạy một seed, nên chênh lệch dưới khoảng 0.3 dB, như phần lớn tác dụng của MLP Fresnel, có thể chỉ là nhiễu.

---

## 6. Việc nên làm tiếp (theo thứ tự ưu tiên)

1. Chạy lại `no_separation` trên Wayve 001.
2. Chạy lại SpectroMotion trên VSGD 0758 với dữ liệu `colmap/`.
3. Chạy `notebooks/colab_masked_eval_baselines.ipynb`. Notebook này:
   - Tính metric vùng kính cho Deformable 3DGS và SpectroMotion: `Masked *` theo định nghĩa của `metrics.py`, cùng `PSNR in-mask` và `PSNR out-of-mask`.
   - Chấm lại mọi method ở cùng độ phân giải với VIGOR.
   - Kiểm tra từng method có dùng đúng ảnh `colmap/` không (`GT check PSNR`), qua đó xác minh luôn SpectroMotion trên VSGD 0758.
4. Chạy các biến thể `*_mlp` cho VSGD 0758 và cập nhật `summary.csv`, thêm dòng RT-Splatting.
5. Chẩn đoán Fresnel: in phân bố F và h_ψ trên vùng kính. So Phase I với bản đầy đủ theo từng view để tìm các view mà Phase II làm hỏng.
6. Sửa `shared_field` để G_R được train đủ, rồi chấm lại.
7. Nếu còn thời gian: chạy thêm seed cho `full` và `no_fresnel` để biết khoảng nhiễu.

---

## Phụ lục A. Deformable 3DGS trên các scene khác

VIGOR chưa chạy được các scene này vì thiếu ảnh clean. Mỗi scene chỉ có một checkpoint `ours_40000`.

| Scene | PSNR | SSIM | LPIPS |
|---|---|---|---|
| VSGD 0763 | 28.64 | 0.9063 | 0.2454 |
| VSGD 0806 | 33.36 | 0.9201 | 0.2675 |
| VSGD 0807 | 31.99 | 0.9155 | 0.3078 |
| VSGD 0936 | 33.80 | 0.9543 | 0.1234 |
| VSGD 1035 | 32.24 | 0.9109 | 0.2697 |
| 3D-FRONT 2 | 31.03 | 0.9190 | 0.1184 |
| 3D-FRONT 3 | 35.68 | 0.9540 | 0.0606 |
| 3D-FRONT 4 | 36.17 | 0.9548 | 0.0683 |
| 3D-FRONT 5 | 35.04 | 0.9622 | 0.0653 |

## Phụ lục B. Metric hình học (Chamfer, Damage Profile MAE, Normal MAE)

Ba metric này **chưa đo được** vì không có ground-truth hình học:

| Metric | Cần gì | Tình trạng |
|---|---|---|
| Chamfer (mm) | Mesh ground-truth theo đơn vị mét hoặc mm | Không scene nào có. VSGD và Wayve dùng COLMAP tự calib nên tỉ lệ tuỳ ý |
| Damage Profile MAE | Vết lõm có mặt cắt ground-truth | Không áp dụng được cho các scene kính này |
| Normal MAE | Pháp tuyến ground-truth | Không có. Normal và depth trong `glint/diffrens/` là dự đoán của DiffusionRenderer (depth tương đối) |

Các thư mục `DEPTH_GT_SHARED` trên Drive chỉ là ảnh visualize lúc train RT-Splatting.

Có thể làm proxy, nhưng phải ghi rõ trong paper là proxy:
- Normal MAE so với pseudo-GT của DiffusionRenderer.
- Chamfer với điểm COLMAP, chỉ cho 3D-FRONT 1 (scene này dùng pose do dataset cung cấp).

## Phụ lục C. Tái lập phân tích

- Script so từng view: `analyze.py` (đọc các `per_view.json` đã tải từ Drive).
- Script ghép đường dẫn Drive: `drive_tree.py`.
- Cả hai đang nằm trong scratchpad của phiên làm việc. Nếu cần giữ lâu dài, có thể chuyển vào `scripts/`.
