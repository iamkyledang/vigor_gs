# VIGOR-GS: bộ ablation đề xuất

Mỗi ablation bỏ hoặc thay **một** thành phần của framework (`docs/framework.pdf`), và trả lời một câu hỏi: thành phần đó có cần thiết không, và nó đóng góp vào đâu. Tất cả đều chạy được bằng `run_ablations.py`. Ablation nào chỉ đổi Phase II hoặc luật chọn kính sẽ **chép Phase I của `full`**, nên khác biệt giữa nó và `full` chỉ đến từ thay đổi đó.

```bash
python run_ablations.py -s data/datasets/datasets/3D-FRONT-T/scene_1 --root output/3dfront_ablations -r 2
python run_ablations.py -s <scene> --root <runs> --variants full,no_occlusion,pure_fresnel -r 2   # một phần của bộ
```

Kết quả nằm ở `<root>/summary.csv`: PSNR, SSIM, LPIPS, Glass PSNR, Opaque PSNR, Transmission PSNR, FPS, thời gian train, và chênh lệch PSNR / Glass PSNR so với `full`.

## Nên đọc metric nào

- **Glass PSNR** (so với ảnh gốc, chỉ trên pixel kính) đo khả năng tái tạo phản chiếu. Hầu hết các ablation của Phase II chỉ thay đổi metric này, vì ngoài kính α = 0.
- **Transmission PSNR** (B so với ảnh clean) đo chất lượng của lớp không phản chiếu, tức chính sự tách lớp. Đây là thước đo phân biệt VIGOR-GS với các phương pháp chỉ khớp ảnh gốc.
- **PSNR** toàn ảnh so với ảnh gốc thì cần có, nhưng luôn ưu tiên những biến thể khớp thẳng ảnh gốc (`no_separation`, RT-Splatting). Không nên dùng riêng nó để kết luận về việc tách lớp.

## Danh sách

| Biến thể | Thay đổi | Phase I | Câu hỏi | Kỳ vọng |
|---|---|---|---|---|
| `full` | Không | riêng | Mốc so sánh | |
| **Biểu diễn interface** | | | | |
| `no_separation` | `--ignore_clean_images`: một field khớp ảnh gốc, không có Phase II, render `B` | riêng | Có cần tách hai trường không? | PSNR so với ảnh gốc gần RT-Splatting; Transmission PSNR thấp vì phản chiếu bị "nướng" vào B |
| `no_factorization` | `--ablate_factorization`: p ≡ 1, chỉ có o | riêng | Tách p·o có giúp hình học kính không? | Trích patch kém hơn (coverage/precision thấp hơn), có thể không qua ngưỡng chất lượng |
| `static_interface` | `--static_interface`: không có D_I | riêng | Deformation của G_I có cần cho cảnh động không? | Kém hơn trên Wayve (xe và người chuyển động); gần như bằng `full` trên cảnh tĩnh |
| **Chọn kính (Eq. 9)** | | | | |
| `no_occlusion` | `--ablate_occlusion`: Eq. 9 nguyên văn, không kiểm tra che khuất | của `full` | Kiểm tra che khuất có cần không? | Glass PSNR giảm ở những scene có vật đứng trước kính; tỉ lệ "occluded" trong `glass_preview/` cho thấy mức độ |
| `single_patch` | `--max_patches 1` | của `full` | Mở rộng nhiều patch có cần không? | Kém hơn rõ trên 3D-FRONT-T scene_1 (hộp kính nhiều mặt) và các mặt tiền nhiều ô kính |
| **Tia và camera ảo (Eq. 5, 9a, 9b)** | | | | |
| `no_virtual_camera` | `--ablate_virtual_camera`: G_R được nhìn theo tia thật, xuyên thẳng qua kính | của `full` | Phản xạ có cần đi qua camera ảo không? | Glass PSNR giảm; G_R phải "vẽ" phản chiếu như một lớp phía sau kính, nên không nhất quán giữa các view |
| `pinhole_virtual_camera` | `--pinhole_virtual_camera`: mọi tia của một vùng bắt đầu tại `c_v` của V_dyn (Eq. 9b), thay vì tại từng điểm kính | của `full` | Tia đúng có tốt hơn xấp xỉ pinhole mà PDF nhắc tới không? | Kém hơn với vật phản chiếu ở gần kính (thị sai); gần bằng khi chúng ở xa |
| **Trọng số phản xạ (Eq. 10–11)** | | | | |
| `pure_fresnel` | `--pure_fresnel`: h_ψ = 0 | của `full` | Hiệu chỉnh h_ψ đóng góp bao nhiêu? | Glass PSNR giảm khi phản chiếu thật mạnh hơn Schlick với F0 = 0.04 (kính phủ, kính bẩn) |
| `no_fresnel` | `--ablate_fresnel`: một hằng số học được thay cho F(θ) (h_ψ vẫn cộng lên) | của `full` | Phụ thuộc góc của Schlick có quan trọng không? | Kém hơn ở các view nghiêng; trước đây trên VSGD 0758, hằng số lại hơn Schlick, nên cần đo lại |
| **Động học và loss chuyển động (Eq. 1, 7b)** | | | | |
| `static_reflection` | `--static_reflection`: không có D_R | của `full` | G_R có cần thay đổi theo thời gian không? | Kém hơn khi phản chiếu có vật chuyển động (Wayve) |
| `no_motion_loss` | `--lambda_motion 0` | của `full` | L_motion có giúp không? | Glass PSNR gần bằng; ổn định theo thời gian (nhấp nháy giữa các frame) kém hơn, nên xem `comparison.mp4` |

## Những ablation nên chạy thêm nếu có điều kiện

- **Hình học kính biết trước và trích được** (`--glass_geometry`). Với 3D-FRONT-T, kính có thể lấy từ mesh của 3D-FRONT. So `full` (trích từ Phase I) với hình học biết trước cho biết bao nhiêu sai số đến từ việc trích patch. Khi có hình học biết trước, có thêm ablation `--lambda_glass 0` (không có prior mask), tức Eq. 3 nguyên văn.
- **Ảnh clean.** DAI và DiffRS (`CLEAN_TAG`). B bị buộc theo ảnh clean, nên chất lượng ảnh clean giới hạn trực tiếp Transmission PSNR và Opaque PSNR.
- **Cùng ngân sách iteration với RT-Splatting** (61k): `--iterations 30500` cho mỗi phase, để so sánh công bằng.
- **Độ nhạy của trọng số loss:** `--lambda_motion` ∈ {0.03, 0.1, 0.3}, `--lambda_optical` ∈ {0.001, 0.01, 0.1}. PDF yêu cầu chọn theo validation; giá trị hiện tại chỉ là đặt tạm.

## Cách thêm một ablation

Thêm một cờ vào nhóm tham số phù hợp trong `arguments/__init__.py`. Cờ phải được lưu vào config của bước đó, để `render.py` dùng lại đúng cờ đó. Sau đó thêm một dòng vào `VARIANTS` trong `run_ablations.py`: `(nguồn Phase I, cờ model, cờ kính, cờ Phase II)`. Một biến thể đổi Phase I phải có nguồn Phase I là chính nó.
