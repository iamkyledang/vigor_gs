# Thay đổi của `f_refactor` so với `quan_dang`

Nhánh `f_refactor` được tách từ `quan_dang` (commit `9fce787`) và refactor dựa trên `docs/framework.pdf`, có tham khảo nhánh `viet_huynh`. Tài liệu này liệt kê những gì được giữ, những gì bị sửa vì sai hoặc thiếu so với framework, và những gì được thêm. Mỗi mục tương ứng một commit trên nhánh (`git log quan_dang..f_refactor`).

## Giữ nguyên ý tưởng (và phần lớn code)

- **Renderer CUDA theo tia phản xạ** (`render_reflected_rays` trong `submodules/diff-surfel-anych/rasterize_points.cu`), cùng test so forward và gradient với bản tham chiếu PyTorch. Đây đúng là renderer "physically correct" của Sec. 3.
- Danh sách ứng viên theo tile, dùng phép chiếu qua ảnh gương của mặt kính.
- Trích patch bằng RANSAC tuần tự, thành phần liên thông và convex hull, có ngưỡng coverage/precision.
- Motion loss dạng độ thay đổi của ảnh (Eq. 7b), cả hai dự đoán đều nhận gradient.
- Loss chuẩn hoá theo pixel (cộng 3 kênh, chia số pixel hợp lệ), L_surf dạng L1 (Eq. 3a), không dùng SSIM.
- Lưu ảnh dạng uint8 trên RAM, linear RGB, checkpoint ghi nguyên tử và lưu trạng thái RNG, video so sánh, metric "Masked (RT)", FPS và thời gian train.

## Sửa vì sai hoặc thiếu so với framework

| Vấn đề ở `quan_dang` | Hậu quả | Sửa ở commit |
|---|---|---|
| FirstHit (Eq. 9) chỉ xét các patch | Kính nằm sau vật đục vẫn được chọn; phản xạ bị blend đè lên vật đó | `f_glass_geometry`, `f_occlusion_glass_layer`: kiểm tra che khuất bằng depth của G_I đã đóng băng, không tính lớp đĩa nằm trên kính; có ablation `--ablate_occlusion` |
| Rasterizer 2DGS của RT-Splatting có `DUAL_VISIABLE 0` | Pháp tuyến đĩa không quay về camera (đo được 41%), nên L1 của Eq. 3a phạt và ép các đĩa xoay 180° | `d_phase_one_interface`: `DUAL_VISIABLE 1` như 2DGS [1] |
| h_ψ nhận toạ độ pixel (u, v) thô, chặn ±0.1 bằng tanh | Cùng một pixel ở hai view là hai chỗ kính khác nhau, nên hiệu chỉnh không áp đúng lên view test; giới hạn không có trong Eq. 11 | `f_reflection_renderer`: đầu vào là điểm kính x_t(u), thời gian và cos θ; mặc định không chặn |
| V_dyn (Eq. 9b) được tính nhưng bị bỏ; u_ref là pixel ở giữa theo thứ tự quét | Không có cách so với xấp xỉ pinhole mà PDF nêu | `f_glass_geometry`, `f_reflection_renderer`: u_ref là pixel gần tâm vùng nhất; pose được dùng cho ablation `--pinhole_virtual_camera` và xuất ra `virtual_cameras.json` |
| Reader đọc mask từ `source_mask` (toạ độ ảnh nguồn còn méo) và fit lại mô hình méo mỗi lần load | Trên dữ liệu đã căn chỉnh (`colmap/masks`, `colmap/clean_image`), mask lệch với ảnh; load chậm | `f_scene_reader`: chỉ đọc layout đã căn chỉnh; `undistort_sources.py` căn một lần cho mỗi scene |
| Không reset presence, không prune theo kích thước trên màn hình, ngân sách điểm tính sai | Gaussian thừa không bị loại; cảnh rộng sinh Gaussian khổng lồ làm buffer rasterizer phình (OOM) | `f_gaussian_fields` |
| Không cập nhật `max_radii2D`; lấy view bằng `random.choice` | Prune theo màn hình không hoạt động; có view bị bỏ sót trong một epoch | `d_phase_one_interface` |
| Trích patch lấy mẫu con bằng `torch.randperm` chưa seed | Mỗi lần trích ra một bộ patch khác nhau (coverage 0.89 rồi 0.72 trên cùng một Phase I) | `f_extraction_determinism` |
| Không có ablation nào | Không kiểm chứng được từng thành phần | `d_phase_two_reflection`, `d_pipeline_ablations`: 12 biến thể (`docs/ablations.md`) |
| Metric chỉ so với ảnh gốc | Không đo được chất lượng tách lớp | `f_render_metrics`: Transmission PSNR (B so với ảnh clean) |

## Dọn dẹp

- Xoá `log/` (gần 1 MB patch và snapshot source), `record_implementation_change`, watcher trong `eval.sh`. Git đã làm việc lưu lịch sử này.
- Xoá các util của RT-Splatting không còn được import: `network_gui.py`, `mesh_utils.py`, `mcube_utils.py`, `render_utils.py`, `easyvolcap_utils.py`, `sph_utils.py`, `camera_utils.py`, `calibration_utils.py`.
- `specification.md` (thông số và kết quả kiểm tra của code cũ) được thay bằng `docs/implementation_details.md`, `docs/ablations.md` và tài liệu này. `framework.tex` được chuyển vào `docs/`.
- Code viết lại theo style dễ đọc: mỗi bước có script riêng (`train_interface.py`, `extract_glass.py`, `train_reflection.py`, `render.py`, `metrics.py`, cùng `train.py` chạy cả chuỗi), docstring dẫn số phương trình, không còn dòng code nén chặt nhiều lệnh.
- Test chuyển sang pytest với một scene tổng hợp có vật lý nhất quán (`tests/synthetic.py`): 50 test gồm hình học, CUDA, từng phase, ablation và pipeline.

## Không tương thích ngược

- Checkpoint version 2 (version 1 của `quan_dang` không đọc được). Cấu trúc thư mục model: `config.json`, `interface/`, `glass_geometry.json`, `reflection/`.
- Dòng lệnh: `train.py --phase ...` và `--resume <path>` được thay bằng các script theo bước (mỗi script có `--resume`) và `train.py` tự bỏ qua hoặc resume từng bước.
- Một số tên tham số đổi: `--reflection_iterations` thành `--iterations` của Phase II (truyền qua `--reflection_args` khi chạy `train.py`); `--plane_*` và các tham số che khuất nằm trong nhóm `GlassParams`.
