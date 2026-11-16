# VIGOR-GS: Kế hoạch Nghiên cứu & Triển khai

## A. Tài liệu tham khảo và mapping module

| Paper/Mã                        | Vấn đề giải quyết                       | Ý tưởng trọng tâm / Thuật toán                                            | Module áp dụng           | Độ khó tái dụng | Repo/Code (License)      | Ưu tiên |
|---------------------------------|----------------------------------------|------------------------------------------------------------------------------|--------------------------|-----------------|--------------------------|---------|
| **Kerbl et al. (2023, SIGGRAPH)** | Cơ bản 3D Gaussian Splatting        | *Khởi tạo từ sparse points (camera calibration) → đại diện cảnh bằng Gaussian 3D anisotropic; fast rendering dựa trên splatting có che khuất.* | Module 3DGS (phải rebuild) | Trung bình      | [Repo Inria](https://repo-sam.inria.fr/rel/0b1) (CC-BY-NC-SA) | P0      |
| **MirrorGaussian (ECCV 2024)**       | Mô hình phản xạ gương phẳng        | *Chiến lược “dual-rendering”: phản chiếu toàn bộ trường Gaussian về phía đối diện mặt gương, rasterize cả hai rồi blend theo mask.* | Phase II (Reflection Geometry/Rendering) | Khó           | – (dự án trang web)     | P0      |
| **TR-Gaussians (2025)**            | Tái dựng scenes có planar glass     | *Tách trường Gaussian thành “transmission” và “reflection” qua mặt phẳng kính learnable; blend theo Fresnel multi-view.* | Fresnel & Composition, Training Reflection | Trung bình      | – (chưa công bố)         | P0      |
| **RT-Splatting (CVPR 2026)**                  | Reflection-transmission splitting   | *Phân thành 2 pass (surface/volume) với occupancy vs opacity; supervised mask kính để học độ trong suốt.* | Module 2DGS (glass mask)  | Trung bình      | – (chưa công bố)         | P1      |
       | –               | –                        | -       |
| **SpectroMotion, TraceFlow, …**             | (Motion & Ray tracing refs)        | *Các tài liệu tham khảo nâng cao, có thể xem thêm cho dynamic/specular.* | (Tham khảo)             | N/A             | –                        | P2      |
| **GLINT**



> **Chú thích:** Các nghiên cứu trên cho thấy việc **phân tách vấn đề thành nhiều giai đoạn** là thiết kế phổ biến. Ví dụ, *MirrorGaussian* dùng dual-rendering và đồng tối ưu scene thật lẫn ảnh gương với mặt phẳng, *TR-Gaussians* dùng plane và Fresnel trong multi-stage. *Kerbl et al.* (3DGS) là tham chiếu cơ bản cho rendering Gaussian real-time. 

## B. Kế hoạch triển khai theo commit

Chúng ta sẽ phát triển tuần tự, từng bước một commit, mỗi bước có mục tiêu rõ, kiểm thử độc lập và điều kiện hợp lệ (acceptance gate) trước khi merge:

- **Commit 00 – Data Contract và Import:** Tạo module đọc ảnh và camera.  
  - *Files:* `data_loader.py`, `camera.py`.  
  - *Mô tả:* Đọc ảnh \(I_t, I_t^{clean}, M_t\) và camera poses từ tập dữ liệu. Hỗ trợ format của COLMAP. Chuyển ảnh sang linear space.  
  - *Test:* Unit test kiểm tra đọc đúng file (chiều ảnh, dtype), ma trận camera hợp lệ.  
  - *Điều kiện:* Không lỗi IO, ma trận P thu được chắc chắn là vô nghịch, ảnh đầu ra thẳng tuyến.  
  - *Commit msg:* `"00-data-contract: Khởi tạo module đọc dữ liệu và camera."`

- **Commit 01 – Xử lý Toạ độ Camera (Virtual Camera):** Triển khai biến đổi camera thật → camera ảo qua mặt phẳng.  
  - *Files:* cập nhật `camera.py`.  
  - *Mô tả:* Hàm `reflect_camera(Camera C, plane)`. Áp dụng \(\mathbf{c}^v = \mathbf{c}-2(\mathbf n^T\mathbf c + d)\mathbf n\), và phản chiếu hướng nhìn.  
  - *Test:* Ví dụ mặt phẳng \(z=0\): camera \((1,0,2)\) cho ra \((1,0,-2)\). Kiểm tra hướng nhìn (ví dụ ray test đơn giản).  
  - *Điều kiện:* Kết quả so với tính bằng tay phải khớp.  
  - *Commit msg:* `"01-camera-geometry: Thêm hàm tạo virtual-camera đối xứng qua mặt phẳng kính."`

- **Commit 02 – Gaussian 2D tĩnh (Interface reconstruction):** Xây dựng trường Gaussian 2D và renderer để tái dựng ảnh trực tiếp.  
  - *Files:* `interface_gaussians.py`, `renderer_2d.py`.  
  - *Mô tả:* Lớp `Gaussian2D` chứa \((\mu,q,s^{(2)},p,o,a)\), và `GaussianField2D` hold tập \(G_I\). `Renderer2D` rasterize sang ảnh (alpha blend).  
  - *Test:* Tạo một số Gaussian màu đơn giản, render ảnh thử, kiểm tra kết quả (màu sắc, hình dạng). Huấn luyện thử (vài bước) trên ảnh clean đơn giản, loss phải giảm.  
  - *Điều kiện:* Renderer không crash, output khớp ước lượng.  
  - *Commit msg:* `"02-static-2dgs-direct: Xây dựng module Gaussian 2D và renderer cho direct image."`

- **Commit 03 – Surface Pass (Mask từ Occupancy):** Tính mask kính từ `p_i`.  
  - *Files:* mở rộng `renderer_2d.py`.  
  - *Mô tả:* Hàm `render_mask(G_I, C_t)` dùng công thức \(1 - \prod(1 - p_i g_i(u))\) để cho ra \(\hat M_t\).  
  - *Test:* Tình huống đơn giản: một Gaussian với \(p=1\) bao trùm toàn ảnh → mask toàn 1. Kiểm thử IoU giữa \(\hat M_t\) và `M_t`.  
  - *Điều kiện:* Công thức đúng (IoU đo lường ý nghĩa).  
  - *Commit msg:* `"03-occupancy-surface-pass: Tính toán ảnh mask kính từ occupancy."`

- **Commit 04 – Huấn luyện Interface + Fit Plane kính:**  
  - *Files:* `train_interface.py`, `plane_fit.py`.  
  - *Mô tả:* Train \(G_I\) với loss ảnh sạch + mask. Sau khi training, dùng `fit_plane()` để tính mặt phẳng tốt nhất từ các \(\mu_i\) thuộc kính.  
  - *Test:* Synthetic test: tạo dataset có kính = plane z=0; huấn luyện phải cho `n≈(0,0,1), d≈0`.  
  - *Điều kiện:* Loss ảnh giảm hợp lý; IoU mask ≥0.9 trên train; plane trích xuất lệch nhỏ (vd. góc <1°) so với thật (trên synthetic).  
  - *Commit msg:* `"04-glass-plane-phase1: Huấn luyện interface và fit plane kính."`

- **Commit 05 – Freeze Interface:**  
  - *Files:* `train_reflection.py`.  
  - *Mô tả:* Đánh dấu tham số \(G_I\) và \(\Pi^*\) không cập nhật. Khởi tạo `G_R`.  
  - *Test:* Chạy vài step của train_reflection, đảm bảo \(G_I\) không đổi (so sánh trước/sau).  
  - *Điều kiện:* \(\Delta G_I = 0\) qua toàn bộ training.  
  - *Commit msg:* `"05-freeze-phase1: Đóng băng tham số interface và plane sau Phase I."`

- **Commit 06 – Gaussian 3D tĩnh (Reflection field):**  
  - *Files:* `reflection_gaussians.py`, `renderer_3d.py`.  
  - *Mô tả:* Lớp `Gaussian3D` với \((\mu,q,s^{(3)},o,a)\), và renderer 3D (có che khuất). Cho phép render \(R_t\) từ camera ảo.  
  - *Test:* Đơn giản: 1 Gaussian 3D màu xanh ở (0,0,5). Render từ camera ảo và camera thật; so sánh vị trí điểm trên ảnh.  
  - *Điều kiện:* Khi camera ảo di chuyển, `R_t` phải thay đổi đúng hướng (kiểm bằng mắt). Không lỗi về trọng số alpha.  
  - *Commit msg:* `"06-reflection-3dgs: Khởi tạo module Gaussian 3D và renderer cho reflection."`

- **Commit 07 – Kiểm thử Virtual Camera:**  
  - *Files:* `test_virtual_camera.py` (hoặc thêm vào suite test).  
  - *Mô tả:* Kiểm tra tổng hợp: chọn cảnh đơn giản, dịch camera thật ngang, xem camera ảo di chuyển đối xứng. Chụp ảnh `R_t` và so sánh.  
  - *Test:* Xây cảnh 3D giả (vd. một chấm sáng), di chuyển camera thật. \(\Rightarrow\) `R_t` của camera ảo tương ứng.  
  - *Điều kiện:* Ảnh kết quả biến đổi như mong đợi (đường đi vật thể phản ngược chiều camera di chuyển).  
  - *Commit msg:* `"07-virtual-camera: Test tính đúng của virtual camera với scenes mẫu."`

- **Commit 08 – Fresnel Compositing:**  
  - *Files:* `fresnel.py`.  
  - *Mô tả:* Tính \(\alpha_t(u)\) từ góc tới (Schlick). Kết hợp \(\hat I_t = (1-\alpha)B_t + \alpha R_t\). Xử lý không gian màu tuyến tính.  
  - *Test:* Giả lập: \(B_t\)=1, \(R_t\)=0 cho góc vuông; \(B_t\)=0, \(R_t\)=1 cho góc xiên. Kiểm tra compositing đúng.  
  - *Điều kiện:* Ảnh \(\hat I\) phù hợp với định luật. Nếu cài MLP `h_ψ`, test nhỏ: với h≡0, kết quả khớp thuần Fresnel.  
  - *Commit msg:* `"08-fresnel-compositor: Cài đặt cân nhắc Fresnel và ghép ảnh cuối."`

- **Commit 09 – Huấn luyện Reflection:**  
  - *Files:* `train_reflection.py`.  
  - *Mô tả:* Tối ưu \(G_R\) (và h_ψ) với loss so khớp ảnh gốc.  
  - *Test:* Kiểm tra loss giảm. Quan sát sự xuất hiện của vật thể phản xạ trong `R_t`.  
  - *Điều kiện:* Lỗi tổng \(\|\hat I_t - I_t\|\) giảm tốt (ví dụ giảm 80–90% trong vài epoch). Hình ảnh tái dựng hợp lý.  
  - *Commit msg:* `"09-phase2-train: Huấn luyện trường Gaussian phản xạ so khớp ảnh quan sát."`

- **Commit 10 – Đánh giá và Triển lãm kết quả:**  
  - *Files:* `evaluate.py`.  
  - *Mô tả:* Tính các metric (PSNR, SSIM, LPIPS) cho ảnh tổng hợp. Xuất ảnh so sánh của direct, reflection, bản tái dựng.  
  - *Test:* Chạy trên tập test. Xuất report.  
  - *Điều kiện:* Không có lỗi, metrics ở mức cao (ví dụ SSIM>0.9). Có thể so sánh với baseline (ví dụ chỉ dùng direct, để thấy ảnh phản xạ đã được tái tạo).  
  - *Commit msg:* `"10-evaluation: Đánh giá chất lượng trên tập kiểm thử và lập báo cáo."`

- **Commit 11 – (Tùy chọn) Động hóa:**  
  - *Files:* `deformation.py`.  
  - *Mô tả:* Thêm network học deformation \(\Delta \mu_i(t)\) nếu vật phản xạ thật sự chuyển động.  
  - *Test:* Với sequence có vật chạy ngang qua, kiểm tra loss giảm hơn so với trường tĩnh.  
  - *Điều kiện:* Nếu dữ liệu thực có chuyển động, deformation MLP giúp giảm lỗi (thấp hơn 1–2\% so với static). Nếu không, revert.  
  - *Commit msg:* `"11-dynamic-reflection: Thêm khả năng deformation MLP cho trường phản xạ (scene động)."`

Mỗi commit được gắn kiểm thử unit riêng và đánh giá (integration test) trên ví dụ nhỏ hoặc dataset con. Chỉ khi cả test pass mới tiến sang bước kế tiếp. 

## C. Chi tiết kỹ thuật (Data/Metrics/Validation)

- **Định dạng dữ liệu:** Ảnh gốc/clean cùng kích thước (ví dụ 1920×1080). Mask nhị phân (0,1). Camera: intrinsics (focal, center) và extrinsics (4×4) dạng chuẩn (OpenCV coordinate).  
- **Ma trận máy và hệ tọa độ:** Giả định camera theo convention OpenCV (z ra phía trước). Phản chiếu vị trí và quay phải được áp dụng thống nhất.  
- **Fresnel:** Tính mỗi pixel \(\cos\vartheta(u)\) = \(|\mathbf n^T (\mathbf c - \mathbf x(u))|/\|\mathbf c-\mathbf x(u)\|\). Tính Schlick. Kết quả \(\alpha(u)\in[0,1]\). *Không dùng* đạo hàm Fresnel phức tạp – Schlick đủ tốt.  
- **Form composite:** \(\hat I = (1-\alpha)\,B + \alpha\,R\). (Trộn linear color.)  
- **Plane fitting:** Dùng các điểm \(\mu_i\) của Gaussian xem là “thuộc kính” (p lớn, hoặc footprint rọi vào mask). Giải SVD cho tối ưu \(\min_{\|n\|=1}\sum(n^T\mu_i+d)^2\). Hoặc RANSAC nếu nhiều outlier. Kiểm tra: nếu data tập hợp (0,0,1) cho \(n\), và \(d\) tương ứng.  
- **Chi tiết losses:** `L_mask = BCE(\hat M_t,M_t)`, `L_RGB = L1(Ĩ_t,I_t)`. Cân chỉnh \(\lambda_s,\lambda_o,\lambda_m\) qua validation.  
- **Chia tập:** Không để frame rất gần về mặt kính lọt cả train/test. Giữ khoảng cách thời gian (VD train: left-to-right, test: reversed path).  
- **Metrics:** Tính trên ảnh full và vùng kính. Ví dụ: PSNR/SSIM toàn cảnh, *PSNR/SSIM vùng mask=1*, IoU mask, angle error plane (so với đánh giá “ground truth” nếu biết). LPIPS đánh giá chất lượng thị giác.  
- **Triển khai:** Dùng PyTorch (nếu cần GPU). Tối ưu Adam hoặc AdamW. Khi đánh giá reproducibility, fix random seed trên GPU.  
- **Ví dụ CLI:** `python train_interface.py --data_dir ./data --epochs 100` và `python train_reflection.py ...`. Kết quả final lưu trong `results/`.

## D. Rủi ro và vấn đề
- **Plane không xác định:** Mask kính (2D) chỉ cho vị trí biên trên ảnh, không cho thông tin sâu. Cần đa view (camera di chuyển) để ước lượng plane. Nếu chỉ di chuyển ngang, mặt phẳng định vị được góc pháp tuyến (cơ bản thẳng đứng) nhưng không khôi phục bias (d) chính xác; máy có thể “bóp” d. (Có thể dùng RANSAC/SVD nhiều view để cố định.)  
- **Nhân đôi phản xạ:** Nếu sai hệ quy chiếu, dễ nhầm tưởng cần phản chiếu kép. *Phải nhớ:* Chỉ phản chiếu **camera**, không phản chiếu Gaussian thêm lần nữa. Hoặc ngược lại, nếu biểu diễn \(G_R\) dưới dạng “scene ảnh” sau kính, cần để camera thật dùng \(\Pi\) với cấu hình khác. (Tóm lại, quy ước phải nhất quán.)  
- **Brightness & Gamma:** Đảm bảo làm việc trong linear space cho blending. Nếu vô tình sử dụng màu gamma, ảnh tổng hợp sẽ sai sáng, bất thường.  
- **Overfit vào Fresnel:** Fresnel cho góc nghiêng thì tăng, góc vuông thì giảm, nhưng dữ liệu thực có nhiễu. `h_ψ` cho phép bù đắp nhưng quá lạm dụng có thể phá vỡ thực tế. Hạn chế `h_ψ` bằng regularizer nhỏ.  
- **Giảm tính khả định:** Loss RGB chỉ đánh giá ảnh cuối, nhiều hình học \(G_R\) khác nhau có thể cho ảnh giống nhau. Cần đủ góc nhìn và mask (tất cả hình tam giác).  
- **Mô hình động:** Nếu cảnh thật có vật di động (người di chuyển trong ảnh phản xạ), trường tĩnh sẽ khó giải thích. Khi đó cần `D_{φ_R}` để học chuyển động. Nếu dùng động không đúng (dữ liệu tĩnh), có thể học noise.  
- **Giới hạn Gaussian:** Số Gaussian quá ít → chi tiết kém. Quá nhiều → quá nặng. Cần test tradeoff (ví dụ 1-5k Gaussian cho cảnh indoor).  
- **Điều kiện rủi ro khác:** Sóc tiếng thể câm: mặt kính không phẳng, nhiễu ánh sáng, đa reflection (không xử lý), biến đổi cường độ (kính có NV hoặc màu). Xử lý dưới dạng post-ion (h_ψ) nếu cần.  

> **Tóm lại:** Thiết kế VIGOR-GS tương tự nhiều nghiên cứu hiện có trong lĩnh vực splatting – chia scene thành phần direct và phản xạ, tối ưu tuần tự. Ví dụ, MirrorGaussian và TR-Gaussians đều tách riêng hai thành phần này. Kế hoạch trên đảm bảo chúng ta từng bước xây dựng hệ thống, kiểm thử cẩn thận và chỉ mở rộng khi cần thiết.

