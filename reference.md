# VIGOR-GS — Research References

**Project:** Virtual-View Interface-Guided Optical Reflection Gaussian Splatting
**Last updated:** 2026-11-16
**Purpose:** Danh mục tài liệu nghiên cứu và mã nguồn tham khảo để triển khai VIGOR-GS theo từng module.

---

## 1. Tổng quan framework

**VIGOR-GS** là framework được thiết kế để tái dựng cảnh động có đồng thời hiện tượng **transmission** và **reflection** qua bề mặt kính, dựa trên Gaussian Splatting.

Framework sử dụng hai Gaussian fields độc lập:

- **Interface Gaussian Field (\(G_I\))**: Biểu diễn hình học bề mặt kính và cảnh trực tiếp quan sát được xuyên qua kính, sử dụng 2D Gaussian Splatting.
- **Reflection Gaussian Field (\(G_R\))**: Biểu diễn cảnh phản xạ trong không gian ảnh ảo, sử dụng 3D Gaussian Splatting.

Cả hai Gaussian fields đều được tích hợp **Deformation MLP** để mô hình hóa các thành phần động theo thời gian. Đây là thành phần bắt buộc trong thiết kế đầy đủ của VIGOR-GS.

Framework được huấn luyện tuần tự qua hai giai đoạn.

### Phase I — Dynamic Interface Reconstruction

**Mục tiêu:** Tái dựng direct scene, học biểu diễn hình học của bề mặt kính và xác định glass plane.

- Sử dụng **paired clean RGB images** (\(I_t^{clean}\)) làm supervision để tái dựng cảnh trực tiếp, không chứa reflection.
- Xây dựng **Interface Gaussian Field (\(G_I\))** dựa trên 2D Gaussian Splatting.
- Tích hợp **Interface Deformation MLP (\(D_{\phi_I}\))** để mô hình hóa chuyển động của các thành phần động trong direct scene, đồng thời duy trì tính ổn định hình học của bề mặt kính.
- Học hai đại lượng độc lập:
  - **Geometric occupancy (\(p\))**: Biểu diễn sự hiện diện hình học của Gaussian, phục vụ việc tái dựng bề mặt kính.
  - **Optical opacity (\(o\))**: Điều khiển mức độ đóng góp quang học, với effective opacity \(p \cdot o\) trong quá trình volume rendering.
- Sử dụng **glass masks (\(M_t\))** để giám sát vùng bề mặt kính và hỗ trợ học hình học interface.
- Trích xuất **glass plane \(\Pi^*=(\mathbf n,d)\)** trực tiếp từ hình học Interface 2DGS đã học (giống như trong gaussian_renderer\__init__.py dòng 103 ). Glass plane không được cung cấp sẵn trong dữ liệu đầu vào.
- Render direct image \(B_t\) từ Interface Gaussian Field.

**Kết quả Phase I:**

- Interface Gaussian Field \(G_I\).
- Interface Deformation MLP \(D_{\phi_I}\).
- Glass plane \(\Pi^*\).
- Direct rendering \(B_t\).

Sau khi hoàn thành Phase I, **freeze toàn bộ Interface Gaussian Field, Interface Deformation MLP và glass plane** để sử dụng trong Phase II.

### Phase II — Dynamic Reflection Reconstruction

**Mục tiêu:** Tái dựng reflection scene độc lập và kết hợp reflection với direct scene để khôi phục ảnh quan sát ban đầu.

- Sử dụng **glass plane \(\Pi^*\)** đã được trích xuất từ Phase I để xác định hình học phản xạ.
- Xây dựng **virtual camera** bằng phép đối xứng camera thực qua glass plane:

  \[
  \mathbf c_t^v=\mathbf c_t-2(\mathbf n^\top\mathbf c_t+d)\mathbf n
  \]

- Xây dựng **Reflection Gaussian Field (\(G_R\))** độc lập với Interface Gaussian Field, dựa trên 3D Gaussian Splatting.
- Tích hợp **Reflection Deformation MLP (\(D_{\phi_R}\))** để mô hình hóa chuyển động của các đối tượng trong reflection scene theo thời gian.
- Render reflection image \(R_t\) bằng virtual camera và Reflection Gaussian Field.
- Sử dụng **Fresnel-based compositing** để kết hợp direct rendering \(B_t\) và reflection rendering \(R_t\), với sự hỗ trợ của glass masks nhằm xác định vùng chịu ảnh hưởng của kính. (vẫn có nhập nhằng bên trong này có thể sử dụng các research bên dưới để hệ thức lại)
- Sử dụng **original RGB images (\(I_t\))** làm supervision để tối ưu Reflection Gaussian Field, Reflection Deformation MLP và các tham số kết hợp quang học.

**Kết quả Phase II:**

- Reflection Gaussian Field \(G_R\).
- Reflection Deformation MLP \(D_{\phi_R}\).
- Reflection rendering \(R_t\).
- Final reconstructed image \(\hat I_t\).

### Final Rendering

Ảnh cuối cùng được tổng hợp từ direct và reflection components:

\[
\boxed{
\hat I_t=(1-\alpha_t)B_t+\alpha_tR_t
}
\]

Trong đó:

- \(B_t\): Direct rendering từ Phase I.
- \(R_t\): Reflection rendering từ Phase II.
- \(\alpha_t\): Hệ số kết hợp được điều khiển bởi Fresnel-based model và giới hạn trong vùng kính bằng glass mask.

**Đặc điểm cốt lõi của VIGOR-GS** là sử dụng hình học interface được học từ clean supervision để điều khiển quá trình tái dựng reflection, đồng thời phân tách hai Gaussian fields và hai deformation models nhằm tái dựng độc lập các thành phần động của direct scene và reflection scene.




## 2. Danh mục tài liệu tổng hợp

| ID  | Paper                                | Năm           | Vai trò trong VIGOR-GS                          | 
| --- | ------------------------------------ | ------------- | ----------------------------------------------- | 
| R01 | 3D Gaussian Splatting                | 2023          | Reflection 3DGS, rasterization                  | 
| R02 | 2D Gaussian Splatting                | 2024          | Interface 2DGS, surface geometry                |
| R03 | RT-Splatting                         | 2026          | Occupancy–opacity, surface-volume rendering     | 
| R04 | MirrorGaussian                       | 2024          | Planar reflection, virtual geometry             | 
| R05 | TR-Gaussians                         | 2026          | Planar transmission/reflection, Fresnel         | 
| R06 | GLINT                                | 2026          | Interface/transmission/reflection decomposition |
| R08 | Deformable 3D Gaussians              | 2024          | Deformation MLP                                 |
| R09 | 4D Gaussian Splatting                | 2024          | Dynamic Gaussian representation                 | 
| R10 | TraceFlow                            | 2025 preprint | Dynamic 2DGS, reflection ray tracing            |
| R11 | SpectroMotion                        | 2025          | Dynamic specular reconstruction                 | 
| R12 | Ref-NeRF                             | 2022          | View-dependent reflection modeling              | 

---

## 3. Core references — P0

### R01. 3D Gaussian Splatting for Real-Time Radiance Field Rendering

- Paper: [https://arxiv.org/abs/2308.04079](https://arxiv.org/abs/2308.04079)
- Project: [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- Official code: [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

### R02. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

- Paper: [https://arxiv.org/abs/2403.17888](https://arxiv.org/abs/2403.17888)
- Project: [https://surfsplatting.github.io/](https://surfsplatting.github.io/)
- Official code: [https://github.com/hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting)
- DOI: [https://doi.org/10.1145/3641519.3657428](https://doi.org/10.1145/3641519.3657428)

### R03. RT-Splatting: Joint Reflection-Transmission Modeling with Gaussian Splatting



- Paper: [https://openaccess.thecvf.com/content/CVPR2026/html/Shi_RT-Splatting_Joint_Reflection-Transmission_Modeling_with_Gaussian_Splatting_CVPR_2026_paper.html](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_RT-Splatting_Joint_Reflection-Transmission_Modeling_with_Gaussian_Splatting_CVPR_2026_paper.html)
- Project: [https://sjj118.github.io/RT-Splatting/](https://sjj118.github.io/RT-Splatting/)
- Official code: [https://github.com/sjj118/RT-Splatting](https://github.com/sjj118/RT-Splatting)



### R04. MirrorGaussian: Reflecting 3D Gaussians for Reconstructing Mirror Reflections

- Project: [https://mirror-gaussian.github.io/](https://mirror-gaussian.github.io/)
- Paper: truy cập từ project page.
- Official code: chưa xác minh được repository chính thức từ project page.

### R05. TR-Gaussians: High-Fidelity Real-Time Rendering of Planar Transmission and Reflection With 3D Gaussian Splatting

- Paper: [https://arxiv.org/abs/2511.13009](https://arxiv.org/abs/2511.13009)
- Official code: [https://github.com/CJAPPLE5/TR-Gaussians](https://github.com/CJAPPLE5/TR-Gaussians)
- DOI: [https://doi.org/10.1109/TVCG.2026.3675416](https://doi.org/10.1109/TVCG.2026.3675416)



### R06. GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport

- Paper: [https://arxiv.org/abs/2603.26181](https://arxiv.org/abs/2603.26181)
- Project: [https://youngju-na.github.io/GLINT/](https://youngju-na.github.io/GLINT/)
- Official code: [https://github.com/Ardynai/glint](https://github.com/Ardynai/glint)

###

---

## 4. Dynamic and advanced references — P1

### R08. Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction

- Paper: [https://arxiv.org/abs/2309.13101](https://arxiv.org/abs/2309.13101)
- Project: [https://ingra14m.github.io/Deformable-Gaussians/](https://ingra14m.github.io/Deformable-Gaussians/)
- Official code: [https://github.com/ingra14m/Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)

### R09. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

- Paper: [https://arxiv.org/abs/2310.08528](https://arxiv.org/abs/2310.08528)
- Official code: [https://github.com/hustvl/4DGaussians](https://github.com/hustvl/4DGaussians)

### R10. TraceFlow: Dynamic 3D Reconstruction of Specular Scenes Driven by Ray Tracing



- Paper: [https://arxiv.org/abs/2512.10095](https://arxiv.org/abs/2512.10095)
- Official code: chưa xác minh repository công khai.

### R11. SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes

- Paper: [https://openaccess.thecvf.com/content/CVPR2025/html/Fan_SpectroMotion_Dynamic_3D_Reconstruction_of_Specular_Scenes_CVPR_2025_paper.html](https://openaccess.thecvf.com/content/CVPR2025/html/Fan_SpectroMotion_Dynamic_3D_Reconstruction_of_Specular_Scenes_CVPR_2025_paper.html)
- Project: [https://cdfan0627.github.io/spectromotion/](https://cdfan0627.github.io/spectromotion/)

---

## 5. Background references — P2

### R12. Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields

**Authors:** Dor Verbin et al.
**Year:** 2022

- Paper: [https://arxiv.org/abs/2112.03907](https://arxiv.org/abs/2112.03907)
- Project: [https://dorverbin.github.io/refnerf/](https://dorverbin.github.io/refnerf/)

##


Thí nghiệm (Ablation)	Mục tiêu/Chất vấn

Loại bỏ factorization ((p_i \times o_i))	Xác định tầm quan trọng của phân tách occupancy và opacity.



Loại bỏ virtual camera (không phản chiếu)	Xem geometry camera ảo quan trọng ra sao cho phản chiếu.

Không phân tách trường phản chiếu độc lập	kết quả sẽ như thế nào


Không dùng Fresnel (blending thô)	Xác định giá trị của Fresnel reflectance trong tổng hợp ảnh.


---
