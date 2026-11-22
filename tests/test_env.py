"""Module 00: the environment can run the compiled extensions."""

import math

import pytest
import torch

from tests.helpers import look_at_camera, project_points


@pytest.mark.gpu
def test_cuda_and_extensions_import():
    assert torch.cuda.is_available()
    import diff_surfel_anych  # noqa: F401
    from simple_knn._C import distCUDA2

    dist2 = distCUDA2(torch.rand(64, 3, device='cuda'))
    assert dist2.shape == (64,)
    assert torch.isfinite(dist2).all()


@pytest.mark.gpu
def test_surfel_rasterizer_smoke():
    """One surfel facing the camera lands on the pixel its center projects to."""
    from diff_surfel_anych import GaussianRasterizationSettings, GaussianRasterizer

    cam = look_at_camera(eye=(0.0, 0.0, -3.0), target=(0.0, 0.0, 0.0))
    settings = GaussianRasterizationSettings(
        image_height=cam.image_height,
        image_width=cam.image_width,
        tanfovx=math.tan(cam.FoVx * 0.5),
        tanfovy=math.tan(cam.FoVy * 0.5),
        bg=torch.zeros(3, device='cuda'),
        scale_modifier=1.0,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=0,
        campos=cam.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    center = torch.tensor([[0.3, -0.2, 0.0]], device='cuda')
    means2D = torch.zeros_like(center, requires_grad=True)
    shs = torch.full((1, 1, 3), 2.0, device='cuda')  # degree-0 SH; bright enough to be clearly visible
    extras = torch.tensor([[0.25, 0.75]], device='cuda')
    scales = torch.full((1, 2), 0.2, device='cuda')
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda')  # normal along world z, facing the camera
    opacities = torch.full((1, 1), 0.9, device='cuda')

    color, extra, radii, allmap = rasterizer(means3D=center, means2D=means2D, shs=shs, extras=extras, opacities=opacities, scales=scales, rotations=rotations, cov3D_precomp=None)

    H, W = cam.image_height, cam.image_width
    assert color.shape == (3, H, W)
    assert extra.shape == (2, H, W)
    assert allmap.shape == (7, H, W)
    assert radii.shape == (1,) and radii[0] > 0

    alpha = allmap[1]
    peak = alpha.flatten().argmax()
    peak_xy = torch.stack([peak % W, peak // W]).float()
    expected_xy = project_points(cam, center)[0]
    assert torch.allclose(peak_xy, expected_xy, atol=1.5), (peak_xy, expected_xy)

    # extras are alpha-blended per-Gaussian values, so at the peak they approach alpha * value
    y, x = int(peak // W), int(peak % W)
    assert alpha[y, x] > 0.5
    assert torch.allclose(extra[:, y, x] / alpha[y, x], extras[0], atol=1e-3)
