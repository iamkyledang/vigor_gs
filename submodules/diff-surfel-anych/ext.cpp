/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "cuda_rasterizer/config.h"


std::vector<torch::Tensor> VigorRayForward(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int,int,int,float);
std::vector<torch::Tensor> VigorRayBackward(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int,int,int,float);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define X(N) \
  m.def("rasterize_gaussians_" #N, &RasterizeGaussiansCUDA<N>); \
  m.def("rasterize_gaussians_backward_" #N, &RasterizeGaussiansBackwardCUDA<N>);
LIST_CHANNELS
#undef X
  m.def("mark_visible", &markVisible);
  m.def("vigor_ray_forward", &VigorRayForward);
  m.def("vigor_ray_backward", &VigorRayBackward);
}