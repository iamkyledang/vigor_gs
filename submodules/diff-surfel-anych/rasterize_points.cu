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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

#define CHECK_INPUT(x)											\
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

template<int NUM_CHANNELS>
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& extras,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  if (background.size(0) != 3) {
    AT_ERROR("background must have 3 channels");
  }

  if (NUM_CHANNELS > 0 && extras.size(-1) != NUM_CHANNELS) {
    AT_ERROR("extras must have " + std::to_string(NUM_CHANNELS) + " channels");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(extras);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({3, H, W}, 0.0, float_opts);
  torch::Tensor out_extra = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_others = torch::full({3+3+1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
	  }

	  rendered = CudaRasterizer::Rasterizer::forward<NUM_CHANNELS>(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		extras.contiguous().data_ptr<float>(),
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_extra.contiguous().data_ptr<float>(),
		out_others.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		debug);

  }
  return std::make_tuple(rendered, out_color, out_extra, out_others, radii, geomBuffer, binningBuffer, imgBuffer);
}

template<int NUM_CHANNELS>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& extras,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_extra,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(radii);
  CHECK_INPUT(extras);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);
  CHECK_INPUT(binningBuffer);
  CHECK_INPUT(imageBuffer);
  CHECK_INPUT(geomBuffer);

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolor = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dextras = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward<NUM_CHANNELS>(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  extras.contiguous().data_ptr<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  transMat_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_extra.contiguous().data<float>(),
	  dL_dout_others.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dnormal.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolor.contiguous().data<float>(),
	  dL_dextras.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dtransMat.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }
  
  return std::make_tuple(dL_dmeans2D, dL_dextras, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}


#define X(N) \
	template std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> \
	RasterizeGaussiansCUDA<N>( \
		const torch::Tensor& background, \
		const torch::Tensor& means3D, \
		const torch::Tensor& extras, \
		const torch::Tensor& opacity, \
		const torch::Tensor& scales, \
		const torch::Tensor& rotations, \
		const float scale_modifier, \
		const torch::Tensor& transMat_precomp, \
		const torch::Tensor& viewmatrix, \
		const torch::Tensor& projmatrix, \
		const float tan_fovx, \
		const float tan_fovy, \
		const int image_height, \
		const int image_width, \
		const torch::Tensor& sh, \
		const int degree, \
		const torch::Tensor& campos, \
		const bool prefiltered, \
		const bool debug); \
	template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> \
	RasterizeGaussiansBackwardCUDA<N>( \
		const torch::Tensor& background, \
		const torch::Tensor& means3D, \
		const torch::Tensor& radii, \
		const torch::Tensor& extras, \
		const torch::Tensor& scales, \
		const torch::Tensor& rotations, \
		const float scale_modifier, \
		const torch::Tensor& transMat_precomp, \
		const torch::Tensor& viewmatrix, \
		const torch::Tensor& projmatrix, \
		const float tan_fovx, \
		const float tan_fovy, \
		const torch::Tensor& dL_dout_color, \
		const torch::Tensor& dL_dout_extra, \
		const torch::Tensor& dL_dout_others, \
		const torch::Tensor& sh, \
		const int degree, \
		const torch::Tensor& campos, \
		const torch::Tensor& geomBuffer, \
		const int R, \
		const torch::Tensor& binningBuffer, \
		const torch::Tensor& imageBuffer, \
		const bool debug);

LIST_CHANNELS

#undef X


// VIGOR-GS: one routed image, shared 3D ellipsoids, exact reflected ray geometry.
// Gaussian centers are sorted per tile as in 3DGS; ray/ellipsoid evaluation is
// perspective-correct, rather than projecting an ordinary camera at the glass.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__device__ bool vigor_sample(int i, int pixel, const float* origins, const float* dirs,
    const float* means, const float* axes, const float* opacity, float cutoff2,
    float& raw, float& alpha, float& distance, float* v, float* w, float* e)
{
    float numerator=0.f, denominator=0.f;
    for (int k=0;k<3;k++) {
        v[k]=0.f; w[k]=0.f;
        for (int l=0;l<3;l++) {
            float a=axes[9*i+3*l+k];
            v[k]+=(means[3*i+l]-origins[3*pixel+l])*a;
            w[k]+=dirs[3*pixel+l]*a;
        }
        numerator+=v[k]*w[k]; denominator+=w[k]*w[k];
    }
    distance=numerator/fmaxf(denominator,1e-20f);
    if (distance<=0.f) return false; // begin at the glass hit, front segment only
    float q=0.f;
    for(int k=0;k<3;k++) { e[k]=distance*w[k]-v[k]; q+=e[k]*e[k]; }
    if(q>cutoff2) return false;
    raw=opacity[i]*expf(-0.5f*q);
    alpha=fminf(0.99f,raw);
    return alpha>=1.f/255.f;
}

__global__ void vigor_forward_kernel(int W,int H,int tile,int N,
    const float* origins,const float* dirs,const int* patches,
    const float* means,const float* axes,const float* opacity,const float* colors,
    const int64_t* entries,const int64_t* ranges,float cutoff2,
    float* image,float* final_t,int64_t* last)
{
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if(pixel>=W*H) return;
    int patch=patches[pixel], x=pixel%W,y=pixel/W;
    int tile_id=(y/tile)*((W+tile-1)/tile)+x/tile;
    float T=1.f,c[3]={0.f,0.f,0.f};
    int64_t end=-1;
    if(patch>=0) for(int64_t j=ranges[tile_id]; j<ranges[tile_id+1];j++) {
        int index=entries[j],i=index%N;
        if(index/N!=patch) continue;
        float raw,alpha,distance,v[3],w[3],e[3];
        if(!vigor_sample(i,pixel,origins,dirs,means,axes,opacity,cutoff2,raw,alpha,distance,v,w,e)) continue;
        for(int k=0;k<3;k++) c[k]+=T*alpha*colors[3*index+k];
        T*=1.f-alpha; end=j;
        if(T<1e-4f) break;
    }
    for(int k=0;k<3;k++) image[k*W*H+pixel]=c[k];
    final_t[pixel]=T; last[pixel]=end;
}

__global__ void vigor_backward_kernel(int W,int H,int tile,int N,
    const float* origins,const float* dirs,const int* patches,
    const float* means,const float* axes,const float* opacity,const float* colors,
    const int64_t* entries,const int64_t* ranges,float cutoff2,
    const float* final_t,const int64_t* last,const float* gradient,
    float* gmean,float* gaxes,float* gopacity,float* gcolors)
{
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if(pixel>=W*H || last[pixel]<0) return;
    int patch=patches[pixel],x=pixel%W,y=pixel/W;
    int tile_id=(y/tile)*((W+tile-1)/tile)+x/tile;
    float T=final_t[pixel],behind[3]={0.f,0.f,0.f};
    for(int64_t j=last[pixel];j>=ranges[tile_id];j--) {
        int index=entries[j],i=index%N;
        if(index/N!=patch) continue;
        float raw,alpha,distance,v[3],w[3],e[3];
        if(!vigor_sample(i,pixel,origins,dirs,means,axes,opacity,cutoff2,raw,alpha,distance,v,w,e)) continue;
        T/=1.f-alpha;
        float ga=0.f;
        for(int k=0;k<3;k++) {
            float gc=gradient[k*W*H+pixel];
            ga+=T*(colors[3*index+k]-behind[k])*gc;
            atomicAdd(gcolors+3*index+k,T*alpha*gc);
            behind[k]=alpha*colors[3*index+k]+(1.f-alpha)*behind[k];
        }
        if(raw>=0.99f) continue;
        atomicAdd(gopacity+i,ga*raw/fmaxf(opacity[i],1e-20f));
        float gq=-0.5f*raw*ga;
        for(int l=0;l<3;l++) {
            float gm=0.f;
            for(int k=0;k<3;k++) {
                float gv=-2.f*e[k]*gq;
                float gw=2.f*distance*e[k]*gq;
                gm+=axes[9*i+3*l+k]*gv;
                float gb=(means[3*i+l]-origins[3*pixel+l])*gv+dirs[3*pixel+l]*gw;
                atomicAdd(gaxes+9*i+3*l+k,gb);
            }
            atomicAdd(gmean+3*i+l,gm);
        }
    }
}

static void vigor_validate(const torch::Tensor& origins,const torch::Tensor& dirs,
    const torch::Tensor& patches,const torch::Tensor& means,const torch::Tensor& axes,
    const torch::Tensor& opacity,const torch::Tensor& colors,const torch::Tensor& entries,
    const torch::Tensor& ranges,int W,int H,int tile)
{
    TORCH_CHECK(W>0 && H>0 && tile>0,"Invalid image/tile dimensions");
    for(const auto& t : {origins,dirs,means,axes,opacity,colors}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type()==torch::kFloat32,
            "Ray renderer expects contiguous CUDA float32 tensors");
        TORCH_CHECK(t.device()==means.device(),"Ray tensors must share a CUDA device");
    }
    TORCH_CHECK(origins.numel()==W*H*3 && dirs.numel()==W*H*3,"Invalid rays");
    TORCH_CHECK(means.dim()==2 && means.size(1)==3 && means.size(0)>0,"Invalid Gaussian means");
    TORCH_CHECK(axes.numel()==means.size(0)*9 && opacity.numel()==means.size(0),"Invalid Gaussian shapes");
    TORCH_CHECK(colors.dim()==2 && colors.size(1)==3 && colors.size(0)%means.size(0)==0,"Invalid colors");
    TORCH_CHECK(patches.is_cuda() && patches.is_contiguous() && patches.scalar_type()==torch::kInt32 && patches.numel()==W*H,"Invalid patch IDs");
    for(const auto& t : {entries,ranges})
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type()==torch::kInt64 && t.device()==means.device(),"Invalid tile indices");
    TORCH_CHECK(ranges.numel()==((W+tile-1)/tile)*((H+tile-1)/tile)+1,"Invalid tile ranges");
}

std::vector<torch::Tensor> VigorRayForward(torch::Tensor origins,torch::Tensor dirs,
    torch::Tensor patches,torch::Tensor means,torch::Tensor axes,torch::Tensor opacity,
    torch::Tensor colors,torch::Tensor entries,torch::Tensor ranges,int W,int H,int tile,float cutoff)
{
    vigor_validate(origins,dirs,patches,means,axes,opacity,colors,entries,ranges,W,H,tile);
    c10::cuda::CUDAGuard guard(means.device());
    auto image=torch::zeros({3,H,W},means.options());
    auto trans=torch::ones({H*W},means.options());
    auto last=torch::full({H*W},-1,entries.options());
    vigor_forward_kernel<<<(W*H+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(W,H,tile,means.size(0),
        origins.data_ptr<float>(),dirs.data_ptr<float>(),patches.data_ptr<int>(),means.data_ptr<float>(),
        axes.data_ptr<float>(),opacity.data_ptr<float>(),colors.data_ptr<float>(),entries.data_ptr<int64_t>(),
        ranges.data_ptr<int64_t>(),cutoff*cutoff,image.data_ptr<float>(),trans.data_ptr<float>(),last.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {image,trans,last};
}

std::vector<torch::Tensor> VigorRayBackward(torch::Tensor origins,torch::Tensor dirs,
    torch::Tensor patches,torch::Tensor means,torch::Tensor axes,torch::Tensor opacity,
    torch::Tensor colors,torch::Tensor entries,torch::Tensor ranges,torch::Tensor trans,
    torch::Tensor last,torch::Tensor gradient,int W,int H,int tile,float cutoff)
{
    vigor_validate(origins,dirs,patches,means,axes,opacity,colors,entries,ranges,W,H,tile);
    c10::cuda::CUDAGuard guard(means.device());
    auto gm=torch::zeros_like(means),ga=torch::zeros_like(axes),go=torch::zeros_like(opacity),gc=torch::zeros_like(colors);
    vigor_backward_kernel<<<(W*H+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(W,H,tile,means.size(0),
        origins.data_ptr<float>(),dirs.data_ptr<float>(),patches.data_ptr<int>(),means.data_ptr<float>(),
        axes.data_ptr<float>(),opacity.data_ptr<float>(),colors.data_ptr<float>(),entries.data_ptr<int64_t>(),
        ranges.data_ptr<int64_t>(),cutoff*cutoff,trans.data_ptr<float>(),last.data_ptr<int64_t>(),gradient.data_ptr<float>(),
        gm.data_ptr<float>(),ga.data_ptr<float>(),go.data_ptr<float>(),gc.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gm,ga,go,gc};
}
