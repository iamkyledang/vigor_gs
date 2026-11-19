"""Fresnel optics, bounded correction and linear-RGB field composition."""
import torch
from torch import nn

from gaussian_renderer.interface import render_interface
from gaussian_renderer.reflection import render_reflection
from utils.glass_geometry import first_hit, dynamic_virtual_camera


class OpticalCorrection(nn.Module):
    """Bounded, zero-initialized h(u,t), without a learned glass mask."""
    def __init__(self, bound=0.1):
        super().__init__()
        self.bound = float(bound)
        self.net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, uv, time, cosine):
        t = uv.new_full((*uv.shape[:-1], 1), float(time))
        features = torch.cat([uv, t, cosine[..., None]], -1)
        return self.bound * torch.tanh(self.net(features)[..., 0])


def ray_geometry(view,patches,args,pipe):
    geometry=first_hit(view,patches,pipe.ray_epsilon)
    F0=((args.eta_air-args.eta_glass)/(args.eta_air+args.eta_glass))**2
    geometry['fresnel']=F0+(1-F0)*(1-geometry['cosine'])**5
    geometry['fresnel']*=geometry['visible']
    # A single function/state is reused for every encountered patch.
    state=None
    for k in range(len(patches)):
        pose=dynamic_virtual_camera(view,geometry,k)
        if pose is not None: state=pose
    geometry['virtual_camera_state']=state
    return geometry


def render_composite(view,interface,interface_deform,reflection,reflection_deform,patches,optical,args,pipe,
                     direct=None,geometry=None,deforming=True):
    if direct is None:
        with torch.no_grad(): direct=render_interface(view,interface,interface_deform,pipe)['direct']
    if geometry is None: geometry=ray_geometry(view,patches,args,pipe)
    out=render_reflection(view,reflection,reflection_deform,patches,geometry,pipe,deforming)
    H,W=view.image_height,view.image_width
    y,x=torch.meshgrid(torch.linspace(-1,1,H,device=direct.device),torch.linspace(-1,1,W,device=direct.device),indexing='ij')
    correction=direct.new_zeros(H,W) if args.pure_fresnel else optical(torch.stack([x,y],-1),view.time,geometry['cosine'])
    correction=correction*geometry['visible']
    alpha=(geometry['fresnel']+correction).clamp(0,1)*geometry['visible']
    composite=(1-alpha[None])*direct+alpha[None]*out['reflection']
    out.update(composite=composite,final_rendering=composite,direct=direct,alpha=alpha[None],h=correction,
               geometry=geometry)
    return out
