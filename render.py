"""Render the learned two-field VIGOR model using frozen 3D glass patches."""
import json
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from scene import Scene
from scene.gaussian_model import OpticalCorrection
from train import load_field
from gaussian_renderer import render_interface, render_composite
from utils.point_utils import validate_patches
from utils.color_utils import linear2srgb


def save_image(tensor,path,linear=False):
    if linear: tensor=linear2srgb(tensor)
    array=(tensor.detach().clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8)
    if array.shape[-1]==1: array=array[...,0]
    Image.fromarray(array).save(path)


@torch.no_grad()
def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('-m','--model_path',required=True)
    parser.add_argument('--skip_train',action='store_true')
    parser.add_argument('--skip_test',action='store_true')
    parser.add_argument('--skip_mesh',action='store_true',help='Compatibility flag; this optical model does not run TSDF extraction.')
    parser.add_argument('--components',action='store_true')
    parser.add_argument('--direct_only',action='store_true')
    parser.add_argument('--time',type=float,default=None,help='Override all timestamps with a value in [0,1].')
    parser.add_argument('--iteration',type=int,default=-1)
    cli=parser.parse_args()
    if cli.time is not None and not 0<=cli.time<=1: parser.error('--time must lie in [0,1]')
    root=Path(cli.model_path)
    args=SimpleNamespace(**json.loads((root/'config.json').read_text()))
    args.model_path=str(root.resolve())
    scene=Scene(args)
    interface,interface_deform,state=load_field(root/'interface/latest.pth',args,'interface',scene.cameras_extent,scene.center)
    if cli.direct_only:
        iteration=state['iteration']
    else:
        reflection,reflection_deform,state=load_field(root/'reflection/latest.pth',args,'reflection',scene.cameras_extent,scene.center)
        iteration=state['iteration']
        optical=OpticalCorrection(args.optical_bound).cuda().eval()
        optical.load_state_dict(state['optical'])
        patches=validate_patches(json.loads((root/'glass_geometry.json').read_text())['patches'])
    if cli.iteration not in (-1,iteration):
        raise ValueError(f'Available saved checkpoint iteration is {iteration}; got {cli.iteration}.')
    timing={}
    for split,cameras,skip in [('train',scene.getTrainCameras(),cli.skip_train),('test',scene.getTestCameras(),cli.skip_test)]:
        if skip or not cameras: continue
        name=('direct_' if cli.direct_only else 'ours_')+str(iteration)
        folder=root/split/name
        subfolders=['renders','gt','transparent_masks','valid_masks']
        if cli.components: subfolders+=['direct','reflection','alpha']
        for sub in subfolders: (folder/sub).mkdir(parents=True,exist_ok=True)
        milliseconds=[]
        for camera in tqdm(cameras,desc='Rendering '+split):
            if cli.time is not None: camera.time=cli.time
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            if cli.direct_only:
                out=render_interface(camera,interface,interface_deform,args)
                image=out['direct']
            else:
                out=render_composite(camera,interface,interface_deform,reflection,reflection_deform,patches,optical,args,args)
                image=out['composite']
            end.record();torch.cuda.synchronize();milliseconds.append(begin.elapsed_time(end))
            filename=camera.image_name+'.png'
            save_image(image,folder/'renders'/filename,linear=True)
            save_image(camera.original_image,folder/'gt'/filename)
            save_image(camera.gt_transparent_mask.float(),folder/'transparent_masks'/filename)
            save_image(camera.valid_mask.float(),folder/'valid_masks'/filename)
            if cli.components:
                save_image(out['direct'],folder/'direct'/filename,linear=True)
                save_image(out.get('reflection',torch.zeros_like(image)),folder/'reflection'/filename,linear=True)
                save_image(out.get('alpha',torch.zeros_like(image[:1])),folder/'alpha'/filename)
        timing[split]={'frames':len(cameras),'mean_ms':float(np.mean(milliseconds)),
            'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20}
    (root/'render_stats.json').write_text(json.dumps(timing,indent=2))
    print(json.dumps(timing,indent=2))


if __name__=='__main__': main()
