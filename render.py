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
from gaussian_renderer.composition import OpticalCorrection
from utils.checkpoint_utils import load_field
from gaussian_renderer import render_interface, render_composite
from utils.glass_geometry import validate_patches
from utils.color_utils import linear2srgb


def save_image(tensor,path,linear=False):
    if linear: tensor=linear2srgb(tensor)
    array=(tensor.detach().clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8)
    if array.shape[-1]==1: array=array[...,0]
    Image.fromarray(array).save(path)


def create_comparison_video(folder,image_names,fps=30.):
    """Stream saved sRGB pairs into an MP4: ours on the left, GT on the right."""
    import cv2

    if not np.isfinite(fps) or fps<=0: raise ValueError('Video FPS must be positive and finite.')
    folder=Path(folder)
    image_names=list(image_names)
    if not image_names: raise ValueError('No frames available for the comparison video.')
    width=height=0
    for name in image_names:
        with Image.open(folder/'renders'/(name+'.png')) as ours, Image.open(folder/'gt'/(name+'.png')) as gt:
            if ours.size!=gt.size: raise ValueError(f'Comparison size mismatch for {name}')
            width=max(width,ours.width);height=max(height,ours.height)
    # Keep native image sizes and pad the canvas to even codec dimensions.
    header=40
    canvas_height=height+header+(height+header)%2
    path=folder/'comparison.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),float(fps),(2*width,canvas_height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f'Cannot open MP4 encoder for {path}; OpenCV needs MPEG-4 video support.')
    try:
        for name in tqdm(image_names,desc='Comparison video',leave=False):
            frame=np.zeros((canvas_height,2*width,3),dtype=np.uint8)
            for side,sub in enumerate(('renders','gt')):
                with Image.open(folder/sub/(name+'.png')) as source:
                    image=np.asarray(source.convert('RGB'))
                h,w=image.shape[:2]
                x=side*width+(width-w)//2;y=header+(height-h)//2
                frame[y:y+h,x:x+w]=image
            cv2.putText(frame,'Ours',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame,'GT',(width+12,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
            writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size==0: raise RuntimeError(f'No comparison video was written to {path}')
    print('Comparison video:',path)
    return path


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
    parser.add_argument('--video_fps',type=float,default=30.,help='Playback FPS of the side-by-side comparison.mp4 (default: 30).')
    cli=parser.parse_args()
    if cli.time is not None and not 0<=cli.time<=1: parser.error('--time must lie in [0,1]')
    if not np.isfinite(cli.video_fps) or cli.video_fps<=0: parser.error('--video_fps must be positive and finite')
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
    stats_path=root/'render_stats.json'
    timing=json.loads(stats_path.read_text()) if stats_path.exists() else {}
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
        mean_ms=float(np.mean(milliseconds))
        timing[split]={'method':name,'frames':len(cameras),'mean_ms':mean_ms,'fps':1000/mean_ms,
            'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20}
        timing.setdefault('methods',{}).setdefault(split,{})[name]=timing[split].copy()
        ordered=sorted(cameras,key=lambda camera:(camera.sequence,camera.sensor_id,camera.time,camera.image_name))
        create_comparison_video(folder,[camera.image_name for camera in ordered],cli.video_fps)
    stats_path.write_text(json.dumps(timing,indent=2))
    print(json.dumps(timing,indent=2))


if __name__=='__main__': main()
