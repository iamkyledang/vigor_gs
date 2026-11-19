"""Streaming sRGB benchmark metrics with explicit glass-pixel normalization."""
import json
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from utils.loss_utils import ssim


def read_image(path,device,gray=False):
    image=np.asarray(Image.open(path).convert('L' if gray else 'RGB')).copy()
    value=torch.from_numpy(image).float().div_(255)
    return (value[None,None] if gray else value.permute(2,0,1)[None]).to(device)


def masked_psnr(prediction,target,mask):
    count=mask.sum()*prediction.shape[1]
    if count==0: return None
    mse=((prediction-target).square()*mask).sum()/count
    return float(-10*torch.log10(mse.clamp_min(1e-12)))


@torch.no_grad()
def evaluate(model_paths,device='cuda',skip_lpips=False,split='test'):
    perceptual=None
    if not skip_lpips:
        from lpipsPyTorch import LPIPS
        perceptual=LPIPS(net_type='vgg').to(device).eval()
    for model_path in model_paths:
        root=Path(model_path);parent=root/split
        if not parent.exists(): raise FileNotFoundError(f'No {split} renders at {parent}; run render.py first.')
        results={};per_view={}
        for method in sorted(parent.iterdir()):
            if not (method/'renders').is_dir(): continue
            values={}
            files=sorted((method/'renders').glob('*.png'))
            if not files: raise ValueError(f'No rendered images in {method}')
            for path in tqdm(files,desc=method.name):
                pred=read_image(path,device);target=read_image(method/'gt'/path.name,device)
                mask=read_image(method/'transparent_masks'/path.name,device,True)>0.5
                valid_path=method/'valid_masks'/path.name
                valid=read_image(valid_path,device,True)>0.5 if valid_path.exists() else torch.ones_like(mask)
                if pred.shape!=target.shape: raise ValueError(f'Size mismatch for {path.name}')
                record={'PSNR':masked_psnr(pred,target,torch.ones_like(mask)),
                    'SSIM':float(ssim(pred,target)),
                    'Valid PSNR':masked_psnr(pred,target,valid),
                    'Glass PSNR':masked_psnr(pred,target,mask&valid),
                    'Opaque PSNR':masked_psnr(pred,target,(~mask)&valid),
                    # These padded metrics reproduce the RT-Splatting comparison protocol.
                    'Masked PSNR (RT padded)':masked_psnr(pred*mask,target*mask,torch.ones_like(mask)),
                    'Masked SSIM (RT padded)':float(ssim(pred*mask,target*mask)) if mask.any() else None}
                if perceptual is not None:
                    record['LPIPS']=float(perceptual(pred*2-1,target*2-1))
                    record['LPIPS (RT input)']=float(perceptual(pred,target))
                    record['Masked LPIPS (RT padded)']=float(perceptual(pred*mask,target*mask)) if mask.any() else None
                values[path.name]=record
            keys=next(iter(values.values())).keys()
            summary={key:float(np.mean([v[key] for v in values.values() if v[key] is not None]))
                     if any(v[key] is not None for v in values.values()) else None for key in keys}
            results[method.name]=summary;per_view[method.name]=values
            print(method.name,json.dumps(summary,indent=2))
        if not results: raise ValueError(f'No render methods found under {parent}')
        (root/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
        (root/'per_view.json').write_text(json.dumps(per_view,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('-m','--model_paths',nargs='+',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--skip_lpips',action='store_true',help='Explicitly omit LPIPS; useful for dependency smoke checks.')
    parser.add_argument('--split',choices=['train','test'],default='test')
    args=parser.parse_args()
    evaluate(args.model_paths,args.device,args.skip_lpips,args.split)
