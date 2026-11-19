"""Streaming sRGB benchmark metrics with explicit glass-pixel normalization."""
import json
import math
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


def phase_training_seconds(path,iteration=None):
    """Sum final cumulative timings per session, including resumed sessions."""
    if not path.exists(): return None
    sessions={};legacy_session=0;previous=None
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        record=json.loads(line)
        elapsed=record.get('elapsed_seconds')
        if elapsed is None or not math.isfinite(elapsed) or elapsed<0: continue
        step=record['iteration']
        # Older logs have no IDs: detect resets in elapsed time or iteration.
        if 'session_id' in record:
            key=('session',record['session_id'])
        else:
            if previous is not None and (elapsed<previous[0] or step<=previous[1]):
                legacy_session+=1
            key=('legacy',legacy_session)
        previous=(elapsed,step)
        if iteration is not None and step>iteration: continue
        sessions[key]=max(sessions.get(key,0.),elapsed)
    return sum(sessions.values()) if sessions else None


def performance_summary(root,method,split,frames,sole_method=False):
    """Rendering throughput and recorded training-loop time for one method."""
    root=Path(root)
    stats_path=root/'render_stats.json'
    stats=json.loads(stats_path.read_text()) if stats_path.exists() else {}
    timing=stats.get('methods',{}).get(split,{}).get(method)
    if timing is None:
        legacy=stats.get(split,{})
        if legacy.get('method')==method or (sole_method and 'method' not in legacy):
            timing=legacy
    mean_ms=timing.get('mean_ms') if timing and timing.get('frames')==frames else None
    fps=1000/mean_ms if mean_ms is not None and math.isfinite(mean_ms) and mean_ms>0 else None
    direct=method.startswith('direct_')
    suffix=method.rsplit('_',1)[-1]
    iteration=int(suffix) if suffix.isdigit() else None
    phases=('interface',) if direct else ('interface','reflection')
    seconds={phase:phase_training_seconds(root/phase/'training.jsonl',
        iteration if phase==('interface' if direct else 'reflection') else None) for phase in phases}
    total=sum(seconds.values()) if all(value is not None for value in seconds.values()) else None
    duration=None
    if total is not None:
        hours,minutes=divmod(int(total//60),60)
        duration=f'{hours} hours and {minutes} mins'
    return {'FPS':fps,'Training Time':duration,'Training Time (seconds)':total,
            'Training Time by Phase (seconds)':seconds}


def update_performance(model_paths,split='test'):
    """Refresh timing fields without recomputing or replacing image metrics."""
    for model_path in model_paths:
        root=Path(model_path)
        results=json.loads((root/'results.json').read_text())
        methods=[method for method in results if (root/split/method/'renders').is_dir()]
        if not methods: raise ValueError(f'No existing results match {split} renders under {root}')
        for method in methods:
            frames=sum(1 for _ in (root/split/method/'renders').glob('*.png'))
            performance=performance_summary(root,method,split,frames,len(methods)==1)
            results[method].update(performance)
            print(method,json.dumps(performance,indent=2))
        (root/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))


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
        methods=[method for method in sorted(parent.iterdir()) if (method/'renders').is_dir()]
        for method in methods:
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
            summary.update(performance_summary(root,method.name,split,len(files),len(methods)==1))
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
    parser.add_argument('--performance_only',action='store_true',help='Update FPS/training time in existing results.json without recomputing image metrics.')
    args=parser.parse_args()
    if args.performance_only: update_performance(args.model_paths,args.split)
    else: evaluate(args.model_paths,args.device,args.skip_lpips,args.split)
