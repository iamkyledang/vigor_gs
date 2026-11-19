#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from errno import EEXIST
from os import makedirs, path


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split('_')[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


import difflib
import json
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def record_implementation_change(note='Implementation snapshot at training start'):
    root=Path(__file__).resolve().parents[1]
    folder=root/'log'; folder.mkdir(exist_ok=True)
    state_file=folder/'implementation_state.json'
    previous=json.loads(state_file.read_text()) if state_file.exists() else {}
    current={}
    for base in [root,root/'arguments',root/'scene',root/'gaussian_renderer',root/'utils',root/'submodules/diff-surfel-anych',root/'submodules/diff-surfel-anych/cuda_rasterizer',root/'submodules/simple-knn',root/'lpipsPyTorch',root/'lpipsPyTorch/modules']:
        for path in base.iterdir():
            if path.is_file() and (path.suffix in ('.py','.cu','.cpp','.h','.md','.tex','.txt','.toml','.sh') or path.name=='.gitignore'):
                current[str(path.relative_to(root))]=path.read_text()
    path=root/'submodules/diff-surfel-anych/diff_surfel_anych/__init__.py'
    current[str(path.relative_to(root))]=path.read_text()
    if not state_file.exists():
        import subprocess
        for key in current:
            result=subprocess.run(['git','show','HEAD:'+key],cwd=root,capture_output=True,text=True)
            if result.returncode==0: previous[key]=result.stdout
    changed=[key for key in sorted(current.keys()|previous.keys()) if current.get(key)!=previous.get(key)]
    if not changed: return
    stamp=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y-%m-%d_%H-%M-%S_%f')
    patch=[]
    for key in changed:
        patch.extend(difflib.unified_diff(previous.get(key,'').splitlines(True),current.get(key,'').splitlines(True),fromfile='before/'+key,tofile='after/'+key))
    (folder/(stamp+'.patch')).write_text(''.join(patch))
    (folder/(stamp+'.md')).write_text(note+'\n\nChanged files:\n'+''.join('- '+key+'\n' for key in changed))
    temporary=folder/(stamp+'.state.tmp')
    temporary.write_text(json.dumps(current))
    os.replace(temporary,state_file)


def watch_implementation(interval=2.):
    print('Watching VIGOR_GS source changes; press Ctrl-C to stop.',flush=True)
    try:
        while True:
            record_implementation_change('Source change recorded by the implementation watcher')
            time.sleep(interval)
    except KeyboardInterrupt:
        print('Change watcher stopped.')
