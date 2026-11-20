"""Pieces shared by the training scripts: seeding, view sampling, the training log and output folders."""

import json
import os
import random
import time

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ViewSampler:
    """Every training view once per epoch, in a new random order each epoch (the 3DGS view stack)."""

    def __init__(self, cameras):
        self.cameras = list(cameras)
        self.stack = []

    def next(self):
        if not self.stack:
            self.stack = random.sample(self.cameras, len(self.cameras))
        return self.stack.pop()


class TrainingLog:
    """
    `<phase dir>/training.jsonl`: one JSON record per report. `elapsed_seconds` is the wall-clock time of the
    optimization loop of this session (resumed runs start a new session); metrics.py sums sessions per phase.
    """

    def __init__(self, phase_dir):
        self.path = os.path.join(phase_dir, 'training.jsonl')
        self.session = str(time.time_ns())
        self.start = time.perf_counter()

    def write(self, iteration, **values):
        torch.cuda.synchronize()
        record = {'iteration': iteration, 'session_id': self.session, 'elapsed_seconds': time.perf_counter() - self.start,
                  'peak_vram_mb': torch.cuda.max_memory_allocated() / 2**20}
        record.update({k: float(v) if torch.is_tensor(v) else v for k, v in values.items()})
        with open(self.path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        return record


def phase_dir(model_path, phase, resume):
    """`<model>/<phase>/`; a finished checkpoint is never overwritten by a fresh run."""
    folder = os.path.join(model_path, phase)
    checkpoint = os.path.join(folder, 'latest.pth')
    if os.path.exists(checkpoint) and not resume:
        raise FileExistsError(f'{checkpoint} exists: pass --resume to continue it, or use a new -m folder')
    os.makedirs(folder, exist_ok=True)
    return folder
