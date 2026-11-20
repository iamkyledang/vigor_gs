"""
Where the per-view data lives, so that large scenes (e.g. WayveScene101 at 3271x1224) fit in both VRAM and host RAM.

Per-view data is every camera's original image, clean image and glass mask, plus Phase II's cached direct images.
`--data_device` chooses where it is kept:

    cuda   everything on the GPU: the fastest, and the original behaviour
    cpu    everything in pinned host RAM, copied to the GPU when the view is used
    auto   on the GPU until `--data_vram_fraction` of the card's memory is used, then in pinned host RAM, so the data
           moves between RAM and VRAM view by view instead of overflowing either of them

Camera images are kept as uint8, exactly as decoded from disk, and turned into float only when a view is used. They
take a quarter of the memory of float32 and give the same values, since the source images are 8-bit anyway.

`LazyImage` stands in for a PIL image in the scene reader: it remembers only the path and size, so reading a scene
neither holds every frame's decoded pixels in RAM nor keeps hundreds of files open.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

DEFAULT_VRAM_FRACTION = 0.35


class DataPlacer:
    """
    Decides, tensor by tensor, whether per-view data stays on the GPU or goes to host RAM.

    budget_bytes overrides the budget computed from vram_fraction (for tests).
    """

    def __init__(self, mode: str = 'cuda', vram_fraction: float = DEFAULT_VRAM_FRACTION, budget_bytes: int = None):
        mode = str(mode)
        if mode.startswith('cuda'):
            mode = 'cuda'
        if mode not in ('cuda', 'cpu', 'auto'):
            raise ValueError(f"data_device must be 'cuda', 'cpu' or 'auto', got {mode!r}")
        self.mode = mode
        if budget_bytes is not None:
            self.budget = int(budget_bytes)
        elif mode == 'auto':
            self.budget = int(vram_fraction * torch.cuda.get_device_properties(0).total_memory)
        else:
            self.budget = 0
        self.gpu_bytes = 0
        self.cpu_bytes = 0
        self._lock = threading.Lock()

    def place(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move the tensor to where it should live; it is always read back through .cuda()."""
        nbytes = tensor.numel() * tensor.element_size()
        with self._lock:
            on_gpu = self.mode == 'cuda' or (self.mode == 'auto' and self.gpu_bytes + nbytes <= self.budget)
            if on_gpu:
                self.gpu_bytes += nbytes
            else:
                self.cpu_bytes += nbytes
        if on_gpu:
            return tensor.to('cuda')
        return pin(tensor.detach().to('cpu'))

    def summary(self) -> str:
        budget = f', VRAM budget {self.budget / 2**30:.2f} GB' if self.mode == 'auto' else ''
        return f'per-view data ({self.mode}{budget}): {self.gpu_bytes / 2**30:.2f} GB on the GPU, {self.cpu_bytes / 2**30:.2f} GB in host RAM'


def pin(tensor: torch.Tensor) -> torch.Tensor:
    """Page-locked host memory makes the copy to the GPU faster; fall back to ordinary memory where pinning fails."""
    if not torch.cuda.is_available():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


_PLACERS = {}


def get_placer(mode='cuda', vram_fraction=DEFAULT_VRAM_FRACTION) -> DataPlacer:
    """One placer per setting and process, so every camera of a run (and Phase II's cache) shares one VRAM budget."""
    key = (str(mode), float(vram_fraction))
    if key not in _PLACERS:
        _PLACERS[key] = DataPlacer(*key)
    return _PLACERS[key]


class LazyImage:
    """
    The path, size and mode of an image file, opened only when its pixels are needed.

    `convert` is the PIL mode the pixels are converted to when opened (e.g. 'L' for masks); `mode` reports that mode.
    """

    def __init__(self, path: str, convert: str = None):
        self.filename = path
        self.convert = convert
        with Image.open(path) as image:
            self.size = image.size
            self.mode = convert or image.mode

    def open(self) -> Image.Image:
        """The decoded image; the file is closed again and nothing is cached here."""
        with Image.open(self.filename) as image:
            image = image.convert(self.convert) if self.convert and image.mode != self.convert else image.copy()
        return image

    def __repr__(self):
        return f'LazyImage({self.filename!r}, size={self.size}, mode={self.mode!r})'


def image_to_tensor(image, resolution):
    """
    (C, H, W) tensor of a PIL image or LazyImage resized to `resolution` (width, height), like PILtoTorch but without
    dividing by 255: uint8 for 8-bit images, float in [0, 1] otherwise.
    """
    pil = image.open() if isinstance(image, LazyImage) else image
    array = np.array(pil.resize(resolution))
    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1) if tensor.ndim == 3 else tensor.unsqueeze(0)
    if tensor.dtype != torch.uint8:
        tensor = tensor.float() / 255.0
    return tensor.contiguous()


class BoundedExecutor:
    """
    A thread pool for saving images that blocks once `max_pending` jobs wait, so rendering cannot pile up the
    images still to be written in host RAM when the GPU renders faster than the disk writes.
    """

    def __init__(self, max_workers: int = 8, max_pending: int = 32):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._slots = threading.BoundedSemaphore(max_pending)

    def submit(self, fn, *args, **kwargs):
        self._slots.acquire()
        try:
            future = self._pool.submit(fn, *args, **kwargs)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._pool.shutdown(wait=True)
        return False
