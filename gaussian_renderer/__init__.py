"""Public rendering API for the direct and reflected VIGOR-GS fields."""
import sys
from pathlib import Path

# Select this checkout's extension. CUDA is loaded only when rendering is called,
# so CPU geometry tests and optical-network imports do not require a built backend.
_extension_path = str(Path(__file__).resolve().parents[1] / 'submodules/diff-surfel-anych')
if _extension_path not in sys.path:
    sys.path.insert(0, _extension_path)

from gaussian_renderer.interface import render_interface
from gaussian_renderer.reflection import tile_candidates, render_reflection
from gaussian_renderer.composition import OpticalCorrection, ray_geometry, render_composite
from scene.gaussian_model import GaussianModel

render = render_interface

__all__ = ['render', 'render_interface', 'render_reflection', 'render_composite',
           'tile_candidates', 'ray_geometry', 'OpticalCorrection', 'GaussianModel']
