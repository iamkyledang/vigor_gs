"""
Renderers of the two VIGOR-GS fields:

    interface.py     2DGS direct and surface passes of G_I (Eq. 2)
    reflection.py    G_R along per-pixel reflected rays (Sec. 3, "Render")
    composition.py   Fresnel weight, optical correction h_psi and the blend of Eq. (6)

The CUDA extension is the one built in this checkout's submodules/diff-surfel-anych (`setup.py build_ext
--inplace`): it adds the reflected-ray renderer to the 2DGS rasterizer. Putting its folder first on sys.path keeps
an older diff_surfel_anych installed in the environment (e.g. for RT-Splatting) untouched.
"""

import os
import sys

_EXTENSION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'submodules', 'diff-surfel-anych')
if _EXTENSION not in sys.path:
    sys.path.insert(0, _EXTENSION)
