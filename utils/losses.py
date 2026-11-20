"""
The loss terms of the framework, with its per-pixel normalization ("Reading the losses", Sec. 2):

    (1 / |Omega|) sum_{u in Omega} || A(u) - B(u) ||_1,   ||.||_1 summing the three RGB differences of one pixel

Omega is the set of evaluated pixels: the valid pixels of the view (scene/cameras.py), and for Eq. (3a) the pixels
where both normal maps are defined. Images are (C, H, W); masks are (1, H, W) bool.
"""

import torch
import torch.nn.functional as F

from utils.loss_utils import ssim


def _count(mask):
    count = mask.sum()
    if count == 0:
        raise ValueError('No evaluated pixels: the view has no valid pixel.')
    return count


def pixel_l1(prediction, target, valid):
    """Image term of Eq. (3) and (7)."""
    return ((prediction - target).abs().sum(0, keepdim=True) * valid).sum() / _count(valid)


def surface_loss(normal, depth_normal, valid):
    """Eq. (3a): absolute mismatch of rendered and depth normals over Omega_s; zero when Omega_s is empty."""
    if not valid.any():
        return normal.sum() * 0
    return ((normal - depth_normal).abs().sum(0, keepdim=True) * valid).sum() / valid.sum()


def motion_loss(prediction, previous_prediction, target, previous_target, valid):
    """Eq. (7a)-(7b): the predicted change between neighboring frames matches the observed change, pixel by pixel."""
    return pixel_l1(prediction - previous_prediction, target - previous_target, valid)


def optical_loss(h, valid):
    """Eq. (7c): mean squared Fresnel correction over Omega; h is zero off the glass."""
    return (h.square() * valid).sum() / _count(valid)


def glass_prior_loss(glass_map, mask, valid):
    """
    Binary cross-entropy between the surface-pass glass map and the training annotation M_t. Not part of Eq. (3):
    an additional prior (the Scope of the framework notes that p and o cannot be separated without one) that makes
    the Phase I geometry carry the glass, so the patches P can be extracted from it.
    """
    prediction = glass_map.clamp(1e-5, 1 - 1e-5)[valid]
    return F.binary_cross_entropy(prediction, mask[valid].float())


def dssim(prediction, target):
    """Optional D-SSIM term (lambda_dssim); the framework omits SSIM."""
    return 1.0 - ssim(prediction, target)
