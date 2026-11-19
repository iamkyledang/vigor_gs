"""Calibrated camera. Large target tensors stay in host memory by default."""
import numpy as np
import torch
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


class Camera:
    def __init__(self, info, data_device='cpu'):
        self.uid = info.uid
        self.colmap_id = info.uid
        self.sensor_id = info.sensor_id
        self.R, self.T = info.R, info.T
        self.image_name = info.image_name
        self.time, self.sequence = info.time, info.sequence
        self.image_width, self.image_height = info.width, info.height
        self.fx, self.fy = float(info.K[0,0]), float(info.K[1,1])
        self.cx, self.cy = float(info.K[0,2]), float(info.K[1,2])
        self.FoVx = 2*np.arctan(self.image_width/(2*self.fx))
        self.FoVy = 2*np.arctan(self.image_height/(2*self.fy))
        self.data_device = torch.device(data_device)
        def tensor(image):
            return torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).to(self.data_device)
        # Keep the dataset as bytes; decode only the sampled frame to float RGB.
        self._original_image = tensor(info.image)
        self._clean_image = tensor(info.clean_image) if info.clean_image is not None else None
        self.gt_transparent_mask = torch.from_numpy(info.transparent_mask.copy())[None].to(self.data_device)
        self.valid_mask = torch.from_numpy(info.valid_mask.copy())[None].to(self.data_device)
        self.znear, self.zfar = 0.01, 1000.
        self.world_view_transform = torch.tensor(getWorld2View2(self.R,self.T)).T.cuda()
        projection = getProjectionMatrix(self.znear,self.zfar,self.FoVx,self.FoVy)
        projection[0,2] = 2*self.cx/self.image_width-1
        projection[1,2] = 2*self.cy/self.image_height-1
        self.projection_matrix = projection.T.cuda()
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3,:3]

    @property
    def original_image(self):
        return self._original_image.float().div_(255)

    @property
    def clean_image(self):
        return self._clean_image.float().div_(255) if self._clean_image is not None else None

    @clean_image.setter
    def clean_image(self, value):
        self._clean_image=None if value is None else value.detach().clamp(0,1).mul(255).round().byte().to(self.data_device)
