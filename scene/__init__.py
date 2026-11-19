"""Scene loading without implicit split changes or model-output side effects."""
from scene.dataset_readers import readVigorSceneInfo
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos


class Scene:
    def __init__(self, args, gaussians=None, load_iteration=None, shuffle=False):
        self.model_path = args.model_path
        self.gaussians = gaussians
        info = readVigorSceneInfo(args)
        self.cameras_extent = info.nerf_normalization['radius']
        self.center = info.nerf_normalization['center']
        self.point_cloud = info.point_cloud
        self.train_cameras = cameraList_from_camInfos(info.train_cameras, 1.0, args)
        self.test_cameras = cameraList_from_camInfos(info.test_cameras, 1.0, args)
        if gaussians is not None:
            gaussians.create_from_pcd(info.point_cloud, self.cameras_extent)

    def getTrainCameras(self, scale=1.0): return self.train_cameras
    def getTestCameras(self, scale=1.0): return self.test_cameras
