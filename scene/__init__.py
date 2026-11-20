"""A scene: its training and test cameras, SfM points and normalization."""

from scene.cameras import Camera
from scene.dataset_readers import readVigorSceneInfo


class Scene:
    """
    Loads every camera of `args.source_path` at `args.resolution`.

    `center` and `extent` (the radius that holds the training cameras) set the spatial learning rates, the
    deformation input normalization and the size of the reflection-field initialization.
    """

    def __init__(self, args, load_clean=True):
        info = readVigorSceneInfo(args.source_path, args.clean_images, ignore_clean_images=args.ignore_clean_images or not load_clean,
                                  init_points=args.init_points, seed=args.seed)
        self.center = info.center
        self.extent = info.extent
        self.point_cloud = info.point_cloud
        self.train_cameras = [Camera(c, args.resolution, args.data_device) for c in info.train_cameras]
        self.test_cameras = [Camera(c, args.resolution, args.data_device) for c in info.test_cameras]

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
