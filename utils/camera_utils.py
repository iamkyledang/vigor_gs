"""Camera adapters retained in the original RT-Splatting module."""
from scene.cameras import Camera


def loadCam(args, id, cam_info, resolution_scale):
    return Camera(cam_info, args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    return [loadCam(args,i,info,resolution_scale) for i,info in enumerate(cam_infos)]


def camera_to_JSON(id, camera):
    return {'id':id,'img_name':camera.image_name,'time':camera.time,
            'width':camera.image_width,'height':camera.image_height,
            'position':camera.camera_center.tolist(),'fx':camera.fx,'fy':camera.fy}
