import open3d as o3d
import camera
import cv2
import numpy as np

def point_cloud(camera, depth):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.
    https://codereview.stackexchange.com/questions/79032/generating-a-3d-point-cloud
    """
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np .arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth / 256.0, np.nan)
    x = np.where(valid, z * (c - camera.cx) / camera.fx, 0)
    y = np.where(valid, z * (r - camera.cy) / camera.fy, 0)
    xyz = np.zeros((x.shape[0]*x.shape[1], 3))
    xyz[:, 0] = np.reshape(x, -1)
    xyz[:, 1] = np.reshape(y, -1)
    xyz[:, 2] = np.reshape(z, -1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

# cam = o3d.camera.PinholeCameraIntrinsic()
# cam.set_intrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

xtion = camera.AsusXtion2()
color, depth = xtion.read_rgb_depth_image(show=True)
# todo : calculate intrinsic matrix
cloud = point_cloud(xtion, depth)
print('cloud: ', cloud)

# color = o3d.geometry.Image((color).astype(np.uint8))
# depth = o3d.geometry.Image((depth).astype(np.uint8))
color = o3d.io.read_image('img/color.png')
depth = o3d.io.read_image('img/depth.png')
# cv2.imshow('color', color)
# cv2.imshow('depth', depth)
# cv2.waitKey(0)
# intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx,fy, cx, cy)
# intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
# cam = o3d.camera.PinholeCameraParameters()
# cam.intrinsic = intrinsic
# cam.extrinsic = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.]])
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# show point cloud
pcd.transform([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
o3d.visualization.draw_geometries([cloud])