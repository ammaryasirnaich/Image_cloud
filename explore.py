import cv2
import numpy as np
import matplotlib.pyplot as plt

import pykitti
import os
import open3d as o3d

# print(os.getcwd())


# load dataset with pykitti
# Load the data. Optionally, specify the frame range to load.
def load_dataset(basedir, date, drive, calibrated=False, frame_range=None):
    """
    Loads the dataset with `date` and `drive`.
    
    Parameters
    ----------
    date        : Dataset creation date.
    drive       : Dataset drive.
    calibrated  : Flag indicating if we need to parse calibration data. Defaults to `False`.
    frame_range : Range of frames. Defaults to `None`.

    Returns
    -------
    Loaded dataset of type `raw`.
    """
    #dataset = pykitti.raw(basedir, date, drive,frames=range(0, 20, 5))
    dataset = pykitti.raw(basedir, date, drive)
   
    
    # Load the data
    if calibrated:
        dataset._load_calib()  # Calibration data are accessible as named tuples

    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive: ' + str(dataset.drive))
    print('\nFrame range: ' + str(dataset.frames))

    if calibrated:
        print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
        print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
        print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    return dataset


# set stereo Settings
def stereo_setting_disparity():
        
    ### START CODE HERE ###
    
    # Parameters
    num_disparities = 6*16
    block_size = 11
    
    min_disparity = 0
    window_size = 6

    # Stereo SGBM matcher
    stereo_matcher_SGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    return stereo_matcher_SGBM


def compute_disparity_map(img_left, img_right, stereo_matcher_SGBM):
    # Compute the left disparity map
    disp_left = stereo_matcher_SGBM.compute(img_left, img_right).astype(np.float32)/16   
    return disp_left


def depth_from_disparity(disparity,k_,t_):
    
    # Get the focal length from the K matrix
    f = k_[0, 0]
    
    print("focal length of kitti lense:{}".format(f))
    # Get the distance between the cameras from the t matrices (baseline)
    #b = t_left[1] - t_right[1]
    
    b = 0.54  # Taken from the kitti website, looking into the sensor suite setup its 0.54 meters

    
    
    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disparity[disparity == 0] = 0.1
    disparity[disparity == -1] = 0.1

    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disparity.shape, np.single)

    # Calculate the depths 
    depth_map[:] = f * b / disparity[:]
    
    return depth_map


def decompose_projection_matrix(P_matrix):
    
    ### START CODE HERE ###
    
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P_matrix)
    t = t / t[3]
    
    ### END CODE HERE ###
    
    return k, r, t







def main():
    # load dataset

    #Path for Kitti Dataset

    basedir = '/home/'
    # Specify the dataset to load
    date = '2011_09_26'
    drive = '0009'
    dataset = load_dataset(basedir, date,drive,True)


    #Compute the disparity map
    # Calling the settings for stereo disparity matcher
    stereo_matcher_SGBM = stereo_setting_disparity()
    #computer the disparity using stereo images

    frame =10
    img_left_rgb = np.array(dataset.get_cam2(frame))
    img_right_rgb = np.array(dataset.get_cam3(frame))
    #print(img_left_rgb.shape)

    # converting RGB to Grey images for computing disparity map
    img_left_grey = cv2.cvtColor(img_left_rgb , cv2.COLOR_RGB2GRAY)
    img_right_grey = cv2.cvtColor(img_right_rgb , cv2.COLOR_RGB2GRAY)

    print(img_left_grey.shape)

    disparaity_map = compute_disparity_map(img_left_grey, img_right_grey,stereo_matcher_SGBM)
    
    
        # Read the calibration 
    # As the P matrix is the combination intrinsic parameters 𝐾  and the extrinsic rotation 𝑅,
    # and translation t as follows:
    
    P_matrix=  dataset.calib.P_rect_00

    # Decompose each matrix
    k_, r_, t_ = decompose_projection_matrix(P_matrix)

    # Display the matrices
    # print("P_matrix")
    # print(P_matrix)


    # print("decompouse matrics")
    # print("k_ \n", k_)
    # print("\nr_ \n", r_)
    # print("\nt_ \n", t_)
    
    # Get the depth map by calling the above function
    depth_map = depth_from_disparity(disparaity_map, k_, t_)

    print(type(img_left_rgb))
    print(type(depth_map))

    color_raw = o3d.geometry.Image(img_left_rgb)
    depth_raw = o3d.geometry.Image(depth_map)

    # color_raw = o3d.io.read_image(img_left_rgb)
    # depth_raw = o3d.io.read_image(depth_map)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
    # print(rgbd_image)

    # plt.subplot(1, 2, 1)
    # plt.title('Kitti grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('kitti depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()
    
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix =  k_
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image( rgbd_image, o3d.camera.PinholeCameraIntrinsic(cam) )
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # o3d.visualization.draw_geometries([pcd], zoom=0.5)
    o3d.visualization.draw_geometries([pcd])


        



if __name__ == '__main__':
    # main()
    # generate some neat n times 3 matrix using a variant of sync function
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    print('xyz')
    print(xyz)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("../../TestData/sync.ply")
    o3d.visualization.draw_geometries([pcd_load])

    # convert Open3D.o3d.geometry.PointCloud to numpy array
    xyz_load = np.asarray(pcd_load.points)
    print('xyz_load')
    print(xyz_load)

    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    img = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
    o3d.io.write_image("../../TestData/sync.png", img)
    o3d.visualization.draw_geometries([img])
    


