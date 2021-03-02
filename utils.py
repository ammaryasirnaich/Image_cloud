import cv2
import numpy as np
import matplotlib.pyplot as plt

import pykitti
import os


# load dataset with pykitti
# Load the data. Optionally, specify the frame range to load.
def load_dataset(date, drive, calibrated=False, frame_range=None):
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
    ### END CODE HERE ###
    
    return disp_left

def depth_from_disparity(disparity,k_,t_):
        ### START CODE HERE ###
    
    # Get the focal length from the K matrix
    f = k_[0, 0]

    # Get the distance between the cameras from the t matrices (baseline)
    #b = t_left[1] - t_right[1]
    b = 0.54  # base = 0.54 meters, taken from the kitti website, looking into the sensor suite setup

    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disparity[disparity == 0] = 0.1
    disparity[disparity == -1] = 0.1

    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disparity.shape, np.single)

    # Calculate the depths 
    depth_map[:] = f * b / disp_left[:]
    
    ### END CODE HERE ###
    return depth_map


def decompose_projection_matrix(P_matrix):
    
    ### START CODE HERE ###
    
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P_matrix)
    t = t / t[3]
    
    ### END CODE HERE ###
    
    return k, r, t




def visualize(img_left_rgb,img_right_rgb,disparaity_map,depth_map):
    # Display some data
    f, ax = plt.subplots(2, 2, figsize=(15, 5))
    ax[0, 0].imshow(img_left_rgb)
    ax[0, 0].set_title('Left RGB Image (cam2)')

    ax[0, 1].imshow(img_right_rgb)
    ax[0, 1].set_title('Right RGB Image (cam3)')

    ax[1, 0].imshow(disparaity_map,cmap='gray')
    ax[1, 0].set_title('RGB Stereo Disparity')

    ax[1, 1].imshow(depth_map, cmap='flag')
    ax[1, 1].set_title('RGB Stereo Depth')

    plt.show()