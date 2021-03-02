import utils

# load dataset
#Path for Kitti Dataset

basedir = '/home/'
# Specify the dataset to load
date = '2011_09_26'
drive = '0009'
load_dataset(date,drive,True)



#Compute the disparity map
# Calling the settings for stereo disparity matcher
stereo_matcher_SGBM = stereo_setting_disparity()
#computer the disparity using stereo images

frame =10
img_left_rgb = np.array(dataset.get_cam2(frame))
img_right_rgb = np.array(dataset.get_cam3(frame))

# converting RGB to Grey images for computing disparity map
img_left_grey = cv2.cvtColor(img_left_rgb , cv2.COLOR_RGB2GRAY)
img_right_grey = cv2.cvtColor(img_right_rgb , cv2.COLOR_RGB2GRAY)


disparaity_map = compute_disparity_map(img_left_grey, img_right_grey,stereo_matcher_SGBM)


# Read the calibration 
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
depth_map= depth_from_disparity(disparaity_map, k_, t_)

visualize()