import cv2
import numpy as np
import matplotlib.pyplot as plt

import pykitti
import os
import open3d as o3d
import shutil
import struct
import time


from pathlib import Path

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
    #print('\nDrive: ' + str(dataset.drive))
    #print('\nFrame range: ' + str(dataset.frames))

    #if calibrated:
    #    print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
    #    print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
    #    print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

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


def depth_from_disparity(disparity,k_=None,t_=None,focal_length=None):
    
    # Get the focal length from the K matrix

    #f = k_[0, 0]

    if k_.any()!= None:
        f = k_[0, 0]
    
    if focal_length != None:
        f = focal_length

    #print("focal length of kitti lense:{}".format(f))
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
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P_matrix)
    #t = t / t[3]
    return k, r, t

def pointcloud_from_StereoImage_kitti(frameNumber,dataset):
    #Compute the disparity map
    # Calling the settings for stereo disparity matcher
    stereo_matcher_SGBM = stereo_setting_disparity()
    #computer the disparity using stereo images


    #img_left_rgb = np.array(cv2.imread("left.png"))
    #img_right_rgb = np.array(cv2.imread("right.png"))

    img_left_rgb = np.array(dataset.get_cam2(frameNumber))
    img_right_rgb = np.array(dataset.get_cam3(frameNumber))
    #print(img_left_rgb.shape)

    # converting RGB to Grey images for computing disparity map
    img_left_grey = cv2.cvtColor(img_left_rgb , cv2.COLOR_RGB2GRAY)
    img_right_grey = cv2.cvtColor(img_right_rgb , cv2.COLOR_RGB2GRAY)

    #print(img_left_grey.shape)

    disparaity_map = compute_disparity_map(img_left_grey, img_right_grey,stereo_matcher_SGBM)



    # Read the calibration 
    # As the P matrix is the combination intrinsic parameters 𝐾  and the extrinsic rotation 𝑅,
    # and translation t as follows:   
    P_matrix=  dataset.calib.P_rect_00
    #camera_extrinsic

    # Decompose each matrix
    #k_method, r_, t_ = decompose_projection_matrix(P_matrix)
    
    camera_intrinsic_para, camera_rotation_para, camera_trans_para,_,_,_,_ = cv2.decomposeProjectionMatrix(P_matrix)

    # Display the matrices
    # print("camera_intrinsic_para")
    # print(camera_intrinsic_para)

    # print("decompouse matrics")
    # print("k_ \n", k_)
    # print("\nr_ \n", r_)
    # print("\nt_ \n", t_)
    
    # Get the depth map by calling the above function
    depth_map = depth_from_disparity(disparaity_map, camera_intrinsic_para, camera_trans_para)


    color_raw = o3d.geometry.Image(img_left_rgb)
    depth_raw = o3d.geometry.Image(depth_map)

    # ** Note add Externsic parameters below
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)


    # print("RGBD information:")
    # print(rgbd_image)

    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix =  camera_intrinsic_para
    

    #cam = o3d.camera.PinholeCameraParameters()
    #cam.intrinsic = camera_intrinsic_para
    #cam.extrinsic = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    extrinsic_para = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 0.], [0., 0., 1., 0.]])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image( rgbd_image, o3d.camera.PinholeCameraIntrinsic(cam))
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image( rgbd_image, cam.intrinsic,cam.extrinsic )
    
    clout = o3d.geometry.PointCloud()

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    #visualize(img_left_rgb,img_right_rgb,disparaity_map,depth_map,rgbd_image,True)


    # adding RGB color to cloud points
    # colors = cv2.cvtColor(img_left_rgb, cv2.COLOR_BGR2RGB)
    mask = disparaity_map > disparaity_map.min()
    cloudpoint_color = img_left_rgb[mask]
    pcd.colors = o3d.utility.Vector3dVector(cloudpoint_color)

    return pcd

def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated

    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()


def conver_bin_file_cloudPoint(directoryPath):
    ## setting up the output directory
    folder = "output"
    path = os.path.join( os.getcwd() , folder )

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


    entries = Path(directoryPath)
    for entry in entries.iterdir():
        #print(entry.name)
        path = directoryPath+"/"+entry.name

        # print(path)

        bin_pcd = np.fromfile(path, dtype=np.float32)

        #Reshape and drop reflection values
        points = bin_pcd.reshape((-1, 4))[:, 0:3]

        #Reshape and include the reflection values
        points = bin_pcd.reshape((-1, 4))[:, 0:3]

        # Convert to Open3D point cloud
        o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        name,_ = entry.name.split(".")
        file_path = "/media/ammar/eecs_ammar/Kitti/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/"+"output/"+name+".ply"
        print(path)
        print(name)
        o3d.io.write_point_cloud(file_path, o3d_pcd)


def visualize(img_left_rgb,img_right_rgb,disparaity_map,depth_map, rgbd = None,withOutRGBD =False):
    
    plt.figure(figsize=(20,10))

    plt.subplot(3, 2, 1)
    plt.title('Left RGB image')
    plt.axis("off")
    plt.imshow(img_left_rgb)
    #plt.colorbar()


    plt.subplot(3, 2, 2)
    plt.title('Right RGB image')
    plt.axis("off")
    plt.imshow(img_right_rgb)
    #plt.colorbar()
    
    #
    plt.subplot(3, 2, 3)
    plt.title('Disparity Map')
    plt.axis("off")
    plt.imshow(disparaity_map,cmap ="plasma")
    #plt.colorbar()
    #
    #
    plt.subplot(3, 2, 4)  # viridis
    plt.title('Depth Estimation Map')
    plt.imshow(depth_map, cmap ="plasma")
    #plt.colorbar()

    if withOutRGBD == True: 
        plt.subplot(3, 2, 5)
        plt.title('RGBD color Map')
        plt.axis("off")
        plt.imshow(rgbd.color)
        #plt.colorbar()
        plt.subplot(3, 2, 6)
        plt.title('RGBD Depth Map')
        plt.axis("off")
        plt.imshow(rgbd.depth, cmap ="plasma")
        #plt.colorbar()


    #plt.show()
    plt.savefig('3D_scene_images.png')


def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''

    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def cv_pointcloud_from_stereo(dataset,frameNumber):

    print('loading images...')

    imgL = cv2.pyrDown(np.array(dataset.get_cam2(frameNumber)))
    imgR = cv2.pyrDown(np.array(dataset.get_cam3(frameNumber)))

    # imgL = cv2.pyrDown(cv2.imread(cv2.samples.findFile('left.png')))  # downscale images for faster processing
    # imgR = cv2.pyrDown(cv2.imread(cv2.samples.findFile('right.png')))
    #

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    # f = 0.8*w                          # guess for focal length
    f = 721.5377
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    
    points = cv2.reprojectImageTo3D(disp, Q)  # getting depth from disparity map
    
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    cloudpoint = points[mask]
    cloudpoint_color = colors[mask]

    #cv2.imshow('left', imgL)
    #cv2.imshow('right',imgR)
    #cv2.imshow('disparity', (disp-min_disp)/num_disp)

    depth_map = depth_from_disparity((disp-min_disp)/num_disp,None,None,f)
    disparity_map = disp
    visualize(imgL,imgR,disparity_map,depth_map,None)

    return cloudpoint , cloudpoint_color

def generate_pointcloud_from_stere(destinationPath,kitti_dateset,Library = None, writepcd = None):

        if not os.path.exists(destinationPath):
            os.makedirs(destinationPath)
        else:
            shutil.rmtree(destinationPath)
            os.makedirs(destinationPath)

        # get total number of frames
        frames = kitti_dateset.__len__()

        # initialize parameter for open3d visualization
        pcd = o3d.geometry.PointCloud()

        # Disable Open 3D viusalization
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd)
        # render_option = vis.get_render_option()
        # render_option.point_size = 0.01
        # to_reset_view_point = True

        if Library is None or Library == "open3d":
            ## Generating Pointclouds from Stere Images using open3D library
            # Note ! The below code write saves the point cloud in .ply for
            # also but with no intensity values. Open3D supports standard
            # format with not intensity pointlcoud values.
            print("open3d Block called")
            for frameNumber in range(0, frames, 1):
                stereCloud = pointcloud_from_StereoImage_kitti(frameNumber, kitti_dateset)
                pcd.points = stereCloud.points
                pcd.colors = stereCloud.colors

                # Disable open 3D visualization Corresponsing to 401 line
                #status = vis.update_geometry(pcd)
                # if to_reset_view_point:
                #     vis.reset_view_point(True)
                #     to_reset_view_point = False
                # vis.poll_events()
                # vis.update_renderer()
                # time.sleep(0.2)

                if writepcd:
                    ## writing to a folder
                    file_path = destinationPath + "/0000" + str(frameNumber) + ".ply"
                    # o3d.io.write_point_cloud(file_path, pcd)
                    xyz_point = np.asarray(pcd.points)
                    rgb_point = np.asarray(pcd.colors)
                    write_ply(file_path, xyz_point,rgb_point )

        elif Library=="opencv":
            print("openCV Block called")
            for frameNumber in range(0, frames, 1):
                pcd, pcd_color = cv_pointcloud_from_stereo(kitti_dateset, frameNumber)
                if writepcd:
                    file_path = destinationPath + "/0000" + str(frameNumber) + ".ply"
                    write_ply(file_path, pcd, pcd_color)

        else:
            return

        # Disable Open3D visualization Refer to line 401
        #vis.destroy_window()




