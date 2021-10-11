import cv2
import numpy as np
import matplotlib.pyplot as plt

import pykitti
import os
import open3d as o3d
import shutil
import struct
import time
import laspy


from pathlib import Path
from utilityPackage import parseTrackletXML as xmlParser
print(os.getcwd())


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

def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklets_types = {}
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []

    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T

            # frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            # frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
            #     tracklet.objectType]

            if absoluteFrameNumber in frame_tracklets:
                frame_tracklets[absoluteFrameNumber] += [cornerPosInVelo]
                frame_tracklets[absoluteFrameNumber] += [tracklet.objectType]
            else:
                frame_tracklets_types[absoluteFrameNumber] = [cornerPosInVelo]
                frame_tracklets_types[absoluteFrameNumber] = [tracklet.objectType]

    return (frame_tracklets, frame_tracklets_types)


def velo_2_img_projection(points):
        """ convert velodyne coordinates to camera image coordinates """

        # rough velodyne azimuth range corresponding to camera horizontal fov
        if h_fov is None:
            h_fov = (-50, 50)
        if h_fov[0] < -50:
            h_fov = (-50,) + h_fov[1:]
        if h_fov[1] > 50:
            h_fov = h_fov[:1] + (50,)

        # R_vc = Rotation matrix ( velodyne -> camera )
        # T_vc = Translation matrix ( velodyne -> camera )
        R_vc, T_vc = calib_velo2cam()

        # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
        P_ = calib_cam2cam()

        """
        xyz_v - 3D velodyne points corresponding to h, v FOV limit in the velodyne coordinates
        c_    - color value(HSV's Hue vaule) corresponding to distance(m)
                 [x_1 , x_2 , .. ]
        xyz_v =  [y_1 , y_2 , .. ]
                 [z_1 , z_2 , .. ]
                 [ 1  ,  1  , .. ]
        """
        xyz_v, c_ = point_matrix(points)

        """
        RT_ - rotation matrix & translation matrix
            ( velodyne coordinates -> camera coordinates )
                [r_11 , r_12 , r_13 , t_x ]
        RT_  =  [r_21 , r_22 , r_23 , t_y ]
                [r_31 , r_32 , r_33 , t_z ]
        """
        RT_ = np.concatenate((R_vc, T_vc), axis=1)

        # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
        for i in range(xyz_v.shape[1]):
            xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

        """
        xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
                 [x_1 , x_2 , .. ]
        xyz_c =  [y_1 , y_2 , .. ]
                 [z_1 , z_2 , .. ]
        """
        xyz_c = np.delete(xyz_v, 3, axis=0)

        # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        for i in range(xyz_c.shape[1]):
            xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

        """
        xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
        ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
                 [s_1*x_1 , s_2*x_2 , .. ]
        xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]
                 [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
        """
        xy_i = xyz_c[::] / xyz_c[::][2]
        ans = np.delete(xy_i, 2, axis=0)

        return ans, c_


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


def conver_bin_file_cloudPoint(sourcePath, destination_path,formatType='LAS'):
    ## setting up the output directory
    folder = "converted_pointcloud_files"

    dest_path = os.path.join(destination_path, folder)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    else:
        shutil.rmtree(dest_path)
        os.makedirs(dest_path)

    entries = Path(sourcePath)
    for entry in entries.iterdir():
        #print(entry.name)
        kitti_lidar_file = sourcePath + "/" + entry.name

        bin_pcd = np.fromfile(kitti_lidar_file, dtype=np.float32)

        #Reshape and drop reflection values
        # points = bin_pcd.reshape((-1, 4))[:, 0:3]

        #Reshape and include the reflection values
        lidar_points = bin_pcd.reshape((-1, 4))[:, 0:4]
        # lidar_itensity = lidar_points[:,3:]
        # lidar_xy = lidar_points[:,0:3]

        name,_ = entry.name.split(".")
        write_file_path = dest_path+"/"+name

        if(formatType =='LAS'):
            write_las(write_file_path, lidar_points)
        elif(formatType =='PLY'):
            write_ply(write_file_path, lidar_points[:, 0:3], lidar_points[:, 3:])
        else:
            # Convert to Open3D point cloud
            o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lidar_points[:,0:3]))
            intensity = np.zeros((np.size(lidar_points[:,3:]), 3))
            intensity[:, 0] = np.reshape( lidar_points[:,3:], -1)
            intensity[:, 1] = np.reshape( lidar_points[:,3:], -1)
            intensity[:, 2] = np.reshape( lidar_points[:,3:], -1)
            o3d_pcd.colors = o3d.utility.Vector3dVector(intensity)
            # o3d_pcd.colors = np.array(lidar_points[:,3:])
            o3d.io.write_point_cloud(write_file_path+'.pcd', o3d_pcd)


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
    with open(fn+".ply", 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def write_las(fn,cloudpoints,rgb_points=None):

    hdr = laspy.header.Header(file_version=1.4, point_format=7)
    outfile = laspy.file.File(fn+".las", mode="w", header = hdr)

    # assert cloudpoints.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    # if rgb_points is None:
    #     rgb_points = np.ascontiguousarray( np.ones(cloudpoints.shape).astype(np.uint8)*255 )
    # assert cloudpoints.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # xyz = np.ascontiguousarray(rawinput[:, 0:3], dtype=’float32′)
    # rgb = np.ascontiguousarray(raw[:, 4:7], dtype=’uint8′) *256     # the lidar doest has rgb hence omiited
    # i = np.ascontiguousarray(rawinput[:, 3], dtype=’float32′) *256   # un-normalize that data

    xyz = np.ascontiguousarray(cloudpoints[:,0:3], dtype='float32')
    i = np.ascontiguousarray(cloudpoints[:,3], dtype='float32') *256  # un-normalize that data
    i = np.squeeze(i)



    # outfile.x = xyz[:, 0]
    # outfile.y = xyz[:, 1]
    # outfile.z = xyz[:, 2]
    # # outfile.Red = rgb_points[:, 0]
    # # outfile.Green = rgb_points[:, 1]
    # # outfile.Blue = rgb_points[:, 2]
    # outfile.Intensity = i
    # # outfile.classification = labels
    # outfile.close()



    xmin = np.floor(np.min(xyz[:, 0]))
    ymin = np.floor(np.min(xyz[:, 1]))
    zmin = np.floor(np.min(xyz[:, 2]))
    imin = np.floor(np.min(i))
    imax = np.max(i)

    outfile.header.offset = [xmin, ymin, zmin,imin]
    outfile.header.scale = [0.001, 0.001, 0.001,0.001]

    outfile.x =  xyz[:, 0]
    outfile.y = xyz[:, 1]
    outfile.z = xyz[:, 2]
    outfile.intensity = i

    outfile.close()


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




