# from utils import *

import utils
import open3d as o3d
import os
import shutil

import numpy as np

def main():
    # load dataset

    # Path for Kitti Dataset

    basedir = '/media/ammar/eecs_ammar/Kitti'

    # Specify the dataset to load
    date = '2011_09_26'
    drive = '0009'
    dataset = utils.load_dataset(basedir, date, drive, True)
    save_result_directory = date + "_" + drive + "_stereo_cloud"
    path = os.path.join(os.getcwd(), save_result_directory)


    ## converting BIN file to PLY
    ##
    # directoryPath = "/media/ammar/eecs_ammar/Kitti/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data"
    # utils.conver_bin_file_cloudPoint(directoryPath)



    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

    frames = dataset.__len__()

    for frameNumber in range(0, frames, 1):

        ## Generate Cloudpoints from Stereo Images
        file_path = path + "/0000" + str(frameNumber) + ".ply"
        pcd = utils.pointcloud_from_StereoImage_kitti(frameNumber,dataset)
        o3d.io.write_point_cloud(file_path, pcd)




        pcd , pcd_color = utils.cv_pointcloud_from_stereo(dataset,frameNumber)
    #     #utils.write_ply(out_fn, pcd, pcd_color)
    #     #print('%s saved' % out_fn)

    # o3d.io.write_point_cloud("copy_of_fragment.ply", previous_pcd)






if __name__ == '__main__':
    main()
