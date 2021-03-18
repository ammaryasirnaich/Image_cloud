# from utils import *

import utils
import open3d
import os
import shutil
import pykitti
import time
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
    #print (dataset.calib.P_rect_00)


    ## converting BIN file to PLY
    # directoryPath = "/media/ammar/eecs_ammar/Kitti/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data"
    # utils.conver_bin_file_cloudPoint(directoryPath)


    ## use the "opencv" or "open3d" library to generate pointcloud data
    #utils.generate_pointcloud_from_stere(path,dataset,"opencv",False)
    utils.generate_pointcloud_from_stere(path, dataset,"open3d", False)



if __name__ == '__main__':
    main()
