
from utils import *
import utils
# import open3d
# import os
# import shutil
# import pykitti
# import time
# import numpy as np

from  utilityPackage import  kitti_util
from utilityPackage.kitti_object import kitti_object, show_lidar_with_depth, show_lidar_on_image, \
                         show_image_with_boxes, show_lidar_topview_with_boxes



def main():
    # pass
    #load dataset

    # Path for Kitti Dataset

    basedir = '/media/ammar/eecs_ammar/Kitti'
    #
    # # Specify the dataset to load
    date = '2011_09_26'
    drive = '0009'
    dataset =  load_dataset(basedir, date, drive, True)
    tracklet_xml_path = dataset.data_path + "/tracklet_labels.xml"

    print(dataset.data_path)



    # # converting BIN file to PLY "
    save_result_directory = date + "_" + drive + "_stereo_cloud"
    path = os.path.join(os.getcwd(), save_result_directory)
    directoryPath = "/media/ammar/eecs_ammar/Kitti/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data"
    utils.conver_bin_file_cloudPoint(directoryPath,"/media/ammar/eecs_ammar/Kitti/2011_09_26","PCD")


    ## use the "opencv" or "open3d" library to generate pointcloud data
    #utils.generate_pointcloud_from_stere(path,dataset,"opencv",False)
    # utils.generate_pointcloud_from_stere(path, dataset,"open3d", False)



if __name__ == '__main__':
    main()
