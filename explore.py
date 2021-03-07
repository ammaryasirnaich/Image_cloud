#from utils import *

import utils
import open3d as o3d




def main():
    # load dataset

    #Path for Kitti Dataset

    basedir = '/media/ammar/eecs_ammar/Kitti'

 
    # Specify the dataset to load
    date = '2011_09_26'
    drive = '0009'
    dataset = utils.load_dataset(basedir, date,drive,True)
    previous_pcd = o3d.cuda.pybind.geometry.PointCloud
    out_fn = 'from_OpenCV_method.ply'

    for frameNumber in range(1,2,1):
        print(frameNumber)  
        pcd = utils.pointcloud_from_StereoImage_kitti(frameNumber,dataset)
        #previous_pcd+=pcd
        #print(type(pcd))
        #o3d.io.write_point_cloud("from_open3d_method.ply", pcd)

        
        #pcd , pcd_color = utils.cv_pointcloud_from_stereo(dataset,frameNumber)
        #utils.write_ply(out_fn, pcd, pcd_color)
        #print('%s saved' % out_fn)


    #o3d.io.write_point_cloud("copy_of_fragment.ply", previous_pcd)


if __name__ == '__main__':
    main()

    


