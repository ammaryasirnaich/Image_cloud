import utils
import open3d
import os
import shutil
import pykitti
import time


basedir = '/media/ammar/eecs_ammar/Kitti'

# Specify the dataset to load
date = '2011_09_26'
drive = '0009'
dataset = utils.load_dataset(basedir, date, drive, True)
save_result_directory = date + "_" + drive + "_stereo_cloud"
path = os.path.join(os.getcwd(), save_result_directory)



pcd = open3d.geometry.PointCloud()
vis = open3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

render_option = vis.get_render_option()
render_option.point_size = 0.01

data = pykitti.raw(basedir, date, drive)
to_reset_view_point = True

for points_with_intensity in data.velo:
    points = points_with_intensity[:, :3]
    pcd.points = open3d.utility.Vector3dVector(points)
    status = vis.update_geometry(pcd)

    print(status)
    if to_reset_view_point:
        vis.reset_view_point(True)
        to_reset_view_point = False
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.2)

vis.destroy_window()

