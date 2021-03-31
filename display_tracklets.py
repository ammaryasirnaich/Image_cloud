
import numpy as np
import glob

import matplotlib.pyplot as plt
from utils import *
import utils
# import open3d
import os
import shutil
import pykitti
import time
import numpy as np

from mayavi import mlab

from utilityPackage.kitti_object import kitti_object, show_lidar_with_depth, show_lidar_on_image, \
                         show_image_with_boxes, show_lidar_topview_with_boxes




root_dir = "/media/ammar/eecs_ammar/Kitti/Multi_object_tracking"
dataset = kitti_object(root_dir, 'training')


frameNumber = 75
sence = 0
objects = dataset.get_frame_Lable_tracks(sence,frameNumber)

pc_velo = dataset.get_lidar(sence)
calib = dataset.get_calibration(sence)

img_left = dataset.get_image_Left(sence,frameNumber)
img_left_height, img_left_width, _ = img_left.shape




img_left_bbox2d, img_left_bbox3d = show_image_with_boxes(img_left, objects, calib)
img_left_bbox2d = cv2.cvtColor(img_left_bbox2d, cv2.COLOR_BGR2RGB)
#
# fig_bbox2d = plt.figure(figsize=(14, 7))
# ax_bbox2d = fig_bbox2d.subplots()
# ax_bbox2d.imshow(img_left_bbox2d)
# plt.show()


img_left_bbox3d = cv2.cvtColor(img_left_bbox3d, cv2.COLOR_BGR2RGB)
fig_bbox3d = plt.figure(figsize=(14, 7))
ax_bbox3d = fig_bbox3d.subplots()
ax_bbox3d.imshow(img_left_bbox3d)
plt.show()
