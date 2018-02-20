from __future__ import unicode_literals

import sys, os
import cv2

import numpy as np

#from AbstractDetector import AbstractDetector
from tcp.object_detection.video_labeler import LabelVideo
from tcp.registration.homography import Homography
from tcp.registration.obs_filtering import ObsFiltering
from tcp.registration.viz_regristration import VizRegristration
from tcp.object_detection.ssd_detector import SSD_VGG16Detector
from tcp.object_detection.hard_code_labels import HardCodeLabel
from tcp.configs.alberta_config import Config
import IPython
import glob
import cPickle as pickle

cnfg = Config()
vr = VizRegristration(cnfg)
hm = Homography(cnfg)
hcl = HardCodeLabel(cnfg)
of = ObsFiltering(cnfg)

camera_view_trajectory = pickle.load(open('test.p','r'))


camera_view_trajectory = hcl.label_video(camera_view_trajectory)
simulator_view_trajectory = hm.transform_trajectory(camera_view_trajectory)
filtered_trajectory = of.heuristic_label(simulator_view_trajectory)
	


vr.visualize_trajectory_dots(filtered_trajectory,plot_traffic_images=True)


IPython.embed()
