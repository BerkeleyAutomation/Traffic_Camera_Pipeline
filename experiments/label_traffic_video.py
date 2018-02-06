from __future__ import unicode_literals

import sys, os
import cv2
import youtube_dl
import numpy as np
from urlparse import parse_qs

#from AbstractDetector import AbstractDetector
from tcp.object_detection.video_labeler import LabelVideo
from tcp.registration.homography import Homography
from tcp.object_detection.ssd_detector import SSD_VGG16Detector
from tcp.configs.alberta_config import Config
import IPython
import glob

VIDEO_FILE = 'train_video/*.mp4'

cnfg = Config()
vl = LabelVideo(cnfg)
hm = Homography()


###GET VIDEOS
videos = glob.glob(VIDEO_FILE)


###LABEL VIDEOS
camera_view_trajectories = []
for video in videos:
	camera_view_trajectory = vl.label_video(video)
	camera_view_trajectories.append(camera_view_trajectory)


####GENERATE HOMOGRAPHY
simulator_view_trajectories = []
for video in videos:
	simulator_view_trajectory = hm.Homography(video)
	simulator_view_trajectories.append(simulator_view_trajectory)


IPython.embed()

