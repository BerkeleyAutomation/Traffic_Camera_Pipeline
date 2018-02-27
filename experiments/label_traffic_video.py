from __future__ import unicode_literals

import os

#from AbstractDetector import AbstractDetector
from tcp.object_detection.video_labeler import LabelVideo
# from tcp.registration.homography import Homography
# from tcp.registration.iterative_filtering import IterativeFiltering
# from tcp.object_detection.ssd_detector import SSD_VGG16Detector
from tcp.configs.alberta_config import Config
import cPickle as pickle
import glob


cnfg = Config()
vl = LabelVideo(cnfg)
# hm = Homography(cnfg)
# iterative_filter = IterativeFiltering(cnfg)


###GET VIDEOS
VIDEO_FILE = 'Train_Videos/*.mp4'
videos = glob.glob(VIDEO_FILE)


###LABEL VIDEOS
for video_path in sorted(videos):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Setting starting point
    datestamp = video_name.split('_')[-2]
    timestamp = video_name.split('_')[-1]

    year, month, date = [int(i) for i in datestamp.split('-')]
    hour, minute, second = [int(i) for i in timestamp.split('-')]

    if date <= 25 or hour <= 15 or minute <= 46:
        continue

    camera_view_trajectory = vl.label_video(video_path, debug_pickle=True)

    with open('{0}/{1}/{1}_trajectories.cpkl'.format(self.config.save_debug_pickles_path, video_name),'wb+') as trajectory_file:
        pickle.dump(camera_view_trajectory, trajectory_file)

    raw_input('\nPress enter to continue...\n')
