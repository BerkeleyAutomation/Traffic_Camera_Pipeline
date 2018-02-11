import os
import numpy as np
#
# path and dataset parameter
#


class Config(object):
###############PARAMETERS TO SWEEP##########


    def __init__(self):
        ###STREAMING####
        self.STREAM_OUTPUT_SEGMENT_TIME_LIMIT = 60  # approximate length of video segments in seconds
        self.STREAM_OUTPUT_DIR_SIZE_LIMIT = 1e9     # maximum size limit for downloaded video in bytes
        
        self.check_point_path = 'Checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        self.video_root_dir = 'Train_Videos'
        self.save_debug_img_path = 'Debug_Imgs/'

        ####REGISTRATION####
        self.street_corners = np.array([[765, 385],
                            [483, 470],
                            [1135, 565],
                            [1195, 425]])

        self.simulator_corners = np.array([[400, 400],
                               [400, 600],
                               [600, 600],
                               [600, 400]])

        self.alberta_img_dim = [1280,720]

        self.sim_scale = [1.04,1.10]

        self.use_pedestrian = False
        self.save_images = True


