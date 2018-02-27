import os
import numpy as np

from gym_urbandriving.assets import Terrain, Lane, Street, Sidewalk,\
    Pedestrian, Car, TrafficLight
#
# path and dataset parameter
#


class Config(object):
###############PARAMETERS TO SWEEP##########


    def __init__(self):
        ###STREAMING####
        self.STREAM_OUTPUT_SEGMENT_TIME_LIMIT = 60  # approximate length of video segments in seconds
        self.STREAM_OUTPUT_DIR_SIZE_LIMIT = 1e10    # maximum size limit for downloaded video in bytes

        self.check_point_path = 'Checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        self.video_root_dir = 'Train_Videos'
        self.save_debug_img_path = 'Debug_Imgs/'
        self.save_debug_pickles_path = 'Debug_Pickles'
        self.crop_image_path = 'alberta_crop_image.png'

        ####REGISTRATION####
        self.street_corners = np.array([[765, 385],
                            [483, 470],
                            [1135, 565],
                            [1195, 425]])

        self.simulator_corners = np.array([[400, 400],
                               [400, 600],
                               [600, 600],
                               [600, 400]])

        self.alberta_img_dim = [1280, 720]

        self.traffic_light_threshold = 100  # any pixel with lightness in HLS color space below threadhold will be set to 0
        self.traffic_light_bboxes = [(665, 244, 678, 280),  # (xmin, ymin, xmax, ymax) of the bounding box
                                     (690, 245, 702, 278)]

        self.pedestrian_light_thresholds = (30, 160)  # any pixel with lightness in HLS color space outside the threadhold will be set to 0
        self.pedestrian_light_zscore = 1.0  # zscore for red channel in RGB distribution above which the light would be detected as red
        self.pedestrian_light_bboxes = [(1220, 306, 1236, 318)]  # (xmin, ymin, xmax, ymax) of the bounding box

        self.sim_scale = [1.04,1.10]

        self.use_pedestrian = False
        self.save_images = True

        ####FILTERING#######
        self.time_limit = 100

        self.vz_time_horizon = 500




        #######LANES######
        #######NSEW########
        self.lanes = [Lane(450, 200, 400, 100, angle=-(np.pi/2)),
                    Lane(550, 800, 400, 100, angle=(np.pi/2)),
                    Lane(800, 450, 400, 100, angle=-np.pi),
                    Lane(200, 550, 400, 100)]

