import cv2

import numpy as np

from tcp.registration.homography import Homography, test_homography_on_img
from tcp.configs.alberta_config import Config

cnfg = Config()
hm = Homography(cnfg)

img = cv2.imread(cnfg.pedestrian_crop_image_path)
img_warped = test_homography_on_img(hm, img)
