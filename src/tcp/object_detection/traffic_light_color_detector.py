import cv2

import numpy as np

from scipy.stats import zscore

class TrafficLightColorDetector:
    def __init__(self, config):
        self.config = config

    def get_car_light_colors(self, img, debug=False):
        retval = []
        for i, bbox in enumerate(self.config.traffic_light_bboxes):
            light_roi = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            light_roi_height, light_roi_width, _ = light_roi.shape
            light_roi_hls = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HLS)
            _, light_roi_thresholded = cv2.threshold(light_roi_hls[:,:,1],
                                                    self.config.traffic_light_threshold,
                                                    255,
                                                    cv2.THRESH_TOZERO)

            regional_brightness = []

            if light_roi_height > light_roi_width:
                regional_brightness.append(np.mean(light_roi_thresholded[0 : light_roi_height // 3, :]))
                regional_brightness.append(np.mean(light_roi_thresholded[light_roi_height // 3 : light_roi_height // 3 * 2, :]))
                regional_brightness.append(np.mean(light_roi_thresholded[light_roi_height // 3 * 2:, :]))
            else:
                regional_brightness.append(np.mean(light_roi_thresholded[:, 0 : light_roi_height // 3]))
                regional_brightness.append(np.mean(light_roi_thresholded[:, light_roi_height // 3 : light_roi_height // 3 * 2]))
                regional_brightness.append(np.mean(light_roi_thresholded[:, light_roi_height // 3 * 2:]))

            if debug:
                print 'Regional brightness for bbox %2d: ' % i, regional_brightness
            
            # No significantly brighter light found
            if np.sum(regional_brightness) == 0.0:
                retval.append(None)
                continue
            
            brightest = np.argmax(regional_brightness)
            
            if brightest == 0:
                retval.append('red')
            elif brightest == 1:
                retval.append('yellow')
            elif brightest == 2:
                retval.append('green')
            else:
                retval.append(None)
        return retval

    def get_pedestrian_light_colors(self, img, debug=False):
        retval = []
        for i, bbox in enumerate(self.config.pedestrian_light_bboxes):
            light_roi = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            light_roi_hls = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HLS)
            
            _, light_roi_thresholded = cv2.threshold(light_roi_hls[:,:,1], self.config.pedestrian_light_thresholds[0], 0, cv2.THRESH_TOZERO)
            _, light_roi_thresholded = cv2.threshold(light_roi_thresholded, self.config.pedestrian_light_thresholds[1], 0, cv2.THRESH_TOZERO_INV)
            
            light_roi_hls[:,:,1] = light_roi_thresholded
            light_roi_bgr = cv2.cvtColor(light_roi_hls, cv2.COLOR_HLS2BGR)

            light_roi_bgr_mean = cv2.mean(light_roi_bgr)[:3]
            light_roi_bgr_zscore = zscore(light_roi_bgr_mean)

            if debug:
                print 'z-scores of BGR channels for bbox %2d: ' % i, light_roi_bgr_zscore

            if light_roi_bgr_zscore[2] > self.config.pedestrian_light_zscore:
                retval.append('red')
            else:
                retval.append('white')
        return retval
