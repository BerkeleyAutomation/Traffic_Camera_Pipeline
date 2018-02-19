import cv2

from tcp.configs.alberta_config import Config
from tcp.object_detection.traffic_light_color_detector import TrafficLightColorDetector

# string to BGR
COLOR_DICT = {'red':    (0, 0, 255),
             'yellow':  (0, 255, 255),
             'green':   (0, 255, 0),
             'white':   (255, 255, 0),
                None:   (255, 255, 255)}

def draw_light_bbox_on_img(img, bboxes, colors):
    assert len(bboxes) == len(colors)
    for i, bbox in enumerate(bboxes):
        color = COLOR_DICT[colors[i]]
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
        cv2.putText(img, colors[i], (bbox[0] - 3, bbox[1] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1, color, 2, cv2.LINE_AA)
    return img

def cv2_display_img(img):
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', window_width, window_height)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    img_paths = ['/nfs/diskstation/jren/alberta_cam_manual_bboxes/day/Images/alberta_cam_original_2017-10-26_15-28-14_0.JPG',
                 '/nfs/diskstation/jren/alberta_cam_manual_bboxes/day/Images/alberta_cam_original_2017-10-26_15-28-14_2.JPG',
                 '/nfs/diskstation/jren/alberta_cam_manual_bboxes/night/Images/alberta_cam_original_2017-10-27_22-02-33_7.JPG',
                 '/nfs/diskstation/jren/alberta_cam_manual_bboxes/night/Images/alberta_cam_original_2017-10-27_22-02-33_4.JPG']

    imgs = [cv2.imread(img_path) for img_path in img_paths]

    cnfg = Config()
    traffic_light_color_detector = TrafficLightColorDetector(cnfg)

    car_light_results = [traffic_light_color_detector.get_car_light_colors(img, debug=True) for img in imgs]
    ped_light_results = [traffic_light_color_detector.get_pedestrian_light_colors(img, debug=True) for img in imgs]

    imgs = [draw_light_bbox_on_img(img, cnfg.traffic_light_bboxes, car_light_results[i]) for i, img in enumerate(imgs)]
    imgs = [draw_light_bbox_on_img(img, cnfg.pedestrian_light_bboxes, ped_light_results[i]) for i, img in enumerate(imgs)]

    for img in imgs:
        cv2_display_img(img)

if __name__== "__main__":
    main()