import os
import cv2
import pdb

import cPickle as pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

"""
AX is a matplotlib axis object.
BBOX is an array like object: [xmin, ymin, xmax, ymax].
IM_SIZE is image (width, height).
"""
# def draw_bbox(ax, bbox, label, im_size, color=None, picker=True, traj_label=None):
#     width = (bbox[2] - bbox[0]) * im_size[0]
#     height = (bbox[3] - bbox[1]) * im_size[1]
    
#     if color is None:
#         color = 'magenta'
#         linestyle = 'dashed'
#     else:
#         linestyle = 'solid'
#     # patches.Rectangle((xmin, ymin), width, height)
#     rect = patches.Rectangle((bbox[0] * im_size[0], bbox[1] * im_size[1]), width, height,
#                              linewidth=1.5, linestyle=linestyle, label=label,
#                              edgecolor=color, facecolor='none', picker=picker)
#     ax.add_patch(rect)
#     if traj_label is not None:
#         ax.text(bbox[0] * im_size[0] + 5, bbox[1] * im_size[1] - 12, traj_label,
#             fontsize=10, bbox=dict(facecolor=color, alpha=0.5))

def cv2_draw_line(img, pt1, pt2, color, thickness=1, style=None, gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1

def cv2_draw_poly(img, pts, color, thickness=1, style=None):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        cv2_draw_line(img, s, e, color, thickness, style)

def cv2_draw_rect(img, pt1, pt2, color, thickness=1, style=None):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    if style == 'dotted':
        cv2_draw_poly(img, pts, color, thickness, style)
    else:
        # cv2.Rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
        cv2.rectangle(img, pt1, pt2, color, thickness)

def cv2_draw_bbox(img, bbox, label, im_size, color=None, traj_label=None):
    scaled_bbox = [int(coord * (im_size[0] if i % 2 == 0 else im_size[1])) for i, coord in enumerate(bbox)]
    width = scaled_bbox[2] - scaled_bbox[0]
    height = scaled_bbox[3] - scaled_bbox[1]
    
    if color is None:
        color = (255, 0, 255)
        linestyle = 'dotted'
    else:
        linestyle = 'solid'
    rect = cv2_draw_rect(img, (scaled_bbox[0], scaled_bbox[1]), (scaled_bbox[2], scaled_bbox[3]), color, 2, linestyle)
    if traj_label is not None:
        # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        cv2.putText(img, str(traj_label), (scaled_bbox[0] + 5, scaled_bbox[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 2, color)
    
def YX_to_XY(bbox):
    return (bbox[1], bbox[0], bbox[3], bbox[2])

def get_midpoint(xmin, ymin, xmax, ymax):
    mid_x = (xmin + (xmax - xmin) / 2)
    mid_y = (ymin + (ymax - ymin) / 2)
    return mid_x, mid_y

def get_color(trajectory_label):
    color = np.uint8([[[30 * int(trajectory_label) % 255, 255, 255]]])
    color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR).flatten().tolist()
    return tuple(color)


class InitLabeler_OpenCV():
    # Warning: Caching frames can be very memory intensive for large videos
    def __init__(self, config, cap, all_rbboxes, all_rclasses, init_labeler_pickle_path=None, cache_frames=False):
        assert len(all_rbboxes) == len(all_rclasses), \
            'Number of frames in list of bboxes (%d) and list of classes (%d) mismatch' % \
            (len(all_rbboxes), len(all_rclasses))

        self.config = config
        self.cap = cap
        self.all_rbboxes = all_rbboxes
        self.all_rclasses = all_rclasses
        self.init_labeler_pickle_path = init_labeler_pickle_path

        self.img_size = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                         self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.trajectories = []
        self.trajectory_label = 0
        self.frame_i = 0
        self.quit_loop = False
        self.pause = False
        self.num_frames = len(self.all_rbboxes)

        self.trajectories = self.load_trajectories()

        if cache_frames:
            cached_frames = []
            print 'Caching frames into memory...'
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if frame is None:
                    break
                cached_frames.append(frame)

        try:
            while not self.quit_loop:
                if cache_frames:
                    self.frame = cached_frames[self.frame_i]
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_i)
                    _, self.frame = self.cap.read()
                    if self.frame is None:
                        break

                bboxes = self.all_rbboxes[self.frame_i]
                classes = self.all_rclasses[self.frame_i]
                midpoints = []
                
                for i, bbox in enumerate(bboxes):
                    midpoint = get_midpoint(*bbox)
                    color = None
                    traj_label = None
                    for tmp_x, tmp_y, tmp_class, tmp_traj in self.trajectories[self.frame_i]:
                        if np.allclose(midpoint, (tmp_x, tmp_y)):
                            color = get_color(tmp_traj)
                            traj_label = tmp_traj
                    cv2_draw_bbox(self.frame, bbox, classes[i], self.img_size, color=color, traj_label=traj_label)

                msg = 'Trajectory #%d Frame: %d/%d' % (self.trajectory_label, self.frame_i, self.num_frames - 1)
                cv2.putText(self.frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200))
                cv2.imshow("InitLabeler", self.frame)
                cv2.setMouseCallback("InitLabeler", self.cv2_on_click)
                
                if self.pause:
                    key = cv2.waitKeyEx(0)
                else:
                    key = cv2.waitKeyEx(33)
                self.cv2_on_key_press(key)

                self.frame_i += 1
                self.frame_i = min(self.num_frames - 1, self.frame_i)
                if self.quit_loop:
                    break
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()
        # print self.trajectories
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # def __init__(self, config, cap, all_rbboxes, all_rclasses, init_labeler_pickle_path=None):
    #     assert len(all_rbboxes) == len(all_rclasses), \
    #         'Number of frames in list of bboxes (%d) and list of classes (%d) mismatch' % \
    #         (len(all_rbboxes), len(all_rclasses))

    #     self.config = config
    #     self.cap = cap
    #     self.all_rbboxes = all_rbboxes
    #     self.all_rclasses = all_rclasses
    #     self.init_labeler_pickle_path = init_labeler_pickle_path

    #     self.img_size = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
    #                      self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #     self.trajectories = []
    #     self.trajectory_label = 0
    #     self.frame_i = 0
    #     self.quit_loop = False
    #     self.pause = False
    #     self.num_frames = len(self.all_rbboxes)
       
    #     plt.ioff()

    #     fig, ax = plt.subplots(1, figsize=(10, 6))
    #     ax.set_aspect('auto')
    #     fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    #     fig.canvas.mpl_connect('pick_event', self.on_pick)
    #     fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    #     plt.tight_layout()
    #     fig.show()

    #     self.trajectories = self.load_trajectories()

    #     while not self.quit_loop:
    #         self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_i)
    #         _, frame = self.cap.read()
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #         bboxes = self.all_rbboxes[self.frame_i]
    #         classes = self.all_rclasses[self.frame_i]
    #         midpoints = []
            
    #         ax.imshow(frame)
    #         plt.draw()
    #         ax.set_title('Trajectory #%d Frame: %d/%d' % (self.trajectory_label, self.frame_i, self.num_frames - 1))
            
    #         for i, bbox in enumerate(bboxes):
    #             midpoint = get_midpoint(*bbox)
    #             color = None
    #             traj_label = None
    #             for tmp_x, tmp_y, tmp_class, tmp_traj in self.trajectories[self.frame_i]:
    #                 if np.allclose(midpoint, (tmp_x, tmp_y)):
    #                     color = 'C%d' % (int(tmp_traj) % 10)
    #                     traj_label = tmp_traj
    #             draw_bbox(ax, bbox, classes[i], self.img_size, color=color, picker=(color is None), traj_label=traj_label)
            
    #         if self.pause:
    #             fig.waitforbuttonpress(timeout=-1)
    #         else:
    #             fig.waitforbuttonpress(timeout=0.001)
    #         ax.clear()
    #         plt.draw()
    #         self.frame_i += 1
    #         self.frame_i = min(self.num_frames - 1, self.frame_i)
    #         if self.quit_loop:
    #             break
    #     ax.clear()
    #     plt.close(fig)
    #     # print self.trajectories
    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    """
    Trajectories is a list of size LEN(IMG_NUMS), or the number of frames present in DATA_DIR.
    Each element of TRAJECTORIES is a list of tuples (x, y, detection_class, trajectory_label).
    Each tuple from above represents one bounding box in the frame.
    """
    def load_trajectories(self):
        assert self.num_frames is not None
        self.trajectories = np.empty((self.num_frames, 0)).tolist()
        if self.init_labeler_pickle_path is None:
            print 'Unable to load: init_labeler_pickle_path not provided.'
        try:
            self.trajectories = pickle.load(open(self.init_labeler_pickle_path, 'r'))
            print '"%s" loaded.' % self.init_labeler_pickle_path
        except IOError as e:
            print 'No "%s" found. Loading empty trajectory.' % self.init_labeler_pickle_path
        return self.trajectories

    def cv2_on_click(self, event, x, y, flags, param):
        # bbox: (xmin, ymin, xmax, ymax)
        def in_bbox(bbox, x, y):
            scaled_bbox = [int(coord * (self.img_size[0] if i % 2 == 0 else self.img_size[1])) for i, coord in enumerate(bbox)]
            return x >= scaled_bbox[0] and x <= scaled_bbox[2] and y >= scaled_bbox[1] and y <= scaled_bbox[3]

        if event == cv2.EVENT_LBUTTONDOWN:
            bboxes = self.all_rbboxes[self.frame_i]
            for i, bbox in enumerate(bboxes):
                if in_bbox(bbox, x, y):
                    print 'in bbox'
                    mid_x, mid_y = get_midpoint(*bbox)
                    self.trajectories[self.frame_i].append((mid_x, mid_y, self.all_rclasses[self.frame_i][i], self.trajectory_label))
                    color = get_color(self.trajectory_label)
                    cv2_draw_bbox(self.frame, bbox, self.all_rclasses[self.frame_i][i], self.img_size, color=color, traj_label=self.trajectory_label)
                    cv2.imshow("InitLabeler", self.frame)
                    self.trajectory_label += 1

    def cv2_on_key_press(self, key):
        if key == ord('q'):
            self.quit_loop = True
            print 'Quitting'
        elif key == ord('a') or key == 65364 or key == 2621440: # down
            self.trajectory_label -= 1
            self.trajectory_label = max(0, self.trajectory_label)
        elif key == ord('s') or key == 65362 or key == 2490368: # up
            self.trajectory_label += 1
        elif key == ord('z') or key == 65361 or key == 2424832: # left
            self.frame_i -= 1
            self.frame_i = max(0, self.frame_i)
        elif key == ord('x') or key == 65363 or key == 2555904: # right
            self.frame_i += 1
            self.frame_i = min(self.num_frames - 1, self.frame_i)
        elif key == ord('.'):
            self.frame_i += 10
            self.frame_i = min(self.num_frames - 1, self.frame_i)
        elif key == ord(','):
            self.frame_i -= 10
            self.frame_i = max(0, self.frame_i)
        elif key == ord(']'):
            self.frame_i += 100
            self.frame_i = min(self.num_frames - 1, self.frame_i)
        elif key == ord('['):
            self.frame_i -= 100
            self.frame_i = max(0, self.frame_i)
        elif key == ord('w'):
            if self.init_labeler_pickle_path is None:
                print 'Unable to save: init_labeler_pickle_path not provided.'
            else:
                init_labeler_pickle_dir = os.path.dirname(self.init_labeler_pickle_path)
                if not os.path.exists(init_labeler_pickle_dir):
                    os.makedirs(init_labeler_pickle_dir)
                pickle.dump(self.trajectories, open(self.init_labeler_pickle_path, 'w+'))
                print 'Written trajectories to %s.' % self.init_labeler_pickle_path
        elif key == ord('p'):
            print self.trajectories[self.frame_i]
        elif key == ord(' '):
            self.pause = not self.pause

        if key != -1:
            self.frame_i -= 1
            # print 'key pressed: ', key


    # def on_pick(self, event):
    #     this = event.artist
    #     bbox = this.get_bbox()
    #     mid_x, mid_y = get_midpoint(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    #     mid_x /= self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #     mid_y /= self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #     self.trajectories[self.frame_i].append((mid_x, mid_y, str(this._label), self.trajectory_label))
    #     self.trajectory_label += 1
        
    # def on_key_press(self, event):
    #     if event.key == 'q':
    #         self.quit_loop = True
    #     elif event.key == 'down':
    #         self.trajectory_label -= 1
    #         self.trajectory_label = max(0, self.trajectory_label)
    #     elif event.key == 'up':
    #         self.trajectory_label += 1
    #     elif event.key == 'left':
    #         self.frame_i -= 1
    #         self.frame_i = max(0, self.frame_i)
    #     elif event.key == 'right':
    #         self.frame_i += 1
    #         self.frame_i = min(self.num_frames - 1, self.frame_i)
    #     elif event.key == '.':
    #         self.frame_i += 10
    #         self.frame_i = min(self.num_frames - 1, self.frame_i)
    #     elif event.key == ',':
    #         self.frame_i -= 10
    #         self.frame_i = max(0, self.frame_i)
    #     elif event.key == ']':
    #         self.frame_i += 100
    #         self.frame_i = min(self.num_frames - 1, self.frame_i)
    #     elif event.key == '[':
    #         self.frame_i -= 100
    #         self.frame_i = max(0, self.frame_i)
    #     elif event.key == 'w':
    #         if self.init_labeler_pickle_path is None:
    #             print 'Unable to save: init_labeler_pickle_path not provided.'
    #         else:
    #             init_labeler_pickle_dir = os.path.dirname(self.init_labeler_pickle_path)
    #             if not os.path.exists(init_labeler_pickle_dir):
    #                 os.makedirs(init_labeler_pickle_dir)
    #             pickle.dump(self.trajectories, open(self.init_labeler_pickle_path, 'w+'))
    #             print 'Written trajectories to %s.' % self.init_labeler_pickle_path
    #     elif event.key == 'p':
    #         print self.trajectories[self.frame_i]
    #     elif event.key == ' ':
    #         self.pause = not self.pause
            
    #     self.frame_i -= 1

    def has_init_label(self, frame_i):
        ''''
        if a new car is present the user should press some button 
        and then stop for some label

        Returns:
        ----------
        Bool, True if a label is present false otherwise 

        '''
        return len(self.trajectories[frame_i]) != 0

    def label_image(self, frame_i):

        '''
        Takes a frame index
        return the bounding boxes corresponding to new objects in the frame
        return format is [(centroid_x, centroid_y, detection_class, trajectory_label), ...]

        '''
        return self.trajectories[frame_i]

    def get_arg_init_label(self, frame_i):
        frame_i_bboxes = self.all_rbboxes[frame_i]
        frame_i_centroids = [get_midpoint(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in frame_i_bboxes]
        init_object_bboxes = self.trajectories[frame_i]
        retval = []

        for mid_x, mid_y, rclass, traj_label in init_object_bboxes:
            try:
                retval.append(frame_i_centroids.index((mid_x, mid_y)))
            except ValueError:
                continue
        return retval