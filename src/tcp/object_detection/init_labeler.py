import os
import pickle
import cv2

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
def draw_bbox(ax, bbox, label, im_size, color=None, picker=True, traj_label=None):
    width = (bbox[2] - bbox[0]) * im_size[0]
    height = (bbox[3] - bbox[1]) * im_size[1]
    
    if color is None:
        color = 'magenta'
        linestyle = 'dashed'
    else:
        linestyle = 'solid'
    # patches.Rectangle((xmin, ymin), width, height)
    rect = patches.Rectangle((bbox[0] * im_size[0], bbox[1] * im_size[1]), width, height,
                             linewidth=1.5, linestyle=linestyle, label=label,
                             edgecolor=color, facecolor='none', picker=picker)
    ax.add_patch(rect)
    if traj_label is not None:
        ax.text(bbox[0] * im_size[0] + 5, bbox[1] * im_size[1] - 12, traj_label,
            fontsize=10, bbox=dict(facecolor=color, alpha=0.5))
    
def YX_to_XY(bbox):
    return (bbox[1], bbox[0], bbox[3], bbox[2])

def get_midpoint(xmin, ymin, xmax, ymax):
    mid_x = (xmin + (xmax - xmin) / 2)
    mid_y = (ymin + (ymax - ymin) / 2)
    return mid_x, mid_y


class InitLabeler():

    def __init__(self, config, cap, all_rbboxes, all_rclasses):
        assert len(all_rbboxes) == len(all_rclasses), \
            'Number of frames in list of bboxes (%d) and list of classes (%d) mismatch' % \
            (len(all_rbboxes), len(all_rclasses))

        self.config = config
        self.cap = cap
        self.all_rbboxes = all_rbboxes
        self.all_rclasses = all_rclasses

        self.img_size = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                         self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.trajectories = []
        self.trajectory_label = 0
        self.frame_i = 0
        self.quit_loop = False
        self.num_frames = len(self.all_rbboxes)
       
        plt.ioff()

        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.set_aspect('auto')
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.mpl_connect('pick_event', self.on_pick)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.tight_layout()
        fig.show()

        self.trajectories = self.load_trajectories(self.config.init_labeler_pickle_path)

        while not self.quit_loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_i)
            _, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            bboxes = self.all_rbboxes[self.frame_i]
            classes = self.all_rclasses[self.frame_i]
            midpoints = []
            
            ax.imshow(frame)
            plt.draw()
            ax.set_title('Trajectory #%d Frame: %d/%d' % (self.trajectory_label, self.frame_i, self.num_frames - 1))
            
            for i, bbox in enumerate(bboxes):
                midpoint = get_midpoint(*bbox)
                color = None
                traj_label = None
                for tmp_x, tmp_y, tmp_class, tmp_traj in self.trajectories[self.frame_i]:
                    if np.allclose(midpoint, (tmp_x, tmp_y)):
                        color = 'C%d' % (int(tmp_traj) % 10)
                        traj_label = tmp_traj
                draw_bbox(ax, bbox, classes[i], self.img_size, color=color, picker=(color is None), traj_label=traj_label)
            
            fig.waitforbuttonpress(timeout=-1)
            ax.clear()
            plt.draw()
            self.frame_i += 1
            self.frame_i = min(self.num_frames - 1, self.frame_i)
            # if self.frame_i >= self.num_frames:
            #     pickle.dump(self.trajectories, open(self.config.init_labeler_pickle_path, 'w+'))
            #     print 'Wrote trajectory %d to %s' % (self.trajectory_label, self.config.init_labeler_pickle_path)
            #     self.trajectory_label += 1
            #     self.frame_i = 0
            if self.quit_loop:
                break
        ax.clear()
        plt.close(fig)
        # print self.trajectories
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    """
    Trajectories is a list of size LEN(IMG_NUMS), or the number of frames present in DATA_DIR.
    Each element of TRAJECTORIES is a list of tuples (x, y, detection_class, trajectory_label).
    Each tuple from above represents one bounding box in the frame.
    """
    def load_trajectories(self, trajectories_path):
        try:
            self.trajectories = pickle.load(open(self.config.init_labeler_pickle_path, 'r'))
            print '"%s" loaded.' % self.config.init_labeler_pickle_path
        except IOError:
            print 'No "%s" found. Loading empty trajectory.' % self.config.init_labeler_pickle_path
            assert self.num_frames is not None
            self.trajectories = np.empty((self.num_frames, 0)).tolist()
        return self.trajectories

    def on_pick(self, event):
        this = event.artist
        bbox = this.get_bbox()
        mid_x, mid_y = get_midpoint(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        mid_x /= self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        mid_y /= self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.trajectories[self.frame_i].append((mid_x, mid_y, str(this._label), self.trajectory_label))
        self.trajectory_label += 1
        
    def on_key_press(self, event):
        if event.key == 'q':
            self.quit_loop = True
        elif event.key == 'down':
            self.trajectory_label -= 1
            self.trajectory_label = max(0, self.trajectory_label)
        elif event.key == 'up':
            self.trajectory_label += 1
        elif event.key == 'left':
            self.frame_i -= 1
            self.frame_i = max(0, self.frame_i)
        elif event.key == 'right':
            self.frame_i += 1
            self.frame_i = min(self.num_frames - 1, self.frame_i)
        elif event.key == '.':
            self.frame_i += 10
            self.frame_i = min(self.num_frames - 1, self.frame_i)
        elif event.key == ',':
            self.frame_i -= 10
            self.frame_i = max(0, self.frame_i)
        elif event.key == ']':
            self.frame_i += 100
            self.frame_i = min(self.num_frames - 1, self.frame_i)
        elif event.key == '[':
            self.frame_i -= 100
            self.frame_i = max(0, self.frame_i)
        elif event.key == 'w':
            init_labeler_pickle_dir = os.path.dirname(self.config.init_labeler_pickle_path)
            if not os.path.exists(init_labeler_pickle_dir):
                os.makedirs(init_labeler_pickle_dir)
            pickle.dump(self.trajectories, open(self.config.init_labeler_pickle_path, 'w+'))
            print 'Written trajectories to %s.' % self.config.init_labeler_pickle_path
        elif event.key == 'p':
            print self.trajectories[self.frame_i]
            
        self.frame_i -= 1

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