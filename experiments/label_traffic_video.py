from __future__ import unicode_literals

import os, sys
sys.path.insert(0,"/home/autolab/Workspaces/jim_working/env/lib/python2.7/site-packages/")

from tcp.object_detection.video_labeler import VideoLabeler
from tcp.configs.alberta_config import Config
import cPickle as pickle
import glob

cnfg = Config()
vl = VideoLabeler(cnfg)

###GET VIDEOS
VIDEO_FILE = '%s/*.mp4' % cnfg.video_root_dir
videos = glob.glob(VIDEO_FILE)

###LABEL VIDEOS
for video_path in sorted(videos):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    datestamp = video_name.split('_')[-2]
    timestamp = video_name.split('_')[-1]

    year, month, date = [int(i) for i in datestamp.split('-')]
    hour, minute, second = [int(i) for i in timestamp.split('-')]

    # Setting first video
    tmp_time = int('%02d%02d%02d' % (date, hour, minute))
    if tmp_time < 281000:
        continue
    # Setting last video
    if tmp_time > 281700:
        break

    # Process video
    print '\nRunning video labeler on %s' % (video_name)
    vl.load_video(video_path)
    all_rclasses, all_rbboxes = vl.generate_bounding_boxes(debug_pickle=True)
    all_rclasses, all_rbboxes = vl.run_init_labeler(debug_pickle=True, no_gui=False)
    camera_view_trajectory = vl.generate_trajectories()

    with open('{0}/{1}/{1}_trajectories.cpkl'.format(cnfg.save_debug_pickles_path, video_name),'wb+') as trajectory_file:
        pickle.dump(camera_view_trajectory, trajectory_file)

    vl.close_video()
    raw_input('\nPress enter to continue...\n')

print 'End of labeling'
