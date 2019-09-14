import cv2
import os

video_set_dir = 'videos/signal_fire_productions/split'
dest_dir = 'videos/signal_fire_productions/frames'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

video_list = os.listdir(video_set_dir)
for video in video_list:
    name, _ = os.path.splitext(video)
    frame_dest_dir = os.path.join(dest_dir, name)
    if not os.path.exists(frame_dest_dir):
        os.makedirs(frame_dest_dir)
    vid_src = os.path.join(video_set_dir, video)
    print(vid_src)
    vid_ptr = cv2.VideoCapture(vid_src)
    count = 0
    limit = 1000

    success, frame = vid_ptr.read()
    while success & (count<limit):
        if count%3 == 0:
            frame_name = '%05d.png' % count
            # print(success, frame_name)
            out_file = os.path.join(frame_dest_dir, frame_name)
            print(out_file)
            cv2.imwrite(out_file, frame)
        count += 1
        success, frame = vid_ptr.read()
