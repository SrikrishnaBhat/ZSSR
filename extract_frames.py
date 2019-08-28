import cv2
import os

vid_id = '100445787'
vid_src = '{}.mp4'.format(vid_id)
print(vid_src)

if not os.path.exists(vid_id):
    os.makedirs(vid_id)

vid_ptr = cv2.VideoCapture(vid_src)
count = 0

success, frame = vid_ptr.read()
while success:
    if count%5 == 0:
        frame_name = '%05d.png' % count
        print(success, frame_name)
        cv2.imwrite(os.path.join(vid_id, frame_name), frame)
    count += 1
    success, frame = vid_ptr.read()

