import cv2
import os

framerate = 3
src_path = 'gt_data'
dest_file = 'gt_data.mp4'
codec = 'MJPG'

dims = (448, 448)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(dest_file, fourcc, framerate, dims)
frames_list = os.listdir(src_path)

for frame_name in frames_list:
    frame = cv2.imread(os.path.join(src_path, frame_name))
    out.write(frame)

out.release()