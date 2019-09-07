import cv2
import os
import sys

def resize_frames(frame_dir, dest_dir, new_shape=None, dest_ext='.png', sf=None):

    frame_list = os.listdir(frame_dir)
    # ref_list = os.listdir('test_data')

    for frame_image in frame_list:
        name, _ = os.path.splitext(frame_image)
        frame_path = os.path.join(frame_dir, frame_image)
        dest_path = os.path.join(dest_dir, name + dest_ext)

        fimg = cv2.imread(frame_path)
        if sf is not None:
            old_shape = fimg.shape[:2]
            new_shape = (int(old_shape[0]/2), int(old_shape[1]/2))
        dimg = cv2.resize(fimg, new_shape)
        cv2.imwrite(dest_path, dimg)


if __name__ == '__main__':
    frame_dir = sys.argv[1] if len(sys.argv) > 1 else 'set14'
    dest_dir = sys.argv[2] if len(sys.argv) > 2 else 'set14_resized'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    sf = [2.0, 2.0]
    resize_frames(frame_dir, dest_dir, sf=sf)
