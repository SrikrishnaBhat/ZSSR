import cv2
import os
import sys

def resize_frames(frame_dir, dest_dir, new_shape=(224, 224)):

    frame_list = os.listdir(frame_dir)

    for frame_image in frame_list:
        frame_path = os.path.join(frame_dir, frame_image)
        dest_path = os.path.join(dest_dir, frame_image)

        fimg = cv2.imread(frame_path)
        dimg = cv2.resize(fimg, new_shape)
        cv2.imwrite(dest_path, dimg)


if __name__ == '__main__':
    frame_dir = sys.argv[1] if len(sys.argv) > 1 else 'test_data/0a7b27fde9'
    dest_dir = sys.argv[2] if len(sys.argv) > 2 else 'test_data'
    new_shape = (224, 224)

    resize_frames(frame_dir, dest_dir, new_shape)