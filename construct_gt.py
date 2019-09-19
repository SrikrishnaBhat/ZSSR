import cv2
import os
import sys
import numpy as np
from imresize import imresize

def resize_frames(src_dir1, src_dir2, dest_dir):

    src_list1 = os.listdir(src_dir1)
    src_list1.sort()
    src_list2 = os.listdir(src_dir2)
    src_list2.sort()

    for src_file1, src_file2 in zip(src_list1, src_list2):
        print(src_file1, src_file2)
        src_img1 = cv2.imread(os.path.join(src_dir1, src_file1))
        src_img2 = cv2.imread(os.path.join(src_dir2, src_file2))

        dest_img = imresize(src_img1, output_shape=src_img2.shape)
        cv2.imwrite(os.path.join(dest_dir, src_file2), dest_img)


if __name__ == '__main__':
    src_dir1 = sys.argv[1] if len(sys.argv) > 1 else 'BSDS300/images/test'
    src_dir2 = sys.argv[2] if len(sys.argv) > 2 else 'BSD100_resize'
    dest_dir = sys.argv[3] if len(sys.argv) > 3 else 'BSD100_gt'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    resize_frames(src_dir1, src_dir2, dest_dir)
