import cv2
import numpy as np
import os

src_dir1 = 'gt_Saat_Ernte'
src_dir2 = 'results_Saat_Ernte_sf4_n12'

dest_dir = src_dir1 + '_mixed_b1'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

src_list1 = os.listdir(src_dir2)
src_list2 = os.listdir(src_dir2)

src_list1.sort()
src_list2.sort()

for i in range(len(src_list1)):
    print(i)
    file1 = os.path.join(src_dir1, src_list1[i])
    file2 = os.path.join(src_dir2, src_list2[i])

    im1 = cv2.imread(file1)
    im2 = cv2.imread(file2)

    res_img = np.append(im1, im2, axis=1)
    cv2.imwrite(os.path.join(dest_dir, '%05d.png' % i), res_img)
