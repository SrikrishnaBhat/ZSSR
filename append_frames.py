import cv2
import numpy as np
import os

video_type = 'nissanmurano'

src_dir1 = 'videos/{}/frames_gt'.format(video_type, video_type)
src_dir2 = 'results_{}_sf2'.format(video_type)

dest_dir = os.path.join('videos', video_type + '_' + 'gt_multi')
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

scene_list = os.listdir(src_dir1)
scene_list.sort()
overall_dest_path = os.path.join(dest_dir, 'combined')
if not os.path.exists(overall_dest_path):
    os.makedirs(overall_dest_path)
j = 0
for scene in scene_list:
    scene_path1 = os.path.join(src_dir1, scene)
    scene_path2 = os.path.join(src_dir2, scene, 'pred_data')
    src_list1 = os.listdir(scene_path1)
    src_list2 = os.listdir(scene_path2)

    src_list1.sort()
    src_list2.sort()
    dest_scene_path = os.path.join(dest_dir, scene)
    if not os.path.exists(dest_scene_path):
        os.makedirs(dest_scene_path)

    for i in range(len(src_list1)):
        file1 = os.path.join(scene_path1, src_list1[i])
        file2 = os.path.join(scene_path2, src_list2[i])

        print(file1, file2)

        im1 = cv2.imread(file1)
        im2 = cv2.imread(file2)
        im_shape = im2.shape
        im1 = cv2.resize(im1, (im_shape[1], im_shape[0]))

        res_img = np.append(im1, im2, axis=1)
        cv2.imwrite(os.path.join(dest_scene_path, '%05d.png' % i), res_img)
        cv2.imwrite(os.path.join(overall_dest_path, '%05d.png' % j), res_img)
        j += 1
