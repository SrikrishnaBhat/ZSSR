import cv2
import os
import sys
import numpy as np
from imresize import imresize

def add_noise(sf, img):
    lengths = [np.random.randint(int(sf[0]**2)+1), np.random.randint(int(sf[1]**2)+1)]
    if lengths[0] % 2 == 0:
        lengths[0] += 1
    if lengths[1] % 2 == 0:
        lengths[1] += 1
    lengths = tuple(lengths)
    print(lengths)
    theta = np.random.rand(1) * np.pi
    length_diag = [[lengths[0], 0], [0, lengths[1]]]
    U = np.array([[np.cos(theta).item(), -np.sin(theta).item()], [np.sin(theta).item(), np.cos(theta).item()]])
    covar = U * length_diag * U.transpose()
    x_var, y_var = np.sqrt(covar[0, 0]), np.sqrt(covar[1, 1])
    return cv2.GaussianBlur(img, lengths, sigmaX=x_var, sigmaY=y_var)

def resize_frames(frame_dir, dest_dir, new_shape=None, dest_ext='.png', sf=None, noise=False):

    frame_list = os.listdir(frame_dir)
    # ref_list = os.listdir('test_data')
    # frame_list.sort()

    for index, frame_image in enumerate(frame_list):
        #if index > 30:
        #    break
        name, _ = os.path.splitext(frame_image)
        frame_path = os.path.join(frame_dir, frame_image)
        dest_path = os.path.join(dest_dir, name + dest_ext)
        print(dest_path)

        fimg = cv2.imread(frame_path)
        old_shape = fimg.shape[:2]
        if sf is not None:
            new_shape = (int(old_shape[1]/sf[1]), int(old_shape[0]/sf[0]))
        # elif new_shape is None:
        else:
            new_shape = [old_shape[1], old_shape[0]]
            if old_shape[0]%10 == 1:
                new_shape[1] -= 1
            if old_shape[1]%10 == 1:
                new_shape[0] -= 1
            new_shape = tuple(new_shape)
        dimg = imresize(fimg, scale_factor=sf)
        if noise:
            noisy_dimg = add_noise(sf, dimg)
            cv2.imwrite(dest_path, noisy_dimg)
        else:
            cv2.imwrite(dest_path, dimg)


if __name__ == '__main__':
    frame_dir = sys.argv[1] if len(sys.argv) > 1 else 'BSDS300/images/test'
    dest_dir = sys.argv[2] if len(sys.argv) > 2 else 'BSD100_assaf'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    sf = [0.5, 0.5]
    resize_frames(frame_dir, dest_dir, sf=sf)
