import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def mse(src: np.ndarray, test: np.ndarray):
    rows, cols = src.shape[:2]

    diff = (src-test) ** 2
    return diff.sum()/(rows*cols)

def psnr(src: np.ndarray, test: np.ndarray):
    max_val = np.max(test)
    mse_val = mse(src, test)
    return (20 * np.log10(max_val).item() - 10*np.log10(mse_val))

def ssim(src: np.ndarray, test: np.ndarray):
    mean_s = np.mean(src)
    mean_t = np.mean(test)
    var_s = np.var(src)
    var_y = np.var(test)
    covar_sy = np.cov(src.flatten(), test.flatten())[0][1]
    k1 = 0.01
    k2 = 0.03
    L = 255

    c1 = (k1*L)**2
    c2 = (k2*L)**2
    num = (2*mean_s*mean_t + c1) * (2*covar_sy + c2)
    denom = (mean_s**2 + mean_t**2 + c1)*(var_s + var_y + c2)
    return num/denom

gt_dir = 'videos/headnshoulders/frames_gt'
test_dir = 'videos/headnshoulders/results_imresize'

gt_scenes = os.listdir(gt_dir)
gt_scenes.sort()
test_scenes = os.listdir(test_dir)
test_scenes.sort()

psnr_list, ssim_list = [], []
for gt_scene, test_f_scene in zip(gt_scenes, test_scenes):
    print(gt_scene)
    gt_scene_path = os.path.join(gt_dir, gt_scene)
    gt_files_list = os.listdir(gt_scene_path)
    gt_files_list.sort()
    test_scene_path = os.path.join(test_dir, test_f_scene)
    test_files_list = os.listdir(test_scene_path)
    test_files_list.sort()
    for (gt, test_f) in zip(gt_files_list, test_files_list):
        try:
            src = cv2.imread(os.path.join(gt_scene_path, gt))
            test = cv2.imread(os.path.join(test_scene_path, test_f))
            test_shape = test.shape

            psnr_list.append(psnr(cv2.resize(src, (test_shape[1], test_shape[0])), test))
            ssim_list.append(ssim(cv2.resize(src, (test_shape[1], test_shape[0])), test))
        except Exception as ex:
            print(ex)
            print(test_f, gt)
            continue


mean_psnr = np.mean(np.asarray(psnr_list).flatten()).item()
mean_ssim = np.mean(np.asarray(ssim_list).flatten()).item()
print('Mean PSNR / Mean SSIM : {} / {}'.format(mean_psnr, mean_ssim))

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(psnr_list)
ax.set_ylim([0, 50])
fig.suptitle('PSNR of each image with mean PSNR: {}'.format(mean_psnr))
fig.savefig('plots/{}_{}_psnr.png'.format(os.path.split(gt_dir)[-1], os.path.split(test_dir)[-1]))

fig = plt.figure()
ax = plt.subplot(111)
plt.plot(ssim_list)
ax.set_ylim([-1, 1])
fig.suptitle('SSIM of each image with mean SSIM: {}'.format(mean_ssim))
fig.savefig('{}_{}_ssim.png'.format(os.path.split(gt_dir)[-1], os.path.split(test_dir)[-1]))

