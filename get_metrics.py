import cv2
import os
import numpy as np

def mse(src: np.ndarray, test: np.ndarray):
    rows, cols = src.shape[:2]

    diff = (src-test) ** 2
    return diff.sum()/(rows*cols)

def psnr(src: np.ndarray, test: np.ndarray):
    max_val = np.max(src)
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

gt_dir = 'gt_data'
test_dir = 'resize_results_zssr_sing_img'

gt_files = os.listdir(gt_dir)
gt_files.sort()
test_files = os.listdir(test_dir)
test_files.sort()

psnr_list, ssim_list = [], []
for gt, test_f in zip(gt_files, test_files):
    src = cv2.imread(os.path.join(gt_dir, gt))
    test = cv2.imread(os.path.join(test_dir, test_f))

    psnr_list.append(psnr(src, test))
    ssim_list.append(ssim(src, test))

mean_psnr = np.mean(np.asarray(psnr_list))
mean_ssim = np.mean(np.asarray(ssim_list))

print('Mean PSNR / Mean SSIM : {} / {}'.format(mean_psnr, mean_ssim))
np.save('{}_{}.npy'.format(gt_dir, test_dir), np.array([mean_psnr, mean_ssim]))