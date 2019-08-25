import sys
import os
import configs
import ZSSR


def main(input_img_list, ground_truth_list, kernels_list, gpu, conf_str, results_path):
    # Choose the wanted GPU
    gpu=0
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = '%s' % gpu

    # 0 input for ground-truth or kernels means None
    ground_truth_list = [None if ground_truth == '0' else ground_truth for ground_truth in ground_truth_list]
    # print('*****', kernels_list)
    kernels_list = [None if kernels == '0' else kernels.split(';')[:-1] for kernels in kernels_list]

    # Setup configuration and results directory
    conf = configs.Config()
    if conf_str is not None:
        exec ('conf = configs.%s' % conf_str)
    conf.result_path = results_path

    # Run ZSSR on the image
    net = ZSSR.ZSSR(input_img_list, conf, ground_truth_list, kernels_list)
    net.run()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
