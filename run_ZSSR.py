import GPUtil
import glob
import os
from utils import prepare_result_dir
import configs
from time import sleep
import sys
import run_ZSSR_single_input
import argparse


def main(args):
    conf_name = args.configs if args.configs != '' else None
    gpu = args.gpu if args.gpu != '' else None
    # Initialize configs and prepare result dir with date
    if conf_name is None:
        conf = configs.Config()
    else:
        conf = None
        exec ('conf = configs.%s' % conf_name)
    conf.batch_size = args.batch_size
    conf.start_ind = args.start
    conf.depth = args.nlayers
    conf.reinitialize_filters()
    res_dir = prepare_result_dir(conf)
    local_dir = os.path.dirname(__file__)

    # We take all png files that are not ground truth
    scene_list = os.listdir(args.src)
    scene_list.sort()
    for i in range(args.start, len(scene_list)):
        scene = scene_list[i]
        conf.input_path = os.path.join(args.src, scene)
        conf.result_path = os.path.join(args.save_path, scene)
        files = [file_path for file_path in glob.glob('%s/*.png' % conf.input_path)
                 if not file_path[-7:-4] == '_gt']
        files.sort()

        # Loop over all the bsd_001.pngfiles
        kernel_files_str_list, ground_truth_file_list = [], []
        for file_ind, input_file in enumerate(files):

            # Ground-truth file needs to be like the input file with _gt (if exists)
            ground_truth_file = input_file[:-4] + '_gt.png'
            if not os.path.isfile(ground_truth_file):
                ground_truth_file = '0'
            ground_truth_file_list.append(ground_truth_file)

            # Numeric kernel files need to be like the input file with serial number
            kernel_files = ['%s_%d.mat;' % (input_file[:-4], ind) for ind in range(len(conf.scale_factors))]
            kernel_files_str = ''.join(kernel_files)
            for kernel_file in kernel_files:
                if not os.path.isfile(kernel_file[:-1]):
                    kernel_files_str = '0'
                    # print('no kernel loaded')
                    break
            kernel_files_str_list.append(kernel_files_str)

            # print('kernel files: ')
            # print(kernel_files)

            # This option uses all the gpu resources efficiently
        # if gpu == 'all':
        #
        #     # Stay stuck in this loop until there is some gpu available with at least half capacity
        #     gpus = []
        #     while not gpus:
        #         gpus = GPUtil.getAvailable(order='memory')
        #
        #     # Take the gpu with the most free memory
        #     cur_gpu = gpus[-1]
        #
        #     # Run ZSSR from command line, open xterm for each run
        #     os.system("xterm -hold -e " + conf.python_path +
        #               " %s/run_ZSSR_single_input.py '%s' '%s' '%s' '%s' '%s' '%s' alias python &"
        #               % (local_dir, input_file, ground_truth_file, kernel_files_str, cur_gpu, conf_name, res_dir))
        #
        #     # Verbose
        #     print('Ran file #%d: %s on GPU %d\n' % (file_ind, input_file, cur_gpu))
        #
        #     # Wait 5 seconds for the previous process to start using GPU. if we wouldn't wait then GPU memory will not
        #     # yet be taken and all process will start on the same GPU at once and later collapse.
        #     sleep(5)
        #
        # # The other option is just to run sequentially on a chosen GPU.
        # else:
        run_ZSSR_single_input.main(files, ground_truth_file_list, kernel_files_str_list, gpu, conf_name, res_dir, conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to run the ZSSR algorithm')
    parser.add_argument('--configs', type=str, default='', help='path to config file')
    parser.add_argument('--gpu', type=str, help='GPU device to be used', default='')
    parser.add_argument('--src', type=str, default='the base directory for the set of scene frames of a video')
    parser.add_argument('-b', default=1, type=int, help='size of training batch', dest='batch_size')
    parser.add_argument('-s', default='results', type=str, help='directory to store results', dest='save_path')
    parser.add_argument('--start', default=0, type=int, help='batch number to re-start the predictions')
    parser.add_argument('--nlayers', default=8, type=int, help='number of layers in neural network')
    # conf_str = sys.argv[1] if len(sys.argv) > 1 else None
    # gpu_str = sys.argv[2] if len(sys.argv) > 2 else None
    args = parser.parse_args()
    main(args)
    # main(conf_str, gpu_str)
