import argparse
import os
import os.path
import random
import time
import numpy as np
import h5py
import cv2

import torch
from patchcraft_utils import *
from data_utils import swap_frames, pad_frames, get_noise_stat, save_patches, get_cov_mean

# import matplotlib.pyplot as plt


def main_preprocess_real_noise(opt):
    train_dir = os.path.join(opt.dataset_dir, 'training_sequences/noisy')
    seq_names = sorted(os.listdir(train_dir))
    for seq_i in range(len(seq_names)):
        seq_path = os.path.join(train_dir, seq_names[seq_i])
        assert_message = ("sequence \"{}\" has {} frames. ".format(seq_path, len(os.listdir(seq_path))) +
        "Each sequence must have exactly {} frames (frame_num = {})".format(opt.frame_num, opt.frame_num))
        assert len(os.listdir(seq_path)) == opt.frame_num, f"{assert_message}"

    if not os.path.isdir('./logs'):
        os.makedirs('./logs')

    log_f_name = './logs/log_preprocess_unknown_noise_{}.txt'.format(opt.dataset_name)
    log_f = open(log_f_name, "w")

    device = torch.device('cuda') if torch.cuda.is_available() and opt.cuda else torch.device('cpu')

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    reflect_pad = (0, opt.patch_size - 1, opt.patch_size - 1)
    const_pad = (opt.search_offs, opt.search_offs,
                 opt.search_offs, opt.search_offs,
                 0, 0)

    print('Processing training data. Please wait, it may take time...')
    log_f.write('Processing training data. Please wait, it may take time...\n')
    log_f.flush()

    stat_dict = {'pc_n2n_mean_arr': [], 'pc_n2n_n_cov_arr': []}

    h5_dir = os.path.join(opt.dataset_dir, 'h5_files')
    if not os.path.isdir(h5_dir):
        os.makedirs(h5_dir)
    cov_mean_dir = os.path.join(opt.dataset_dir, 'cov_mean_files')
    if not os.path.isdir(cov_mean_dir):
        os.makedirs(cov_mean_dir)

    h5f_train_file = os.path.join(h5_dir, 'train_{}.h5'.format(opt.dataset_name))
    with h5py.File(h5f_train_file, "w") as h5f_train:
        patch_params = {'h5f_train': h5f_train, 'arr_s': None, 'arr_e': 0, 'patch_num': None,
                        'patches_per_offs': None, 'train_size': 0, 'patch_i_arr': None}

        for seq_i in range(len(seq_names)):
            name_tmp = seq_names[seq_i]
            seq_path = os.path.join(train_dir, name_tmp)
            fr_names = sorted(os.listdir(seq_path))

            seq_noisy = list()
            for fr_i in range(opt.frame_num):
                im_path = os.path.join(seq_path, fr_names[fr_i])
                im_read = cv2.imread(im_path, -1)
                im_read = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)
                if opt.image_format == 'png_16_bit':
                    seq_noisy.append(np.float32(im_read) / 65535)
                else:
                    seq_noisy.append(np.float32(im_read))
            seq_noisy = np.array(seq_noisy)

            with torch.no_grad():
                seq_noisy = torch.from_numpy(seq_noisy).permute(3, 0, 1, 2).to(device)
                if opt.replicate_dataset:
                    fr_ind_list = list(range(opt.frame_num))
                else:
                    fr_ind_list = list((opt.frame_num // 2,))
                for fr_i in fr_ind_list:
                    time_s = time.time()

                    seq_noisy_swap_pad = swap_frames(opt, fr_i, seq_noisy)
                    seq_noisy_swap_pad = pad_frames(seq_noisy_swap_pad, reflect_pad, const_pad)
                    min_i = find_nn(opt, seq_noisy_swap_pad)

                    im_n = seq_noisy[:, fr_i, ...].clone()

                    v_num = len(range(0, im_n.shape[-2] - opt.tile_size + 1, opt.stride))
                    h_num = len(range(0, im_n.shape[-1] - opt.tile_size + 1, opt.stride))
                    patch_params['patch_num'] = v_num * h_num
                    patch_params['patch_i_arr'] = torch.randperm(patch_params['patch_num'])
                    patch_params['patches_per_offs'] = patch_params['patch_num'] / \
                        (opt.patch_size ** 2 * opt.frame_num)

                    for offs_i in range(opt.patch_size ** 2):
                        im_pc = create_im(opt, seq_noisy_swap_pad, min_i, offs_i)
                        get_noise_stat(opt.tile_size, im_n, im_pc, opt.patch_size ** 2, stat_dict)
                        n2n_image = torch.cat((im_n, im_pc), dim=0).cpu()
                        save_patches(opt, patch_params, n2n_image, offs_i)

                    time_e = time.time()
                    time_es = time_e - time_s
                    if opt.replicate_dataset:
                        print("sequence \'{}\': frame {} done, elapsed time {:.2f}".format(
                            seq_names[seq_i].upper(), fr_i, time_es))
                        log_f.write("sequence \'{}\': frame {} done, elapsed time {:.2f}\n".format(
                            seq_names[seq_i].upper(), fr_i, time_es))
                    else:
                        print("sequence \'{}\' done, elapsed time {:.2f}".format(
                            seq_names[seq_i].upper(), time_es))
                        log_f.write("sequence \'{}\' done, elapsed time {:.2f}\n".format(
                            seq_names[seq_i].upper(), time_es))
                    log_f.flush()
    stat_dict['pc_n2n_mean_arr'] = np.array(stat_dict['pc_n2n_mean_arr'])
    stat_dict['pc_n2n_n_cov_arr'] = np.array(stat_dict['pc_n2n_n_cov_arr'])

    cov_mean_dict = get_cov_mean(opt, stat_dict)
    cov_mean_file = os.path.join(cov_mean_dir, 'cov_mean_{}.pt'.format(opt.dataset_name))
    torch.save(cov_mean_dict, cov_mean_file)

    print("Creating training set for {} dataset done.".format(opt.dataset_name.upper()))
    print("Training size {}".format(patch_params['train_size']))
    log_f.write("Creating training set for dataset {} done.".format(opt.dataset_name.upper()))
    log_f.write("Training size {}\n".format(patch_params['train_size']))
    log_f.flush()


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # parser.add_argument("--dataset_name", default='CRVD_ISO25600', type=str, help="dataset name")
    # parser.add_argument("--dataset_dir", default="./data_set/CRVD/CRVD_ISO25600", help="path to data directory")
    parser.add_argument("--dataset_name", default='CRVD_ISO1600', type=str, help="dataset name")
    parser.add_argument("--dataset_dir", default="./data_set/CRVD/CRVD_ISO1600", help="path to data directory")
    parser.add_argument("--tile_size", default=50, help="tile size")
    parser.add_argument("--stride", default=10, help="stride")
    parser.add_argument("--aug-times", default=2, help="number of augmentations")

    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--cuda", default=1, type=int, help="use gpu")

    # nn search arguments
    parser.add_argument("--frame_num", default=7, type=int, help="number of frames in the input video sequence")
    parser.add_argument("--search_skip", default=3, type=int, help="search skip")
    parser.add_argument("--search_offs", default=500, type=int, help="search offset")
    parser.add_argument("--search_offs2", default=37, type=int, help="search offset2")
    # parser.add_argument("--patch_size", default=37, type=int, help="patch size for nearest neighbor search")
    parser.add_argument("--patch_size", default=15, type=int, help="patch size for nearest neighbor search")

    # parser.add_argument("--replicate_dataset", default=0, type=int, help="1 - replicate dataset, 0 - don't replicate dataset")
    parser.add_argument("--replicate_dataset", default=1, type=int, help="1 - replicate dataset, 0 - don't replicate dataset")
    parser.add_argument("--image_format", default='png_16_bit', type=str, help='8_bit | png_16_bit')

    parser.add_argument("--bins", type=int, default=50001, help="number of bins")
    parser.add_argument("--std_num", type=float, default=1, help="number of STDs")

    opt = parser.parse_args()

    assert opt.frame_num % 2 == 1, f"frame num = {opt.frame_num} is illegal, frame_num must be an odd number"

    return opt


if __name__ == "__main__":
    """ This script prepares a training set for self-supervised patch-craft training with an unknown noise model. 
    The script takes as input a path to a folder containing noisy video sequences (or bursts of images). 
    The script does not require knowlege of the noise model. 
    
    The folder contaning the noisy video sequences should be organized as follows:
    
    dataset_folder
    └── train_sequences
        └── noisy
            ├── sequence_0
            |     ├── frame_0
            |     ├── frame_1
            |     ├── ...
            |     └── frame_K
            ├── sequence_1
            |     ├── frame_0
            |     ├── frame_1
            |     ├── ...
            |     └── frame_K
            |
            ├── ...
            |
            └── sequence_M
                ├── frame_0
                ├── frame_1
                ├── ...
                └── frame_K

    Note that each video sequence must contain the same number of frames which is specified by 'frame_num' parameter.

    Arguments:
    --dataset_name (string): Name of the dataset.
    --dataset_dir (string): Path to a folder containing noisy video sequences or bursts of images.
    --tile_size (int): Size of samples in the training set.
    --frame_num(int): Number of frames in each video sequence.
    --search_offs(int): Defines spatial bounding box for nearest neaighbor search. The bounding box size is (2 * search_offs + 1) x (2 * search_offs + 1)
    --search_skip(int): Defines skip size for nearest neaighbor search.
    --search_offs2(int): Applicable only when search_skip > 1. Defines spatial bounding box for nearest neaighbor search with skip = 1. 
                         The bounding box size is (2 * search_offs2 + 1) x (2 * search_offs2 + 1)
    --patch_size (int): Specifies patch size used in patch matching.
    --replicate_dataset(int): If replicate_dataset = 1, then the dataset is augmented by replicating each sequence frame_num - 1 times 
                              (to the total amount of frame_num frames), where in each replica a different image is used as a middle frame.
    --image_format(string): Defines images format. 8_bit - any 8 bit format supported by OpenCV imread function (e.g., PNG, JPG), png_16_bit - PNG-16 bit.
                            To reduce quantization errors we recommend to store noisy images in PNG-16 format. 

    Outputs:
    The script outputs are 'h5_files/train_<dataset name>.h5' and 'cov_mean_files/cov_mean_<dataset name>.pt' files. 

    Example usage:
    python data/preprocess_synthetic_noise.py --dataset_name CRVD_ISO25600 --dataset_dir "./data_set/CRVD/CRVD_ISO25600" --replicate_dataset 1

    """

    args = get_args()
    main_preprocess_real_noise(args)

