import argparse
import os
import os.path
import re
import random
import time
import numpy as np
import h5py
import cv2

import torch
from patchcraft_utils import *
from data_utils import pad_frames, get_noise_stat, save_patches, get_cov_mean


class AddCorrelatedGaussianNoise(object):
    """ This class provides an object for adding correlated Gaussian noise to an input 
    video sequence or a burst of images. The object generates independent and identically 
    distributed (i.i.d.) Gaussian noise, convolves the noise with a rectangular kernel of 
    a given size and adds it to the input video sequence.

    Arguments:
        - sigma (float): The standard deviation of the Gaussian noise
        - noise_kernel_size (int): The size of the rectangular kernel used to convolve the noise.
        - sequence (tensor of size 3 x K x M x N): An input video sequence composed of K frames of size M x N

    Example usage:
    
    # Instantiate the noise object with a noise sigma of 20 and a kernel size of 3x3
    add_noise = AddCorrelatedGaussianNoise(20, 3)

    # Add noise to an input video sequence
    noisy_sequence = add_noise(sequence)

    """

    def __init__(self, sigma, noise_kernel_size):

        self.sigma = sigma
        self.noise_kernel_size = noise_kernel_size

        return


    def __call__(self, sequence):
        noisy_sequence = None
        
        sequence = sequence.unsqueeze(0)
        b, c, t, v, h = sequence.shape
        noise_size = (b, c, t, v + self.noise_kernel_size - 1, h + self.noise_kernel_size - 1)
        noise = (self.sigma / 255) * torch.randn(noise_size, dtype=sequence.dtype).to(sequence.device)
        noise = noise.permute(0, 2, 1, 3, 4).reshape(b * t, c, noise_size[-2], noise_size[-1])
        noise_kernel = torch.ones(3, 1, self.noise_kernel_size, self.noise_kernel_size).\
            to(sequence.device) / self.noise_kernel_size
        noise = torch.nn.functional.conv2d(noise, noise_kernel, padding=0, groups=3)
        noise = noise.reshape(b, t, c, v, h).permute(0, 2, 1, 3, 4)
        noisy_sequence = sequence + noise
        noisy_sequence = noisy_sequence.squeeze(0)

        return noisy_sequence


class AddCustomNoise(object):
    """ This class is a template for creating a custom noise object for adding noise to an input 
    video sequence or a burst of images. To use this class, the user should implement 
    the __init__ and __call__ functions, defining the necessary arguments 
    and functionality for their custom noise generation.


    Arguments: The user should define the necessary arguments for their custom noise generation.

    Example usage:
    
    # Instantiate the custom noise object
    add_noise = AddCustomNoise(arg1, arg2, ...)

    # Add custom noise to an input video sequence
    noisy_sequence = add_noise(sequence)

    """

    def __init__(self, *args):

        """ Add your code here (replace *args with your arguments) """
        
        return


    def __call__(self, sequence):
        noisy_sequence = None
        
        """ Implement adding noise here """

        return noisy_sequence


def main_preprocess_synthetic_noise(opt, add_noise):
    train_dir = os.path.join(opt.dataset_dir, 'training_sequences')
    seq_names = sorted(os.listdir(train_dir))
    for seq_i in range(len(seq_names)):
        seq_path = os.path.join(train_dir, seq_names[seq_i])

    if opt.noise_type == 'correlated_gaussian':
        name_str = 'noise_sig{}_ker{}'.format(opt.sigma, opt.noise_kernel_size)
    else:
        name_str = 'noise'

    if not os.path.isdir('./logs'):
        os.makedirs('./logs')
    log_f_name = './logs/log_preprocess_{}_{}_{}_patch{}_search_offs{}.txt'.format(\
        opt.dataset_name, opt.noise_type, name_str, opt.patch_size, opt.search_offs)
    log_f = open(log_f_name, "w")

    device = torch.device('cuda') if torch.cuda.is_available() and opt.cuda else torch.device('cpu')

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    reflect_pad = (0, opt.patch_size - 1, opt.patch_size - 1)
    const_pad = (opt.search_offs, opt.search_offs, opt.search_offs, opt.search_offs, 0, 0)

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

    if opt.davis_use_same_frames:
        center_i_file = os.path.join('davis_center_i_files', 'center_i_sig{}_ker{}.pt'.format(\
            opt.sigma, opt.noise_kernel_size))
        center_i_arr = torch.load(center_i_file) 

    h5f_train_file = os.path.join(h5_dir, 'train_{}_{}_{}.h5'.format(\
        opt.dataset_name, opt.noise_type, name_str))
    with h5py.File(h5f_train_file, "w") as h5f_train:
        patch_params = {'h5f_train': h5f_train, 'arr_s': None, 'arr_e': 0, 'patch_num': None,
                        'patches_per_offs': None, 'train_size': 0, 'patch_i_arr': None}
        for seq_i in range(len(seq_names)):
            time_s = time.time()
            name_tmp = seq_names[seq_i]
            seq_path = os.path.join(train_dir, name_tmp)
            fr_names = sorted(os.listdir(seq_path))

            if opt.davis_use_same_frames:
                center_i = center_i_arr[seq_i]
            else:
                center_i = np.random.randint(opt.frame_num // 2, len(fr_names) - opt.frame_num // 2)                    

            seq_clean = list()
            for fr_i in range(center_i - opt.frame_num // 2, center_i + opt.frame_num // 2 + 1):
                im_path = os.path.join(seq_path, fr_names[fr_i])
                im_read = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
                seq_clean.append(im_read)
            seq_clean = np.array(seq_clean, dtype=np.float32) / 255
            seq_clean = torch.from_numpy(seq_clean).permute(3, 0, 1, 2)
            seq_noisy = add_noise(seq_clean)
            with torch.no_grad():
                seq_noisy = seq_noisy.to(device)
                seq_noisy_pad = pad_frames(seq_noisy, reflect_pad, const_pad)
                min_i = find_nn(opt, seq_noisy_pad)
                im_n = seq_noisy[:, opt.frame_num // 2, ...]

                v_num = len(range(0, im_n.shape[-2] - opt.tile_size + 1, opt.stride))
                h_num = len(range(0, im_n.shape[-1] - opt.tile_size + 1, opt.stride))
                patch_params['patch_num'] = v_num * h_num
                patch_params['patch_i_arr'] = torch.randperm(patch_params['patch_num'])
                patch_params['patches_per_offs'] = patch_params['patch_num'] / (opt.patch_size ** 2 * opt.patch_factor)

                for offs_i in range(opt.patch_size ** 2):
                    im_pc = create_im(opt, seq_noisy_pad, min_i, offs_i)
                    get_noise_stat(opt.tile_size, im_n, im_pc, opt.patch_size ** 2, stat_dict)
                    n2n_image = torch.cat((im_n, im_pc), dim=0).cpu()
                    save_patches(opt, patch_params, n2n_image, offs_i)

                time_e = time.time()
                time_es = time_e - time_s
                print("{}/{}: sequence {} done, elapsed time {:.2f}".format(
                    seq_i, len(seq_names) - 1, name_tmp.upper(), time_es))
                log_f.write("{}/{}: sequence {} done, elapsed time {:.2f}\n".format(
                    seq_i, len(seq_names) - 1, name_tmp.upper(), time_es))
                log_f.flush()
    stat_dict['pc_n2n_mean_arr'] = np.array(stat_dict['pc_n2n_mean_arr'])
    stat_dict['pc_n2n_n_cov_arr'] = np.array(stat_dict['pc_n2n_n_cov_arr'])

    cov_mean_dict = get_cov_mean(opt, stat_dict)
    cov_mean_file = os.path.join(cov_mean_dir, 'cov_mean_{}_{}_{}.pt'.format(\
        opt.dataset_name, opt.noise_type, name_str))
    torch.save(cov_mean_dict, cov_mean_file)

    print("Creating training set 'train_{}_{}_{}.h5' done.".format(\
        opt.dataset_name, opt.noise_type, name_str))
    print("Training size {}".format(patch_params['train_size']))
    log_f.write("Creating training set 'train_{}_{}_{}.h5' done.".format(\
        opt.dataset_name, opt.noise_type, name_str))
    log_f.write("Training size {}\n".format(patch_params['train_size']))
    log_f.flush()


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset_name", default='DAVIS', type=str, help="dataset name")
    parser.add_argument("--dataset_dir", default="./data_set/DAVIS", help="path to data directory")
    parser.add_argument("--tile_size", default=50, help="tile size")
    parser.add_argument("--stride", default=30, help="stride")
    parser.add_argument("--aug_times", default=2, help="number of augmentations")
    parser.add_argument("--patch_factor", type=int, default=1, help="patch factor")

    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--noise_type", type=str, default="correlated_gaussian", help="correlated_gaussian | custom")

    # for correlated Gaussian noise
    parser.add_argument("--sigma", default=20.0, type=float, help="gaussian noise sigma (applicable when the 'noise_type' is 'correlated_gaussian')")
    parser.add_argument("--noise_kernel_size", default=3, type=int, help="noise kernel size (applicable when the 'noise_type' is 'correlated_gaussian')")
    parser.add_argument("--davis_use_same_frames", action='store_true', help="for simulations with DAVIS: use same frame numbers at any simulation ")

    parser.add_argument("--cuda", default=1, type=int, help="use gpu")

    parser.add_argument("--frame_num", default=7, type=int, help="number of frames in the input video sequence")
    parser.add_argument("--search_offs", default=37, type=int, help="search offset")
    parser.add_argument("--patch_size", default=43, type=int, help="patch size for nearest neighbor search")

    parser.add_argument("--bins", type=int, default=50001, help="number of bins")
    parser.add_argument("--std_num", type=float, default=3, help="number of STDs")

    opt = parser.parse_args()
    opt.search_offs2 = None
    opt.search_skip = 1

    assert opt.frame_num % 2 == 1, f"frame num = {opt.frame_num} is illegal, frame_num must be an odd number"

    return opt


if __name__ == "__main__":
    """ This script prepares a training set for self-supervised patch-craft training with synthetic noise. 
    The script takes as input a path to a folder containing clean video sequences (or bursts of images). 
    The script can be used with any noise model. 
    
    For using correlated Gaussian noise, set the 'noise_type' to 'correlated_gaussian' and select
    the values of 'sigma' and 'noise_kernel_size'. For applying the script with any other noise model, 
    set the 'noise_type' to 'custom' and implement the noise model in the 'AddCustomNoise' class.

    Note that to use 'AddCustomNoise', the user should implement __init__ and __call__ functions, 
    defining the necessary arguments and functionality for their custom noise generation.

    The folder contaning the clean video sequences should be organized as follows
    
    dataset_folder
    └── train_sequences
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
    --dataset_dir (string): Path to a folder containing clean video sequences (or bursts of images).
    --tile_size (int): Size of samples in the training set.
    --noise_type (string): Noise type. For correlated Gaussian noise set to 'correlated_gaussian' and 
                           configure 'sigma" and 'noise_kernel_size' parameters. For any other noise model 
                           set to 'custom' and implement __init__ and __call__ functions of the 'AddCustomNoise' class.
    --sigma (float): The standard deviation of the input noise (applicable only when 
                     the 'noise_type' is 'correlated_gaussian').
    --noise_kernel_size (int): the size of the rectangular kernel used to convolve the noise. 
                               (applicable only when the 'noise_type' is 'correlated_gaussian').
    --frame_num(int): Number of frames in each video sequence.
    --search_offs(int): Defines spatial bounding box for nearest neaighbor search. The bounding box size is (2 * search_offs + 1) x (2 * search_offs + 1)
    --patch_size (int): Specifies patch size used in patch matching. It should be increased with 'noise_sigma' and 'noise_kernel_size'.

    Outputs:
    The script outputs are 'h5_files/train_synthetic_noise_<noise type related string>.h5' and 'cov_mean_files/cov_mean_<noise type related string>.pt' files. 

    Example usage:
    python data/preprocess_synthetic_noise.py --dataset_name DAVIS --dataset_dir "./data_set/DAVIS" --noise_type correlated_gaussian --sigma 20.0 --noise_kernel_size 3 

    """

    opt = get_args()

    if opt.noise_type == "custom":
        add_noise = AddCustomNoise("replace this string with a parameter list")
    else:
        add_noise = AddCorrelatedGaussianNoise(opt.sigma, opt.noise_kernel_size)

    main_preprocess_synthetic_noise(opt, add_noise)

