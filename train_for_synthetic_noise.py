import argparse
import sys
import numpy as np
import os
import time
import cv2

import torch
import torch.nn.functional as F

from skimage.metrics import structural_similarity

from utils import data
import utils, models
from models.model_utils import *

sys.path.insert(1, 'data')
from preprocess_synthetic_noise import AddCorrelatedGaussianNoise, AddCustomNoise


def training_func(opt, load_model):
    if opt.noise_type == 'correlated_gaussian':
        name_str = 'noise_sig{}_ker{}'.format(opt.sigma, opt.noise_kernel_size)
        add_noise = AddCorrelatedGaussianNoise(opt.sigma, opt.noise_kernel_size)
    else:
        name_str = 'noise'
        add_noise = AddCustomNoise("replace this string with a parameter list")

    log_f_name = './logs/log_train_{}_{}_{}_{}.txt'.format(\
        opt.model, opt.dataset_name, opt.noise_type, name_str)
    log_f = open(log_f_name, "w")

    if opt.reduce_dependency:
        cov_mean_dir = os.path.join(opt.dataset_dir, 'cov_mean_files')
        cov_mean_file = os.path.join(cov_mean_dir, 'cov_mean_{}_{}_{}.pt'.format(\
            opt.dataset_name, opt.noise_type, name_str))
        if not os.path.exists(cov_mean_file):
            assert_message = ("file \"{}\" does not exist. Please run \"./data/preprocess_synthetic_noise.py\" script to create it.".format(cov_mean_file))
            assert True, f"{assert_message}"
        cov_mean_dict = torch.load(cov_mean_file)
        opt.min_cov = cov_mean_dict['min_cov']
        opt.min_mean = cov_mean_dict['min_mean']
        opt.max_mean = cov_mean_dict['max_mean']
        opt.remove_mean = 1
        opt.mean_val = cov_mean_dict['mean_val']
    else:
        opt.min_cov = -np.inf
        opt.min_mean = -np.inf
        opt.max_mean = np.inf
        opt.remove_mean = 0
        opt.mean_val = 0

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(opt.seed)
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load model
    in_model_path = './experiments/{}/orig/model.pt'.format(opt.model)
    model = load_model(opt, device, in_model_path)
    print(f"Built {opt.model.upper()} model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")
    log_f.write(f"Built {opt.model.upper()} model consisting of {sum(p.numel() for p in model.parameters()):,} parameters\n")
    log_f.flush()

    out_model_dir = './experiments/{}/learned/{}'.format(opt.model, opt.dataset_name)
    if not os.path.isdir(out_model_dir):
        os.makedirs(out_model_dir)

    # Build data loader and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.5)
    
    dataset_path = os.path.join(opt.dataset_dir, 'h5_files')
    train_loader = data.build_dataset('synthetic_noise', dataset_path, opt.dataset_name, batch_size=opt.batch_size,
        name_str=name_str, noise_type=opt.noise_type, num_workers=opt.num_workers)

    valid_dir = os.path.join(opt.dataset_dir, 'training_sequences')
    seq_names = sorted(os.listdir(valid_dir))
    val_ind = 2

    train_loss_mean_arr = list()
    train_psnr_mean_arr = list()
    valid_psnr_mean_arr = list()
    valid_ssim_mean_arr = list()

    toolbar_width = 40
    print("Training {}".format(opt.model.upper()))
    print("-" * toolbar_width)
    for epoch in range(opt.num_epochs):
        time_s = time.time()

        tl_inc = train_loader.__len__() // toolbar_width + 1
        head_str = "[training {}/{}]".format(epoch, opt.num_epochs - 1)
        sys.stdout.write("{}[{}]".format(head_str, " " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width + 1))
        char_num = len(head_str) + toolbar_width + 2
        print_cnt = 0

        train_loss_arr = list()
        train_psnr_arr = list()
        model.train()
        for batch_id, sample in enumerate(train_loader):
            im_in = sample[:, 0:3, ...].clone().to(device)
            im_ref = sample[:, 3:6, ...].clone().to(device)
            if (not np.isinf(opt.min_cov)) or (not np.isinf(opt.max_mean)) or (not np.isinf(opt.min_mean)):
                im_z = im_ref - im_in
                in_z_cov = (im_in * im_z).mean((1, 2, 3)) - im_in.mean((1, 2, 3)) * im_z.mean((1, 2, 3))
                im_z_mean = im_z.mean((1, 2, 3))
                im_ind = opt.min_cov < in_z_cov
                im_ind = torch.logical_and(im_ind, im_z_mean < opt.max_mean)
                im_ind = torch.logical_and(im_ind, im_z_mean > opt.min_mean)
                im_in = im_in[im_ind, ...]
                im_ref = im_ref[im_ind, ...]
            if opt.remove_mean:
                im_ref = im_ref - opt.mean_val

            im_out = model(im_in)
            loss = F.mse_loss(im_out, im_ref, reduction="sum") / (im_ref.size(0) * 2)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_psnr = -10 * ((im_out - im_ref) ** 2).mean().log10().item()
            train_loss_arr.append(loss.item())
            train_psnr_arr.append(train_psnr)

            if batch_id % tl_inc == 0:
                print_cnt += 1
                sys.stdout.write("-")
                sys.stdout.flush()
        sys.stdout.write("{}]".format("-" * (toolbar_width - print_cnt)))
        sys.stdout.write("\b" * char_num)

        if epoch % opt.valid_interval == 0:
            tl_inc = len(seq_names) // toolbar_width + 1
            head_str = "[validation {}/{}]".format(epoch, opt.num_epochs - 1)
            sys.stdout.write("{}[{}]".format(head_str, " " * toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width + 1))
            char_num = len(head_str) + toolbar_width + 2
            print_cnt = 0

            valid_psnr_arr = list()
            valid_ssim_arr = list()
            model.eval()

            for s_i in range(len(seq_names)):
                seq_dir = os.path.join(valid_dir, seq_names[s_i])
                clean_im_name = sorted(os.listdir(seq_dir))[val_ind]
                clean_path_in = os.path.join(seq_dir, clean_im_name)
                valid_im_c = cv2.cvtColor(cv2.imread(clean_path_in), cv2.COLOR_BGR2RGB)
                valid_im_c = np.float32(valid_im_c) / 255
                valid_im_c = torch.from_numpy(valid_im_c).permute(2, 0, 1).to(device)
                with torch.no_grad():
                    valid_im_n = valid_im_c.clone().unsqueeze(1)
                    valid_im_n = add_noise(valid_im_n).squeeze(1)
                    valid_im_dn = model(valid_im_n.unsqueeze(0)).squeeze(0).clamp(0, 1)
                    valid_psnr = -10 * ((valid_im_dn - valid_im_c) ** 2).mean().log10().item()
                    valid_ssim = structural_similarity(valid_im_dn.permute(1, 2, 0).cpu().numpy(),
                                                       valid_im_c.permute(1, 2, 0).cpu().numpy(), multichannel=True, data_range=1.0)
                    valid_psnr_arr.append(valid_psnr)
                    valid_ssim_arr.append(valid_ssim)
                    if s_i % tl_inc == 0:
                        print_cnt += 1
                        sys.stdout.write("-")
                        sys.stdout.flush()
            sys.stdout.write("{}]".format("-" * (toolbar_width - print_cnt)))
            sys.stdout.write("\b" * char_num)

            train_loss_mean_arr.append(np.array(train_loss_arr).mean())
            train_psnr_mean_arr.append(np.array(train_psnr_arr).mean())
            valid_psnr_mean_arr.append(np.array(valid_psnr_arr).mean())
            valid_ssim_mean_arr.append(np.array(valid_ssim_arr).mean())

            time_e = time.time()
            time_es = time_e - time_s
            print("epoch {}/{}: train loss {:.3f} | train psnr {:.3f} | valid psnr {:.3f} | valid ssim {:.3f} | elapsed time {:.2f}".
                format(epoch, opt.num_epochs - 1, train_loss_mean_arr[-1], train_psnr_mean_arr[-1], 
                       valid_psnr_mean_arr[-1], valid_ssim_mean_arr[-1], time_es))
            log_f.write("epoch {}/{}: train loss {:.3f} | train psnr {:.3f} | valid psnr {:.3f} | valid ssim {:.3f} | elapsed time {:.2f}\n".
                        format(epoch, opt.num_epochs - 1, train_loss_mean_arr[-1], train_psnr_mean_arr[-1], 
                               valid_psnr_mean_arr[-1], valid_ssim_mean_arr[-1], time_es))
            log_f.flush()

        if epoch % opt.save_interval == 0:
            state_params = {'model_params': model.state_dict(),
                            'optimizer_params': optimizer.state_dict(),
                            'scheduler_params': scheduler.state_dict(),
                            'train_loss_mean_arr': train_loss_mean_arr,
                            'train_psnr_mean_arr': train_psnr_mean_arr,
                            'valid_psnr_mean_arr': valid_psnr_mean_arr,
                            'valid_ssim_mean_arr': valid_ssim_mean_arr}
            out_model_path = os.path.join(out_model_dir, 'state_params_{}_{}.pt'.format(opt.noise_type, name_str))
            torch.save(state_params, out_model_path)

        scheduler.step()


def custom_training_func(opt, load_model):

    """ Add your code here """

    return


def main(opt):
    if opt.model == 'dncnn':
        load_model = load_model_dncnn
    elif opt.model == 'unet':
        load_model = load_model_unet
    elif opt.model == 'custom':
        load_model = load_model_custom
    else:
        assert_message = ("model = {} is undefined. ".format(opt.model))
        assert True, f"{assert_message}"

    if opt.training_function == 'ready-made':
        train_model = training_func
    elif opt.training_function == 'custom':
        train_model = custom_training_func
    else:
        assert_message = ("training_function = {} is undefined. ".format(opt.training_function))
        assert True, f"{assert_message}"

    train_model(opt, load_model)


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--dataset_name", default='DAVIS', type=str, help="dataset name")
    parser.add_argument("--dataset_dir", default="./data_set/DAVIS", type=str, help="path to the dataset")
    parser.add_argument("--batch_size", default=128, type=int, help="train batch size")
    parser.add_argument("--reduce_dependency", type=int, default=1, help="reduce dependency")

    # Add noise arguments
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--noise_type", type=str, default="correlated_gaussian", help="correlated_gaussian | custom_noise")

    # For correlated Gaussian noise
    parser.add_argument("--sigma", default=20.0, type=float, help="gaussian noise sigma (applicable when the 'noise_type' is 'correlated_gaussian')")
    parser.add_argument("--noise_kernel_size", default=3, type=int, help="noise kernel size (applicable when the 'noise_type' is 'correlated_gaussian')")

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num_epochs", default=30, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid_interval", default=1, type=int, help="evaluate every N epochs")
    parser.add_argument("--save_interval", default=1, type=int, help="save a checkpoint every N steps")

    parser.add_argument("--num_workers", default=2, type=int, help="num_workers for dataloader")

    # Add model arguments
    parser.add_argument("--model", default="dncnn", type=str, help="dncnn | unet | custom")
    parser.add_argument("--training_function", default="ready-made", type=str, help="ready-made | custom")

    args, _ = parser.parse_known_args()
    models.MODEL_REGISTRY[args.model].add_args(parser)
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    """ 
    IMPORTANT:
    PRIOR APPLYING THIS SCRIPT WE RECOMMEND PRETRAINING THE NETWORK FOR BLIND GAUSSIAN DENOISISNG.

    - This script trains a given denoising architecture with synthetic noise.
    - The script reads 'train_<some string>.h5' and 'cov_mean_<some string>.pt' files 
      prepared by the './data/preprocess_synthetic_noise.py' script. These files are located 
      in './h5_files' and './cov_mean_files' folders respectively.
    - The script can be used for training a custom architecture. To training a custom architecture, the user 
      must implement the __init__ and forward(self, x) functions in the './models/custom_model.py' file, and 
      load_model_custom(opt, device, model_path) function in './models/model_utils.py' file.
    - This script can train models using user defined (custom) training function. To do this, the user must 
      implement custom_training_func(opt, load_model) function.


    Arguments:
    --dataset_name (string): Name of the dataset. 
                             This argument must match the corresponding argument of './data/preprocess_synthetic_noise.py' script.
    --dataset_dir (string): Path to a folder containing clean video sequences (or bursts of images).
                            This argument must match the corresponding argument of './data/preprocess_synthetic_noise.py' script.
    --batch_size (int): Batch size for training.
    --reduce_dependency (int): When equal to '1' applies dependency reduction method described in the paper. 
                               We recommend setting this oparameter to '1'.
    --noise_type (string): Noise type. For correlated Gaussian noise set to 'correlated_gaussian' and 
                           configure 'sigma" and 'noise_kernel_size' parameters. For any other noise model 
                           set to 'custom' and implement __init__ and __call__ functions of the 'AddCustomNoise' class.
                           This argument must match the corresponding argument of './data/preprocess_synthetic_noise.py' script.
    --sigma (float): The standard deviation of the input noise (applicable only when 
                     the 'noise_type' is 'correlated_gaussian').
                     This argument must match the corresponding argument of './data/preprocess_synthetic_noise.py' script.
    --noise_kernel_size (int): the size of the rectangular kernel used to convolve the noise. 
                               (applicable only when the 'noise_type' is 'correlated_gaussian').
                               This argument must match the corresponding argument of './data/preprocess_synthetic_noise.py' script.
    --lr (float): Initial learning rate.
    --num_epochs (int): Number of epochs for training.
    --model (string): Training architecture. Currently supported models are: 'dncnn', 'unet', and 'custom'. For training custom architecture user 
                      must set this parameter to 'custom' and implement functions __init__ and forward(self, x) in './models/custom_model.py' file, and 
                      load_model_custom(opt, device, model_path) function in './models/model_utils.py' file.
    --training_function (string): Training function. When set to 'ready-made' uses training_func(opt, load_model) function (implemented in the current file) for training. 
                                  When set to 'custom' uses user defined (custom) training function. 
                                  For training using user defined function user must implement custom_training_func(opt, load_model) function.


    Outputs:
    The script outcome is '/experiments/<model_name>/learned/<dataset_name>/state_params_<some string>.pt file.
    
    Example usage:
    python train_for_synthetic_noise.py --dataset_name DAVIS --dataset_dir "./data_set/DAVIS" --noise_type correlated_gaussian --sigma 20.0 --noise_kernel_size 3 --model dncnn

    """

    opt = get_args()
    main(opt)
