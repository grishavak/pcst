import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F

from utils import data
import utils, models
from models.model_utils import *

import cv2
import os
from skimage.metrics import structural_similarity

def test_func(opt, load_model, load_state_params):
    log_f_name = './logs/log_test_{}_{}.txt'.format(opt.model, opt.dataset_name)
    log_f = open(log_f_name, "w")

    device = torch.device('cuda') if (torch.cuda.is_available() and opt.use_gpu) else torch.device('cpu')

    torch.manual_seed(opt.seed)
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load model
    pcst_path = 'experiments/{}/learned/{}/state_params.pt'.format(
        opt.model, opt.dataset_name)
    model_pcst = load_state_params(opt, device, pcst_path)
    pretrained_path = './experiments/{}/orig/model.pt'.format(opt.model)
    model_pretrained = load_model(opt, device, pretrained_path)
    print(f"Built {opt.model.upper()} model consisting of {sum(p.numel() for p in model_pcst.parameters()):,} parameters")
    log_f.write(f"Built {opt.model.upper()} model consisting of {sum(p.numel() for p in model_pcst.parameters()):,} parameters\n")
    log_f.flush()

    model_pcst.eval()
    model_pretrained.eval()

    noisy_psnr_arr = list()
    pretrained_psnr_arr = list()
    pcst_psnr_arr = list()

    noisy_ssim_arr = list()
    pretrained_ssim_arr = list()
    pcst_ssim_arr = list()

    # Prepare folder names
    in_dir = os.path.join(opt.dataset_dir, 'test_images')
    in_clean_dir = os.path.join(in_dir, 'clean')
    in_noisy_dir = os.path.join(in_dir, 'noisy')
    out_dir = os.path.join(opt.dataset_dir, 'output_images')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_out_dir = os.path.join(out_dir, opt.model)
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    out_dir = os.path.join(model_out_dir, '{}'.format(opt.dataset_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_clean_dir = os.path.join(out_dir, 'clean')
    if not os.path.exists(out_clean_dir):
        os.makedirs(out_clean_dir)
    out_noisy_dir = os.path.join(out_dir, 'noisy')
    if not os.path.exists(out_noisy_dir):
        os.makedirs(out_noisy_dir)
    pcst_dir = os.path.join(out_dir, 'pcst')
    if not os.path.exists(pcst_dir):
        os.makedirs(pcst_dir)
    pretrained_dir = os.path.join(out_dir, 'pretrained')
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)

    clean_im_names = sorted(os.listdir(in_clean_dir))
    noisy_im_names = sorted(os.listdir(in_noisy_dir))
    missing_im_names = set(clean_im_names).difference(set(noisy_im_names))
    assert_message = ("noisy versions of sequences \"{}\" are missing".format(', '.join(missing_im_names)))
    assert len(missing_im_names) == 0, f"{assert_message}"
    missing_im_names = set(noisy_im_names).difference(set(clean_im_names))
    assert_message = ("clean versions of sequences \"{}\" are missing".format(', '.join(missing_im_names)))
    assert len(missing_im_names) == 0, f"{assert_message}"

    toolbar_width = 40
    print("Testing {}".format((opt.model.upper())))
    print("-" * toolbar_width)

    tl_inc = len(clean_im_names) // toolbar_width + 1
    head_str = "[test]"
    sys.stdout.write("{}[{}]".format(head_str, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))
    char_num = len(head_str) + toolbar_width + 2
    print_cnt = 0

    for im_i in range(len(clean_im_names)):
        clean_path = os.path.join(in_clean_dir, clean_im_names[im_i])
        clean_im = cv2.imread(clean_path, -1)
        clean_im = cv2.cvtColor(clean_im, cv2.COLOR_BGR2RGB)
        clean_im = np.float32(clean_im) / 65535
        clean_im = torch.from_numpy(clean_im).permute(2, 0, 1).unsqueeze(0)
        noisy_path = os.path.join(in_noisy_dir, clean_im_names[im_i])
        noisy_im = cv2.imread(noisy_path, -1)
        noisy_im = cv2.cvtColor(noisy_im, cv2.COLOR_BGR2RGB)
        noisy_im = np.float32(noisy_im) / 65535
        noisy_im = torch.from_numpy(noisy_im).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            denoised_im_pcst = model_pcst(noisy_im.to(device)).cpu().clamp(0, 1)
            test_psnr_pcst = -10 * ((denoised_im_pcst - clean_im) ** 2).mean().log10().item()
            denoised_im_pretrained = model_pretrained(noisy_im.to(device)).cpu().clamp(0, 1)
            test_psnr_pretrained = -10 * ((denoised_im_pretrained - clean_im) ** 2).mean().log10().item()
            test_psnr_noisy = -10 * ((noisy_im - clean_im) ** 2).mean().log10().item()

            test_ssim_pcst = structural_similarity(denoised_im_pcst.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                                                    clean_im.squeeze(0).permute(1, 2, 0).cpu().numpy(), multichannel=True, data_range=1.0)
            test_ssim_pretrained = structural_similarity(denoised_im_pretrained.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                                                    clean_im.squeeze(0).permute(1, 2, 0).cpu().numpy(), multichannel=True, data_range=1.0)
            test_ssim_noisy = structural_similarity(noisy_im.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                                                    clean_im.squeeze(0).permute(1, 2, 0).cpu().numpy(), multichannel=True, data_range=1.0)

            noisy_psnr_arr.append(test_psnr_noisy)
            pretrained_psnr_arr.append(test_psnr_pretrained)
            pcst_psnr_arr.append(test_psnr_pcst)

            noisy_ssim_arr.append(test_ssim_noisy)
            pretrained_ssim_arr.append(test_ssim_pretrained)
            pcst_ssim_arr.append(test_ssim_pcst)

            if opt.save_images:
                reconst_path = os.path.join(reconst_dir, 'im_s{}_f{}.jpg'.format(im_i, fr_i))
                im_dn_write = (denoised_im_pcst.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
                cv2.imwrite(reconst_path, cv2.cvtColor(im_dn_write, cv2.COLOR_RGB2BGR))

                pretrained_path = os.path.join(pretrained_dir, 'im_s{}_f{}.jpg'.format(im_i, fr_i))
                im_pretrained_write = (denoised_im_pretrained.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
                cv2.imwrite(pretrained_path, cv2.cvtColor(im_pretrained_write, cv2.COLOR_RGB2BGR))

        if im_i % tl_inc == 0:
            print_cnt += 1
            sys.stdout.write("-")
            sys.stdout.flush()

    sys.stdout.write("{}]".format("-" * (toolbar_width - print_cnt)))
    sys.stdout.write("\b" * char_num)

    noisy_psnr_arr = np.array(noisy_psnr_arr)
    pretrained_psnr_arr = np.array(pretrained_psnr_arr)
    pcst_psnr_arr = np.array(pcst_psnr_arr)

    noisy_ssim_arr = np.array(noisy_ssim_arr)
    pretrained_ssim_arr = np.array(pretrained_ssim_arr)
    pcst_ssim_arr = np.array(pcst_ssim_arr)

    pretrained_psnr_mean = pretrained_psnr_arr.mean()
    pcst_psnr_mean = pcst_psnr_arr.mean()
    noisy_psnr_mean = noisy_psnr_arr.mean()

    pretrained_ssim_mean = pretrained_ssim_arr.mean()
    pcst_ssim_mean = pcst_ssim_arr.mean()
    noisy_ssim_mean = noisy_ssim_arr.mean()

    if opt.save_psnr_arr:
        psnr_ssim_dict = {'noisy_psnr_arr': noisy_psnr_arr, 'orig_psnr_arr': pretrained_psnr_arr,
                          'new_psnr_arr': pcst_psnr_arr, 'noisy_ssim_arr': noisy_ssim_arr,
                          'orig_ssim_arr': pretrained_ssim_arr, 'new_ssim_arr': pcst_ssim_arr,
                          }
        psnr_ssim_path = os.path.join(data_dir, 'psnr_ssim_arr.pt')
        torch.save(psnr_ssim_dict, psnr_ssim_path)

    print("noisy psnr {:.2f} | noisy ssim {:.3f} | pretrained psnr {:.2f} | pretrained ssim {:.3f} | pcst psnr {:.2f} | pcst ssim {:.3f}".format(
          noisy_psnr_mean, noisy_ssim_mean, pretrained_psnr_mean, pretrained_ssim_mean, pcst_psnr_mean, pcst_ssim_mean))
    log_f.write("noisy psnr {:.2f} | noisy ssim {:.3f} | pretrained psnr {:.2f} | pretrained ssim {:.3f} | pcst psnr {:.2f} | pcst ssim {:.3f}".format(
          noisy_psnr_mean, noisy_ssim_mean, pretrained_psnr_mean, pretrained_ssim_mean, pcst_psnr_mean, pcst_ssim_mean))
    log_f.flush()


def custom_test_func(opt, load_model):

    """ Add your code here """

    return


def main(opt):
    if opt.model == 'dncnn':
        load_state_params = load_state_params_dncnn
        load_model = load_model_dncnn
    elif opt.model == 'unet':
        load_state_params = load_state_params_unet
        load_model = load_model_unet
    elif opt.model == 'custom':
        load_state_params = load_state_params_custom
        load_model_custom = load_model_custom
    else:
        assert_message = ("model = {} is undefined. ".format(opt.model))
        assert True, f"{assert_message}"

    if opt.test_function == 'ready-made':
        test_model = test_func
    elif opt.test_function == 'custom':
        test_model = custom_test_func
    else:
        assert_message = ("test_function = {} is undefined. ".format(opt.test_function))
        assert True, f"{assert_message}"

    test_model(opt, load_model, load_state_params)


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--dataset_name", default='CRVD_ISO25600', type=str, help="dataset name")
    parser.add_argument("--dataset_dir", default="./data_set/CRVD/CRVD_ISO25600", help="path to data directory")

    parser.add_argument("--save_images", default=0, type=int, help="save noisy and reconstructed images")
    parser.add_argument("--save_psnr_arr", default=0, type=int, help="save PSNR and SSIM arrays")

    # Add model arguments
    parser.add_argument("--model", default="dncnn", type=str, help="dncnn | unet | custom")
    parser.add_argument("--test_function", default="ready-made", type=str, help="ready-made | custom")

    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--use_gpu", default=1, type=int, help="0 - use CPU, 1 - use GPU")

    opt, _ = parser.parse_known_args()
    models.MODEL_REGISTRY[opt.model].add_args(parser)
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    """ 
    - This script evaluates a given denoising architecture with noise with unknown model.
    - This script can evaluate models using user defined (custom) test function. To do this, the user must 
      implement custom_test_func(opt, load_model) function.


    Arguments:
    --dataset_name (string): Name of the dataset. 
                             This argument must match the corresponding argument of 'train_for_synthetic_noise.py' script.
    --dataset_dir (string): Path to a folder containing clean video sequences (or bursts of images).
                            This argument must match the corresponding argument of '.train_for_synthetic_noise.py' script.
    --save_images (int): When set to '1', the script saves clean, noisy and reconstructed images.
    --save_psnr_arr (int): When set to '1', the script saves arrays with PSNR and SSIM values.
    --model (string): Tested architecture. Currently supported models are: 'dncnn', 'unet', and 'custom'. For evaluating custom architecture user 
                      must set this parameter to 'custom' and implement load_state_params_custom(opt, device, model_path) function in './models/model_utils.py' file.
    --test_function (string): Test function. When set to 'ready-made' uses training_func(opt, load_model) function (implemented in the current file) for training. 
                              When set to 'custom' uses user defined (custom) training function. 
                              For training using user defined function user must implement custom_training_func(opt, load_model) function.


    Outputs:
    The script outcome is: 
    - clean images
    - noisy images
    - denoised images
    - psnr_ssim_arr.pt file that contain PSNR values.
    
    
    Example usage:
    python test_with_unknown_noise.py --dataset_name CRVD_ISO25600 --dataset_dir "./data_set/CRVD_ISO25600" --model dncnn 

    """
    opt = get_args()
    main(opt)

