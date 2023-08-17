import os
import argparse
import collections

if __name__ == "__main__":

    sigma_list = [5, 10, 15, 20]
    k_list = [2, 3, 4]
    patch_size_dict = collections.defaultdict(dict)
    patch_size_dict[5][2] = 19
    patch_size_dict[5][3] = 19
    patch_size_dict[5][4] = 25
    patch_size_dict[10][2] = 27
    patch_size_dict[10][3] = 37
    patch_size_dict[10][4] = 43
    patch_size_dict[15][2] = 31
    patch_size_dict[15][3] = 41
    patch_size_dict[15][4] = 43
    patch_size_dict[20][2] = 33
    patch_size_dict[20][3] = 43
    patch_size_dict[20][4] = 45

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sigma", default=20.0, type=float, help="gaussian noise sigma (applicable when the 'noise_type' is 'correlated_gaussian')")
    parser.add_argument("--k", default=3, type=int, help="noise kernel size (applicable when the 'noise_type' is 'correlated_gaussian')")
    parser.add_argument("--model", default="dncnn", type=str, help="dncnn | unet | custom")
    parser.add_argument("--skip_preprocessing", action='store_true', help="skips the preprocessing stage")
    opt = parser.parse_args()

    if not opt.skip_preprocessing:
        os.system("python ./data/preprocess_synthetic_noise.py --sigma {} --noise_kernel_size {} --patch_size {} --davis_use_same_frames".format(\
            opt.sigma, opt.k, patch_size_dict[opt.sigma][opt.k]))
        print("\n")
    os.system("python train_for_synthetic_noise.py --sigma {} --noise_kernel_size {} --model {}".format(\
        opt.sigma, opt.k, opt.model))
    print("\n")
    os.system("python test_with_synthetic_noise.py --sigma {} --noise_kernel_size {} --model {}".format(\
        opt.sigma, opt.k, opt.model))
