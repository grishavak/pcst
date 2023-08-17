import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sigma", default=20.0, type=float, help="gaussian noise sigma (applicable when the 'noise_type' is 'correlated_gaussian')")
    parser.add_argument("--k", default=3, type=int, help="noise kernel size (applicable when the 'noise_type' is 'correlated_gaussian')")
    parser.add_argument("--model", default="dncnn", type=str, help="dncnn | unet | custom")
    opt = parser.parse_args()

    os.system("python test_with_synthetic_noise.py --sigma {} --noise_kernel_size {} --model {}".format(\
        opt.sigma, opt.k, opt.model))
