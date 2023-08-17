import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--iso", default=25600, type=int, help="ISO")
    parser.add_argument("--model", default="dncnn", type=str, help="dncnn | unet | custom")
    opt = parser.parse_args()

    dataset_name = "CRVD_ISO{}".format(opt.iso)
    dataset_dir = "./data_set/CRVD/CRVD_ISO{}".format(opt.iso)
    os.system("python test_with_unknown_noise_training_sequences.py --dataset_name {} --dataset_dir {} --model {}".format(\
        dataset_name, dataset_dir, opt.model))
