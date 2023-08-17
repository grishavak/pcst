import os
import argparse
import collections

if __name__ == "__main__":

    iso_list = [1600, 3200, 6400, 12800, 25600]
    patch_size_dict = dict()
    patch_size_dict[1600] = 15
    patch_size_dict[3200] = 25
    patch_size_dict[6400] = 27
    patch_size_dict[12800] = 35
    patch_size_dict[25600] = 37

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--iso", default=25600, type=int, help="ISO")
    parser.add_argument("--model", default="unet", type=str, help="dncnn | unet | custom")
    parser.add_argument("--skip_preprocessing", action='store_true', help="skips the preprocessing stage")
    opt = parser.parse_args()

    dataset_name = "CRVD_ISO{}".format(opt.iso)
    dataset_dir = "./data_set/CRVD/CRVD_ISO{}".format(opt.iso)


    if not opt.skip_preprocessing:
        os.system("python ./data/preprocess_unknown_noise.py --dataset_name {} --dataset_dir {} --patch_size {} --replicate_dataset 1".format(\
            dataset_name, dataset_dir, patch_size_dict[opt.iso]))
        print("\n")
    os.system("python train_for_unknown_noise.py --dataset_name {} --dataset_dir {} --model {}".format(\
        dataset_name, dataset_dir, opt.model))
    print("\n")
    os.system("python test_with_unknown_noise_training_sequences.py --dataset_name {} --dataset_dir {} --model {}".format(\
        dataset_name, dataset_dir, opt.model))
