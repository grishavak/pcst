import os
import os.path
import numpy as np
import h5py
import torch

import cv2


DATASET_REGISTRY = {}


def build_dataset(name, *args, **kwargs):
    return DATASET_REGISTRY[name](*args, **kwargs)


def register_dataset(name):
    def register_dataset_fn(fn):
        if name in DATASET_REGISTRY:
            raise ValueError(
                "Cannot register duplicate dataset ({})".format(name))
        DATASET_REGISTRY[name] = fn
        return fn

    return register_dataset_fn


@register_dataset("synthetic_noise")
def load_davis(dataset_path, dataset_name, batch_size=100, num_workers=0, \
    noise_type='correlated_gaussian', name_str='noise'):
    dataset_path = os.path.join(dataset_path, "train_{}_{}_{}.h5".format(dataset_name, noise_type, name_str))
    train_dataset = Dataset(filename=dataset_path)
    if not os.path.exists(dataset_path):
        assert_message = ("file \"{}\" does not exist. Please run \"./data/preprocess_synthetic_noise.py\" script to create it.".format(train_dataset))
        assert True, f"{assert_message}"
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return train_loader


@register_dataset("unknown_noise")
def load_dataset_unknown_noise(dataset_path, dataset_name, batch_size=100, num_workers=0):
    dataset_path = os.path.join(dataset_path, "train_{}.h5".format(dataset_name))
    train_dataset = Dataset(filename=dataset_path)
    if not os.path.exists(dataset_path):
        assert_message = ("file \"{}\" does not exist. Please run \"./data/preprocess_unknown_noise.py\" script to create it.".format(train_dataset))
        assert True, f"{assert_message}"
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_loader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super().__init__()
        self.h5f = h5py.File(filename, "r")
        self.keys = list(self.h5f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        data = np.array(self.h5f[key])
        return torch.Tensor(data)
