import torch
import torch.nn.functional as nn_func
from torch.nn.functional import unfold
from torch.nn.functional import fold
import numpy as np
import matplotlib.pyplot as plt


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0: endw - win + 0 +
                1: stride, 0: endh - win + 0 + 1: stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i: endw - win + i +
                        1: stride, j: endh - win + j + 1: stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def get_cov_mean(opt, stat_dict):
    cov_n, cov_bins, _ = plt.hist(stat_dict['pc_n2n_n_cov_arr'], bins=opt.bins, density=True)
    cov_val = (cov_bins[1:] + cov_bins[:-1]) / 2
    peak_ind = cov_n.argmax()
    peak_val = cov_val[peak_ind]
    cov_val_center = cov_val - peak_val

    right_sum = np.abs((cov_n[peak_ind + 1:] * cov_val_center[peak_ind + 1:])).sum()
    left_cumsum = np.flip(np.flip(np.abs(cov_n[:peak_ind] * cov_val_center[:peak_ind])).cumsum())
    min_cov = cov_val[:peak_ind][left_cumsum > right_sum][-1]
    cov_ind = stat_dict['pc_n2n_n_cov_arr'] > min_cov

    pc_n2n_mean_arr2 = stat_dict['pc_n2n_mean_arr'][cov_ind]

    max_mean = pc_n2n_mean_arr2.mean() + opt.std_num * pc_n2n_mean_arr2.std()
    min_mean = pc_n2n_mean_arr2.mean() - opt.std_num * pc_n2n_mean_arr2.std()
    mean_ind = (min_mean < pc_n2n_mean_arr2) * (pc_n2n_mean_arr2 < max_mean)

    pc_n2n_mean_arr3 = pc_n2n_mean_arr2[mean_ind]
    mean_val = pc_n2n_mean_arr3.mean()

    stat_dict['min_cov'] = min_cov
    stat_dict['min_mean'] = min_mean
    stat_dict['max_mean'] = max_mean
    stat_dict['mean_val'] = mean_val

    cov_mean_dict = {'min_cov': min_cov, 'min_mean': min_mean,
                     'max_mean': max_mean, 'mean_val': mean_val}

    return cov_mean_dict


def get_noise_stat(tile_size, im_n, im_pc, ind_ratio, stat_dict):
    im_n = im_n.unsqueeze(0)
    im_pc = im_pc.unsqueeze(0)
    pc_n2n = im_pc - im_n

    mean_k = torch.ones(1, 3, tile_size, tile_size) / (tile_size ** 2 * 3)
    mean_k = mean_k.to(im_n.device)

    pc_n2n_mean_tmp = torch.nn.functional.conv2d(pc_n2n, mean_k)
    im_n_mean_tmp = torch.nn.functional.conv2d(im_n, mean_k)
    pc_n2n_n_mean_tmp = torch.nn.functional.conv2d(pc_n2n * im_n, mean_k).squeeze(0).squeeze(0)
    pc_n2n_n_cov_tmp = (pc_n2n_n_mean_tmp - pc_n2n_mean_tmp * im_n_mean_tmp).squeeze(0).squeeze(0)

    ind_num = pc_n2n_mean_tmp.numel()
    pc_ind = np.random.choice(ind_num, ind_num // ind_ratio, False)
    stat_dict['pc_n2n_mean_arr'] += pc_n2n_mean_tmp.flatten()[pc_ind].tolist()
    stat_dict['pc_n2n_n_cov_arr'] += pc_n2n_n_cov_tmp.flatten()[pc_ind].tolist()


def save_patches(opt, patch_params, n2n_image, offs_i):
    patches = Im2Patch(
        n2n_image, win=opt.tile_size, stride=opt.stride)
    patch_params['arr_s'] = patch_params['arr_e']
    arr_e_tmp = int(
        np.ceil(patch_params['patches_per_offs'] * (offs_i + 1)))
    patch_params['arr_e'] = min(arr_e_tmp, patch_params['patch_num'])
    for arr_i in range(patch_params['arr_s'], patch_params['arr_e']):
        patch_i = patch_params['patch_i_arr'][arr_i]
        data = patches[:, :, :, patch_i].copy()
        patch_params['h5f_train'].create_dataset(
            str(patch_params['train_size']), data=data)
        patch_params['train_size'] += 1
        for m in range(opt.aug_times - 1):
            data_aug = data_augmentation(
                data, np.random.randint(1, 8))
            patch_params['h5f_train'].create_dataset(
                str(patch_params['train_size']) + "_aug_%d" % (m + 1), data=data_aug)
            patch_params['train_size'] += 1


def replication_pad_3d(in_seq, pad):
    pad = list(x for x in reversed(pad)
               for _ in range(2))  # fixing conv / pad inconsistency bug

    return nn_func.pad(in_seq, pad, 'replicate')


def reflection_pad_vh_3d(in_seq, pad):
    mode = 'reflect' if pad[0] > 1 or pad[1] > 1 else 'replicate'
    pad = list(x for x in reversed(pad)
               for _ in range(2))  # fixing conv / pad inconsistency bug

    b, c, t, v, h = in_seq.shape
    in_seq = in_seq.transpose(1, 2).reshape(b * t, c, v, h)
    out_seq = nn_func.pad(in_seq, pad, mode).\
        view(b, t, c, v + pad[0] + pad[1], h + pad[2] + pad[3]).transpose(1, 2)

    return out_seq


def reflection_pad_3d(in_seq, pad):
    if pad[0] <= 1 and pad[1] <= 1 and pad[2] <= 1:
        return replication_pad_3d(in_seq, pad)

    else:
        if pad[1] != 0 or pad[2] != 0:
            out_seq = reflection_pad_vh_3d(in_seq, pad[1:3])
        else:
            out_seq = in_seq

        return out_seq


def pad_frames(im_n_pad, reflect_pad, const_pad):
    im_n_pad = im_n_pad.unsqueeze(0)
    im_n_pad = reflection_pad_3d(im_n_pad, reflect_pad)
    im_n_pad = nn_func.pad(im_n_pad, const_pad, mode='constant', value=-1)
    im_n_pad = im_n_pad.squeeze(0)
    return im_n_pad


def swap_frames(opt, fr_i, seq_n):
    im_offs = opt.frame_num // 2
    im_n_swap = seq_n.clone()
    im_n_swap[:, im_offs, ...] = seq_n[:, fr_i, ...]
    im_n_swap[:, fr_i, ...] = seq_n[:, im_offs, ...]

    return im_n_swap



