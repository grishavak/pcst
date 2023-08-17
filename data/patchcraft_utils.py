import torch
import torch.nn.functional as nn_func
from torch.nn.functional import unfold
from torch.nn.functional import fold
import numpy as np


def find_nn(opt, seq_pad):
    t_offs = (seq_pad.shape[1] - 1) // 2
    t_s = t_offs
    t_e = -t_offs if t_offs != 0 else None
    v_s = opt.search_offs
    v_e = -opt.search_offs if opt.search_offs != 0 else None
    h_s = opt.search_offs
    h_e = -opt.search_offs if opt.search_offs != 0 else None
    seq_nn = seq_pad[:, t_s:t_e, v_s:v_e, h_s:h_e]
    vh_tot_pad = opt.patch_size - 1 + opt.search_offs
    w_patch = seq_pad.shape[-1] - 2*vh_tot_pad + opt.patch_size - 1
    h_patch = seq_pad.shape[-2] - 2*vh_tot_pad + opt.patch_size - 1
    min_d = torch.full((h_patch, w_patch),
                       float('inf'), dtype=seq_pad.dtype, device=seq_pad.device)
    min_i = torch.full((h_patch, w_patch, 3), -(seq_nn.numel() + 1),
                       dtype=torch.long, device=seq_pad.device)
    i_arange_patch = torch.arange(np.array(min_d.shape).prod(), dtype=torch.long,
                                  device=seq_pad.device).view(min_d.shape)

    for i_t in range(-t_offs, t_offs + 1):
        if i_t != 0:
            t_s = t_offs + i_t
            t_e = -(t_offs - i_t) if t_offs - i_t != 0 else None
            for i_v in range(-opt.search_offs, opt.search_offs + 1, opt.search_skip):
                v_s = opt.search_offs + i_v
                v_e = -(opt.search_offs - i_v) if opt.search_offs - i_v != 0 else None
                for i_h in range(-opt.search_offs, opt.search_offs + 1, opt.search_skip):
                    h_s = opt.search_offs + i_h
                    h_e = -(opt.search_offs - i_h) if opt.search_offs - \
                        i_h != 0 else None

                    seq_d = ((seq_pad[..., t_s:t_e, v_s:v_e, h_s:h_e] - seq_nn) ** 2).\
                        mean(dim=(0, 1), keepdim=False)

                    seq_d = torch.cumsum(seq_d, dim=-1)
                    tmp = seq_d[..., 0:-opt.patch_size]
                    seq_d = seq_d[..., (opt.patch_size - 1):]
                    seq_d[..., 1:] = seq_d[..., 1:] - tmp

                    seq_d = torch.cumsum(seq_d, dim=-2)
                    tmp = seq_d[..., 0:-opt.patch_size, :]
                    seq_d = seq_d[..., (opt.patch_size - 1):, :]
                    seq_d[..., 1:, :] = seq_d[..., 1:, :] - tmp

                    i_change = seq_d < min_d
                    min_d.flatten()[i_arange_patch[i_change]] = seq_d.flatten()[
                        i_arange_patch[i_change]]
                    min_i[..., 0].flatten()[i_arange_patch[i_change]] = i_t
                    min_i[..., 1].flatten()[i_arange_patch[i_change]] = i_v
                    min_i[..., 2].flatten()[i_arange_patch[i_change]] = i_h

    if opt.search_offs2 is not None and opt.search_skip > 1:
        for i_t in range(-t_offs, t_offs + 1):
            if i_t != 0:
                t_s = t_offs + i_t
                t_e = -(t_offs - i_t) if t_offs - i_t != 0 else None
                for i_v in range(-opt.search_offs2, opt.search_offs2 + 1):
                    v_s = opt.search_offs + i_v
                    v_e = -(opt.search_offs - i_v) if opt.search_offs - \
                        i_v != 0 else None
                    for i_h in range(-opt.search_offs2, opt.search_offs2 + 1):
                        h_s = opt.search_offs + i_h
                        h_e = -(opt.search_offs - i_h) if opt.search_offs - \
                            i_h != 0 else None

                        seq_d = ((seq_pad[..., t_s:t_e, v_s:v_e, h_s:h_e] - seq_nn) ** 2).\
                            mean(dim=(0, 1), keepdim=False)

                        seq_d = torch.cumsum(seq_d, dim=-1)
                        tmp = seq_d[..., 0:-opt.patch_size]
                        seq_d = seq_d[..., (opt.patch_size - 1):]
                        seq_d[..., 1:] = seq_d[..., 1:] - tmp

                        seq_d = torch.cumsum(seq_d, dim=-2)
                        tmp = seq_d[..., 0:-opt.patch_size, :]
                        seq_d = seq_d[..., (opt.patch_size - 1):, :]
                        seq_d[..., 1:, :] = seq_d[..., 1:, :] - tmp

                        i_change = seq_d < min_d
                        min_d.flatten()[i_arange_patch[i_change]] = seq_d.flatten()[
                            i_arange_patch[i_change]]
                        min_i[..., 0].flatten()[i_arange_patch[i_change]] = i_t
                        min_i[..., 1].flatten()[i_arange_patch[i_change]] = i_v
                        min_i[..., 2].flatten()[i_arange_patch[i_change]] = i_h

    return min_i


def create_im(opt, seq_pad, min_i, f_i):
    pad_patch = opt.patch_size - 1

    c_ch, fr_pad01, h_pix_pad11, w_pix_pad11 = seq_pad.shape

    t_offs = (seq_pad.shape[1] - 1) // 2
    h_pix_pad10 = h_pix_pad11 - 2 * opt.search_offs
    w_pix_pad10 = w_pix_pad11 - 2 * opt.search_offs
    out_im = torch.full((c_ch, h_pix_pad10, w_pix_pad10),
                        float('nan'), dtype=seq_pad.dtype, device=seq_pad.device)
    t_s = t_offs
    t_e = -t_offs if t_offs != 0 else None
    v_s = h_s = opt.search_offs
    v_e = h_e = -opt.search_offs if opt.search_offs != 0 else None

    v_offs = f_i // opt.patch_size
    h_offs = f_i % opt.patch_size
    map_v_s = (pad_patch + v_offs) % opt.patch_size
    size_v_patch = (h_pix_pad10 - map_v_s) // opt.patch_size
    size_v_pix = opt.patch_size * size_v_patch
    map_v_e = map_v_s + size_v_pix
    map_v_e_patch = map_v_s + size_v_pix - pad_patch
    map_h_s = (pad_patch + h_offs) % opt.patch_size
    size_h_patch = (w_pix_pad10 - map_h_s) // opt.patch_size
    size_h_pix = opt.patch_size * size_h_patch
    map_h_e = map_h_s + size_h_pix
    map_h_e_patch = map_h_s + size_h_pix - pad_patch
    min_i_im = min_i[map_v_s:map_v_e_patch:opt.patch_size,
                     map_h_s:map_h_e_patch:opt.patch_size, ...]
    for v_i in range(min_i_im.shape[0]):
        v_out_s = map_v_s + v_i * opt.patch_size
        v_out_e = v_out_s + opt.patch_size
        for h_i in range(min_i_im.shape[1]):
            h_out_s = map_h_s + h_i * opt.patch_size
            h_out_e = h_out_s + opt.patch_size
            p_ind = min_i_im[v_i, h_i, :].tolist()
            t_in = p_ind[0] + t_offs
            v_in_s = p_ind[1] + v_i * opt.patch_size + opt.search_offs + map_v_s
            v_in_e = v_in_s + opt.patch_size
            h_in_s = p_ind[2] + h_i * opt.patch_size + opt.search_offs + map_h_s
            h_in_e = h_in_s + opt.patch_size

            out_im[:, v_out_s:v_out_e, h_out_s:h_out_e] = \
                seq_pad[:, t_in, v_in_s:v_in_e, h_in_s:h_in_e]

    out_im = out_im[..., pad_patch:-pad_patch, pad_patch:-pad_patch]

    return out_im
