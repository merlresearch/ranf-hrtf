# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from ranf.utils.config import LSDFREQIDX


def lsd_loss(target, pred, dim=-1, use_index=False):
    if use_index:
        freqidx = torch.tensor(LSDFREQIDX, device=target.device)
    else:
        freqidx = torch.arange(target.shape[-1], device=target.device)

    mse = torch.mean(torch.square(target - pred)[:, :, freqidx], dim=dim)
    rmse = torch.sqrt(mse)
    return torch.mean(rmse, -1)


def itd_diff_loss(target, pred, sr=48000, threshold=0.0):
    error = torch.abs(target - pred)
    retval = torch.clamp(error, min=threshold)
    return (1.0e6 / sr) * retval


def ild_diff_loss(target, pred, target_ild=None):
    if target_ild is None:
        target_ild = compute_ild(target)

    return torch.abs(target_ild - compute_ild(pred))


def compute_ild(hrtf):
    hrtf = torch.cat([hrtf, hrtf[..., 1:-1]], dim=-1)
    rms = torch.linalg.norm(hrtf, dim=-1)
    logrms = 20 * torch.log10(rms)
    ild = logrms[:, 0] - logrms[:, 1]
    return ild
