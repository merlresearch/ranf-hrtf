# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import fft
from spatialaudiometrics import hrtf_metrics as hf
from spatialaudiometrics import load_data as ld


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def linear2db(spec, eps=1.0e-5):
    return 20 * torch.log10(spec + eps)


def db2linear(spec):
    return 10.0 ** (spec / 20.0)


def plot_hrtf(fname, target, pred, lsd):
    target = target.detach().to("cpu")
    pred = pred.detach().to("cpu")

    plt.subplot(1, 2, 1)
    plt.plot(target[0, 0, :])
    plt.plot(pred[0, 0, :])

    plt.subplot(1, 2, 2)
    plt.plot(target[0, 1, :])
    plt.plot(pred[0, 1, :])

    plt.title(f"LSD: {lsd} dB")
    plt.savefig(fname, format="png", dpi=300)
    plt.clf()
    plt.close()


def extract_features(path):
    hrtf = ld.HRTF(path)
    spec = np.abs(fft(hrtf.hrir, axis=-1))[..., : hrtf.hrir.shape[-1] // 2 + 1]
    ild = hf.ild_estimator_rms(hrtf.hrir)
    _, itd, _ = hf.itd_estimator_maxiacce(hrtf.hrir, hrtf.fs)
    loc = hrtf.locs
    return spec, ild, itd, loc


def count_parameters(model):
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param.numel())
    return sum(params)


def to_cartesian(x):
    if x.ndim == 1:
        x = x[None, :]
        ndim = 1
    else:
        ndim = 2

    y = np.stack([np.cos(x[:, 0]) * np.cos(x[:, 1]), np.sin(x[:, 0]) * np.cos(x[:, 1]), np.sin(x[:, 1])], -1)
    y *= x[:, 2, None]

    if ndim == 1:
        return y[0, :]
    else:
        return y
