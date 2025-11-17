# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch

from ranf.utils.loss_functions import ild_diff_loss, itd_diff_loss, lsd_loss


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("f_nyquist", [129])
@pytest.mark.parametrize("use_index", [False, True])
def test_lsd_loss(batch_size, f_nyquist, use_index):
    ground_truth = torch.randn(batch_size, 2, f_nyquist)
    loss = torch.mean(lsd_loss(ground_truth, ground_truth, use_index=use_index))
    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=1.0e-5, atol=1.0e-6)

    loss = torch.mean(lsd_loss(ground_truth, ground_truth + 1.0, use_index=use_index))
    torch.testing.assert_close(loss, torch.tensor(1.0), rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_itd_diff_loss(batch_size):
    ground_truth = torch.randn(batch_size)
    loss = torch.mean(itd_diff_loss(ground_truth, ground_truth))
    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=1.0e-5, atol=1.0e-6)

    loss = torch.mean(itd_diff_loss(ground_truth, ground_truth + 1.0))
    torch.testing.assert_close(loss, torch.tensor(1.0e6 / 48000), rtol=1.0e-5, atol=1.0e-6)

    loss = torch.mean(itd_diff_loss(ground_truth, ground_truth, threshold=1.0))
    torch.testing.assert_close(loss, torch.tensor(1.0e6 / 48000), rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("f_nyquist", [129])
@pytest.mark.parametrize("target_ild", [None, 0.0])
def test_ild_diff_loss(batch_size, f_nyquist, target_ild):
    ground_truth = torch.randn(batch_size, 2, f_nyquist)

    if target_ild is None:
        loss = torch.mean(ild_diff_loss(ground_truth, ground_truth, target_ild))
        torch.testing.assert_close(loss, torch.tensor(0.0), rtol=1.0e-5, atol=1.0e-6)

    else:
        loss = torch.mean(ild_diff_loss(ground_truth, ground_truth, target_ild))
        loss > 0.0
