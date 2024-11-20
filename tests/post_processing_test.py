# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from scipy.fft import fft
from spatialaudiometrics import hrtf_metrics as hf

from ranf.utils.reconstruction import hrtf2hrir_minph


@pytest.mark.parametrize("n_directions", [10, 20])
@pytest.mark.parametrize("f_nyquist", [129])
@pytest.mark.parametrize("itd_search_width", [0, 1, 5, 10])
def test_hrtf2hrir_minph(
    n_directions,
    f_nyquist,
    itd_search_width,
):
    mag = np.random.rand(n_directions, 2, f_nyquist)
    itd = np.round(np.random.rand(n_directions, 1) * 90 - 45)
    hrir = hrtf2hrir_minph(mag, itd, itd_search_width=itd_search_width)

    _mag = np.abs(fft(hrir, axis=-1))[..., : hrir.shape[-1] // 2 + 1]
    np.testing.assert_allclose(_mag, mag)

    if itd_search_width > 1:
        mag = np.ones((n_directions, 2, f_nyquist))
        itd = np.round(np.random.rand(n_directions, 1) * 90 - 45)
        hrir = hrtf2hrir_minph(mag, itd, itd_search_width=itd_search_width)

        _, _itd, _ = hf.itd_estimator_maxiacce(hrir, fs=np.array(48000))
        np.testing.assert_allclose(_itd, itd[:, 0], atol=1.0 + 1.0e-5)
