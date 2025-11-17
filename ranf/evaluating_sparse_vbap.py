# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
import os
from pathlib import Path

import cvxpy as cp
import numpy as np
import sofar as sf
from omegaconf import OmegaConf
from scipy.fft import fft
from spatialaudiometrics import hrtf_metrics as hf
from spatialaudiometrics import lap_challenge as lap
from spatialaudiometrics import load_data as ld

from ranf.utils.reconstruction import hrtf2hrir_minph
from ranf.utils.util import load_seen_unseen_didxs, seed_everything, to_cartesian


def load_hrtf(fname):
    hrtf = ld.HRTF(fname)
    specs = np.abs(fft(hrtf.hrir, axis=-1))[..., : hrtf.hrir.shape[-1] // 2 + 1]
    _, itds, _ = hf.itd_estimator_maxiacce(hrtf.hrir, hrtf.fs)
    itds = np.array(itds)
    locs = np.deg2rad(hrtf.locs) - np.pi
    return hrtf.hrir, specs, itds, locs, hrtf.locs


def sparse_vbap(loc, measured_locs, normalize=True):
    c = np.ones(measured_locs.shape[0])
    b = np.zeros_like(c)
    ref_mat = to_cartesian(measured_locs).T
    loc = to_cartesian(loc)

    gain = cp.Variable(measured_locs.shape[0])
    prob = cp.Problem(cp.Minimize(c.T @ gain), [ref_mat @ gain == loc, gain >= b])
    prob.solve()

    assert gain.value is not None, "No feasible solution"
    gain = np.maximum(gain.value, 0)

    if normalize:
        gain = gain / np.sum(gain)
    return gain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--remove_hrtf_files", action="store_true")
    args = parser.parse_args()

    path = Path(args.config_path)
    config = OmegaConf.load(path.joinpath("config.yaml"))

    seed_everything(config.seed)

    seen_didxs, unseen_didxs = load_seen_unseen_didxs(config.dataset.upsample)

    eval_path = path.joinpath("log").joinpath("eval")
    eval_path.mkdir(parents=True, exist_ok=True)
    log_name = eval_path.joinpath("eval.log")
    logging.basicConfig(filename=log_name, level=logging.INFO)

    sonicom_path = Path(config.dataset.features).parent.parent.joinpath("sonicom").joinpath("subjects")
    hrtf_type = "FreeFieldCompMinPhase_48kHz"
    for sidx in config.dataset.test_subjects:
        fname = sonicom_path.joinpath(f"P{sidx+1:04}_{hrtf_type}.sofa")
        hrir, specs, itds, locs, locs_deg = load_hrtf(fname)

        measured_specs = specs[seen_didxs, ...]
        measured_itds = itds[seen_didxs]
        measured_locs = locs[seen_didxs, :]

        pred_mag, pred_itd = np.zeros_like(specs[unseen_didxs, :]), np.zeros_like(itds[unseen_didxs])
        for idx, didx in enumerate(unseen_didxs):
            gain = sparse_vbap(locs[didx, :], measured_locs, normalize=True)
            pred_mag[idx, ...] = np.sum(gain[:, None, None] * measured_specs, axis=0)

            gain = sparse_vbap(locs[didx, :], measured_locs, normalize=False)
            pred_itd[idx] = np.sum(gain * measured_itds, axis=0)

        sofa_file = sf.read_sofa(fname)
        pred_hrir = hrtf2hrir_minph(pred_mag, itd=np.expand_dims(pred_itd, -1), nfft=sofa_file.Data_IR.shape[-1])

        target_path = eval_path.joinpath(f"target_p{sidx+1:04}.sofa")
        sf.write_sofa(target_path, sofa_file)

        pred_path = eval_path.joinpath(f"pred_p{sidx+1:04}.sofa")

        sofa_file.Data_IR[unseen_didxs, ...] = pred_hrir.astype(np.float64)

        sf.write_sofa(pred_path, sofa_file)

        metrics = lap.calculate_task_two_metrics(str(target_path), str(pred_path))[0]

        logging.info(f"P{sidx+1:04} evaluation")
        logging.info(f"ITD difference (µs): {metrics[0]}")
        logging.info(f"ILD difference (dB): {metrics[1]}")
        logging.info(f"LSD (dB): {metrics[2]}")

        if args.remove_hrtf_files:
            os.remove(target_path)
            os.remove(pred_path)


if __name__ == "__main__":
    main()
