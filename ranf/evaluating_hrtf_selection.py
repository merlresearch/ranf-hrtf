# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import sofar as sf
from omegaconf import OmegaConf
from spatialaudiometrics import lap_challenge as lap
from spatialaudiometrics import load_data as ld

from ranf.utils.config import NUMSONICOMDIRECTIONS, TGTDIDXS003, TGTDIDXS005, TGTDIDXS019, TGTDIDXS100
from ranf.utils.util import seed_everything


def load_hrtf(fname):
    hrtf = ld.HRTF(fname)
    return hrtf.hrir, hrtf.locs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--remove_hrtf_files", action="store_true")
    args = parser.parse_args()

    path = Path(args.config_path)
    config = OmegaConf.load(path.joinpath("config.yaml"))

    seed_everything(config.seed)

    if config.dataset.upsample == 3:
        seen_didxs = TGTDIDXS003

    elif config.dataset.upsample == 5:
        seen_didxs = TGTDIDXS005

    elif config.dataset.upsample == 19:
        seen_didxs = TGTDIDXS019

    elif config.dataset.upsample == 100:
        seen_didxs = TGTDIDXS100

    else:
        raise ValueError(f"dataset.upsample should be in (3, 5, 19, 100) but is {config.dataset.upsample}.")

    unseen_didxs = sorted(list(set(range(NUMSONICOMDIRECTIONS)) - set(seen_didxs)))

    eval_path = path.joinpath("log").joinpath("eval")
    eval_path.mkdir(parents=True, exist_ok=True)
    log_name = eval_path.joinpath("eval.log")
    logging.basicConfig(filename=log_name, level=logging.INFO)

    sonicom_path = Path(config.dataset.features).parent.parent.joinpath("sonicom").joinpath("subjects")
    hrtf_type = "FreeFieldCompMinPhase_48kHz"

    npz = np.load(config.dataset.retrieval)
    lsd_mat = npz["lsd_mat"]
    itdd_mat = npz["itdd_mat"]

    for sidx in config.dataset.test_subjects:
        if config.dataset.retrieval_priority == "itdd":
            _sidxs = np.where(itdd_mat[sidx, :] == np.min(itdd_mat[sidx, :]))[0]
            pred_sidx = _sidxs[np.argmin(lsd_mat[sidx, _sidxs])]

        elif config.dataset.retrieval_priority == "lsd":
            pred_sidx = np.argmin(lsd_mat[sidx, :])

        else:
            raise ValueError(f"Invalid retrieval option: {config.dataset.retrieval_priority}")

        prediction = sonicom_path.joinpath(f"P{pred_sidx+1:04}_{hrtf_type}.sofa")
        pred_hrir, _ = load_hrtf(prediction)

        fname = sonicom_path.joinpath(f"P{sidx+1:04}_{hrtf_type}.sofa")

        sofa_file = sf.read_sofa(fname)

        target_path = eval_path.joinpath(f"target_p{sidx+1:04}.sofa")
        sf.write_sofa(target_path, sofa_file)

        pred_path = eval_path.joinpath(f"pred_p{sidx+1:04}.sofa")

        sofa_file.Data_IR[unseen_didxs, ...] = pred_hrir[unseen_didxs, ...].astype(np.float64)
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
