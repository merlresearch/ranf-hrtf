# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_path", type=str)
    parser.add_argument("conf_path", type=str)
    parser.add_argument("exp_path", type=str)
    parser.add_argument("sonicom_path", type=str)
    parser.add_argument("upsample", type=int)
    parser.add_argument("valid_size", type=int)
    parser.add_argument("test_size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_78", action="store_true")
    parser.add_argument("--calibrate_itdoffset", action="store_true")
    args = parser.parse_args()

    conf_path = Path(args.conf_path)
    config = OmegaConf.load(conf_path.joinpath("original_config.yaml"))
    config.dataset.upsample = args.upsample

    config.dataset.retrieval = str(Path(args.dump_path).joinpath("lsd_itdd_mats.npz"))

    npz_path = Path(args.sonicom_path).joinpath("npzs")
    if args.calibrate_itdoffset:
        config.dataset.features = str(npz_path.joinpath("features_and_locs_with_azimuth_calibration.npz"))
    else:
        config.dataset.features = str(npz_path.joinpath("features_and_locs_wo_azimuth_calibration.npz"))

    npz = np.load(config.dataset.features)
    nsubjects = npz["itds"].shape[0]
    assert nsubjects == 200, "The number of subjects in the SONICOM dataset should be 200."

    config.seed = 0
    train_size = nsubjects - (args.valid_size + args.test_size)
    train_subjetcs = list(range(train_size))
    if args.skip_78:
        train_subjetcs.remove(78)
    config.dataset.train_subjects = train_subjetcs

    train_valid_size = train_size + args.valid_size
    valid_subjects = list(range(train_size, train_valid_size))
    config.dataset.valid_subjects = valid_subjects

    test_subjetcs = list(range(train_valid_size, nsubjects))
    config.dataset.test_subjects = test_subjetcs
    assert len(test_subjetcs) == args.test_size

    exp_path = Path(args.exp_path)
    exp_path.mkdir(parents=True, exist_ok=True)
    with open(exp_path.joinpath("config.yaml"), "w") as f:
        OmegaConf.save(config=config, f=f)


if __name__ == "__main__":
    main()
