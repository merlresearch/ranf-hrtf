# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
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
    parser.add_argument("test_subjects", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_78", action="store_true")
    parser.add_argument("--calibrate_itdoffset", action="store_true")
    args = parser.parse_args()

    conf_path = Path(args.conf_path)
    config = OmegaConf.load(conf_path.joinpath("original_config.yaml"))
    config.dataset.upsample = args.upsample

    assert args.valid_size % len(args.test_subjects.split(",")) == 0, "This is assumed for ease of implementation"

    dump_path = Path(args.dump_path)
    config.dataset.retrieval = str(dump_path.joinpath("lsd_itdd_mats.npz"))

    if args.calibrate_itdoffset:
        config.dataset.features = str(dump_path.joinpath("features_and_locs_with_azimuth_calibration.npz"))
    else:
        config.dataset.features = str(dump_path.joinpath("features_and_locs_wo_azimuth_calibration.npz"))

    config.seed = args.seed

    # The given indices of `test_subjects` are assumed to start from 1.
    test_subjetcs = [int(x) - 1 for x in args.test_subjects.split(",")]
    config.dataset.test_subjects = test_subjetcs

    # The subjects whose HRTFs are similar to those of the test subjects are assigned to the training set.
    npz = np.load(Path(args.dump_path).joinpath("lsd_itdd_mats.npz"))
    itdd_mat, lsd_mat = npz["itdd_mat"], npz["lsd_mat"]
    special_subjects_train, special_subjects_valid = set(), set()
    for idx, sidx in enumerate(test_subjetcs):
        sidxs = []
        for itdd in sorted(list(set(itdd_mat[sidx, :]))):
            _sidxs = np.where(itdd_mat[sidx, :] == itdd)[0]
            order = np.argsort(lsd_mat[sidx, _sidxs])
            sidxs += _sidxs[order].tolist()

        for _sidx in sidxs:
            if _sidx in special_subjects_train or _sidx in special_subjects_valid:
                continue

            if len(special_subjects_train) < (idx + 1) * 5:
                special_subjects_train.add(_sidx)
            else:
                special_subjects_valid.add(_sidx)

            if len(special_subjects_valid) == (idx + 1) * (args.valid_size // len(test_subjetcs)):
                break

    config.dataset.valid_subjects = sorted(list(special_subjects_valid))
    assert len(config.dataset.valid_subjects) == args.valid_size

    train_subjects = set(range(200)) - special_subjects_valid
    if args.skip_78:
        train_subjects.discard(78)

    config.dataset.train_subjects = sorted(list(train_subjects))

    if hasattr(config, "model"):
        if "RANF" in config.model.name:
            if hasattr(config.dataset, "signal"):
                # The following lines are for RANF+.
                signal_attrs = dir(config.dataset.signal)
                assert "concat_tgt_signal" in signal_attrs
                num_signal_processing_methdos = sum(1 for attr in signal_attrs if attr.endswith("_path"))
                num_signal_processing_methdos *= 2 if config.dataset.signal.concat_tgt_signal else 1
                config.model.config.conv_in = 2 * (1 + num_signal_processing_methdos)
                config.model.config.emb_in = 3 + num_signal_processing_methdos

            else:
                # The following (2, 3) are the default values for RANF.
                config.model.config.conv_in = 2
                config.model.config.emb_in = 3

            if args.calibrate_itdoffset:
                config.dataset.azimuth_calibration = True
                config.model.config.azimuth_calibration = True

    exp_path = Path(args.exp_path)
    exp_path.mkdir(parents=True, exist_ok=True)
    with open(exp_path.joinpath("config.yaml"), "w") as f:
        OmegaConf.save(config=config, f=f)


if __name__ == "__main__":
    main()
