# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ranf.utils.config import LSDFREQIDX, TGTDIDXS003, TGTDIDXS005, TGTDIDXS019, TGTDIDXS100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_path", type=str)
    parser.add_argument("upsample", type=int)
    parser.add_argument("--skip_78", action="store_true")
    parser.add_argument("--calibrate_itdoffset", action="store_true")
    args = parser.parse_args()

    Path(args.dump_path).mkdir(parents=True, exist_ok=True)

    if args.upsample == 3:
        seen_didxs = TGTDIDXS003

    elif args.upsample == 5:
        seen_didxs = TGTDIDXS005

    elif args.upsample == 19:
        seen_didxs = TGTDIDXS019

    elif args.upsample == 100:
        seen_didxs = TGTDIDXS100
    else:
        raise ValueError(f"given upsample is invalid. It should be in (3, 5, 19, 100) but {args.upsample}.")

    if args.calibrate_itdoffset:
        npz_path = Path(args.dump_path).joinpath("features_and_locs_with_azimuth_calibration.npz")
    else:
        npz_path = Path(args.dump_path).joinpath("features_and_locs_wo_azimuth_calibration.npz")

    npz = np.load(npz_path)
    specs = npz["specs"][:, seen_didxs, :, :]
    itds = npz["itds"][:, seen_didxs]
    nsubjects = itds.shape[0]
    nsubjects_train_valid = 200
    specs = 20 * np.log10(specs[..., LSDFREQIDX] + 1e-15)
    lsd_mat = np.inf * np.ones((nsubjects, nsubjects), dtype=np.float32)
    itdd_mat = np.inf * np.ones((nsubjects, nsubjects), dtype=np.float32)

    for n in tqdm(range(nsubjects_train_valid)):
        if args.skip_78 and n == 78:
            continue

        for m in range(n + 1, nsubjects):
            if args.skip_78 and m == 78:
                continue

            mse = np.mean(np.square(specs[n, ...] - specs[m, ...]), -1)
            lsd = np.mean(np.sqrt(mse))
            lsd_mat[n, m] = lsd
            lsd_mat[m, n] = lsd

            itdd = np.mean(np.abs(itds[n, :] - itds[m, :]))
            itdd_mat[n, m] = itdd
            itdd_mat[m, n] = itdd

    lsd_mat[:, nsubjects_train_valid:] = np.inf
    itdd_mat[:, nsubjects_train_valid:] = np.inf

    np.savez(
        Path(args.dump_path).joinpath("lsd_itdd_mats.npz"),
        lsd_mat=lsd_mat,
        itdd_mat=itdd_mat,
    )


if __name__ == "__main__":
    main()
