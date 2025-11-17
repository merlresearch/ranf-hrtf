# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ranf.utils.config import NUMSONICOMDIRECTIONS
from ranf.utils.util import extract_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--upsample", type=int, default=3)
    parser.add_argument("--calibrate_itdoffset", action="store_true")
    args = parser.parse_args()

    sofa_paths = Path(args.input_path).glob("*.sofa")
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    specs, ilds, itds, locs = [], [], [], []
    for idx, path in enumerate(tqdm(sorted(sofa_paths))):
        spec, ild, itd, loc = extract_features(path)

        assert int(path.stem.split("_")[0][1:]) - 1 == idx, (idx, path)
        assert len(itd) == NUMSONICOMDIRECTIONS, "HRTF should follow the SONICOM spatial grid"

        if args.calibrate_itdoffset:
            # NOTE: The following lines are for the training and validation sets.
            if idx < 200:
                alpha = (itd[414] - itd[4]) / (loc[414, 0] - loc[4, 0])
                beta = itd[4] / alpha
                loc[:, 0] = (loc[:, 0] + beta + 360) % 360

            # NOTE: The available directions depend on the sparsity level on the challenge test set.
            else:
                if args.upsample == 3:
                    alpha = (itd[414] - itd[4]) / (loc[414, 0] - loc[4, 0])
                    beta = itd[4] / alpha
                    loc[:, 0] = (loc[:, 0] + beta + 360) % 360
                elif args.upsample == 5:
                    alpha = (itd[612] - itd[203]) / 90
                    beta = itd[4] / alpha
                    loc[:, 0] = (loc[:, 0] + beta + 360) % 360
                elif args.upsample == 19:
                    alpha = (itd[269] - itd[546]) / 120
                    beta = itd[4] / alpha
                    loc[:, 0] = (loc[:, 0] + beta + 360) % 360
                elif args.upsample == 100:
                    alpha = (itd[436] - itd[788]) / 80
                    beta = (itd[788] - alpha * 5.0) / alpha
                    loc[:, 0] = (loc[:, 0] + beta + 360) % 360
                else:
                    raise ValueError

        specs.append(spec.astype(np.float32))
        ilds.append(ild.astype(np.float32))
        itds.append(np.array(itd))
        locs.append(loc.astype(np.float32))

    if args.calibrate_itdoffset:
        output_path = output_path.joinpath("features_and_locs_with_azimuth_calibration.npz")
    else:
        output_path = output_path.joinpath("features_and_locs_wo_azimuth_calibration.npz")

    np.savez(
        output_path,
        specs=np.array(specs),
        ilds=np.array(ilds),
        itds=np.array(itds),
        locs=np.array(locs),
    )


if __name__ == "__main__":
    main()
