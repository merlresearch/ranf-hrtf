# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ranf.utils.util import extract_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--calibrate_itdoffset", action="store_true")
    args = parser.parse_args()

    sofa_paths = Path(args.input_path).glob("*.sofa")
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    specs, ilds, itds, locs = [], [], [], []
    for idx, path in enumerate(tqdm(sorted(sofa_paths))):
        spec, ild, itd, loc = extract_features(path)

        assert int(path.stem.split("_")[0][1:]) - 1 == idx, (idx, path)
        assert len(itd) == 793, "HRTF should follow the SONICOM spatial grid"

        if args.calibrate_itdoffset:
            # 4 and 414 correspond to the front and left, respectively.
            alpha = (itd[414] - itd[4]) / (loc[414, 0] - loc[4, 0])
            beta = itd[4] / alpha
            loc[:, 0] = (loc[:, 0] + beta + 360) % 360

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
