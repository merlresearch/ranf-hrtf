# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import re
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    eval_path = Path(args.config_path).joinpath("log").joinpath("eval")
    with open(eval_path.joinpath("eval.log")) as f:
        lines = f.readlines()

    results = {"ITD": [[], []], "ILD": [[], []], "LSD": [[], []]}
    threhold = {"ITD": 62.5, "ILD": 4.4, "LSD": 7.4}

    for line in lines:
        tmp = line.split(":")[2].split()[0]
        if re.match(r"^P\d{4}$", tmp):
            pidx = int(tmp[1:])

        for key in results.keys():
            if key in line:
                x = float(line.rstrip().split(":")[-1])
                results[key][0].append(x)
                if x > threhold[key]:
                    results[key][1].append(pidx)

    for key in results.keys():
        print(key)
        print(f"Mean: {np.mean(results[key][0])}")
        print(f"Max:  {np.max(results[key][0])}")
        print(f"Subjects over threshold: {results[key][1]}")


if __name__ == "__main__":
    main()
