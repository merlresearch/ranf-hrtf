# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

import numpy as np
import sofar as sf

from ranf.evaluating_sparse_vbap import load_hrtf, sparse_vbap
from ranf.utils.config import NUMSONICOMDIRECTIONS
from ranf.utils.reconstruction import hrtf2hrir_minph
from ranf.utils.util import load_seen_unseen_didxs, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sonicom_path", type=str)
    parser.add_argument("upsample", type=int)
    parser.add_argument("maximum_subject_index", type=int)
    args = parser.parse_args()
    seed_everything(0)

    seen_didxs, _ = load_seen_unseen_didxs(args.upsample)

    sonicom_path = Path(args.sonicom_path)
    hrtf_type = "FreeFieldCompMinPhase_48kHz"
    pred_path = sonicom_path.joinpath(f"spvbap_{args.upsample:03}")
    pred_path.mkdir(parents=True, exist_ok=True)

    fname = sonicom_path.joinpath("subjects").joinpath(f"P{1:04}_{hrtf_type}.sofa")
    _, _, _, locs, _ = load_hrtf(fname)

    gain_mag = np.zeros((NUMSONICOMDIRECTIONS, NUMSONICOMDIRECTIONS), dtype=locs.dtype)
    gain_itd = np.zeros((NUMSONICOMDIRECTIONS, NUMSONICOMDIRECTIONS), dtype=locs.dtype)

    for didx in range(NUMSONICOMDIRECTIONS):
        tmp_seen_didxs = seen_didxs
        measured_locs = locs[tmp_seen_didxs, :]

        gain_mag[didx, tmp_seen_didxs] = sparse_vbap(locs[didx, :], measured_locs, normalize=True)
        gain_itd[didx, tmp_seen_didxs] = sparse_vbap(locs[didx, :], measured_locs, normalize=False)

    specs_all, itds_all = [], []

    for sidx in range(args.maximum_subject_index):
        pidx = sidx + 1
        fname = sonicom_path.joinpath("subjects").joinpath(f"P{pidx:04}_{hrtf_type}.sofa")
        sofa_file = sf.read_sofa(fname)
        _, specs, itds, locs, _ = load_hrtf(fname)

        pred = np.einsum("nm,mcf->ncf", gain_mag, specs)
        pred_itd = np.einsum("nm,m->n", gain_itd, itds)
        pred_hrir = hrtf2hrir_minph(pred, itd=pred_itd[:, None], nfft=sofa_file.Data_IR.shape[-1])

        sofa_file.Data_IR = pred_hrir.astype(np.float64)
        sf.write_sofa(pred_path.joinpath(fname.stem), sofa_file)
        specs_all.append(pred)
        itds_all.append(pred_itd)

    npz_dir = sonicom_path.parent.joinpath(f"sp_level_{args.upsample:03}")
    npz_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_dir.joinpath(f"spvbap_features_original_{args.upsample:03}.npz"),
        specs=np.array(specs_all).astype(np.float32),
        itds=np.array(itds_all).astype(np.float32),
    )


if __name__ == "__main__":
    main()
