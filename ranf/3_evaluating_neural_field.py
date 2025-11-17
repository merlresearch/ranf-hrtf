# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
import os
import pathlib

import numpy as np
import sofar as sf
import torch
from omegaconf import OmegaConf
from spatialaudiometrics import lap_challenge as lap

from ranf.utils import neural_field_ojsp as neural_field
from ranf.utils.reconstruction import hrtf2hrir_minph
from ranf.utils.sonicom_dataset_retrieval import SONICOMMultiInferenceWithSP
from ranf.utils.util import db2linear, linear2db, load_seen_unseen_didxs, seed_everything, torch_reproducible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--remove_hrtf_files", action="store_true")
    args = parser.parse_args()

    path = pathlib.Path(args.config_path)
    config = OmegaConf.load(path.joinpath("config.yaml"))
    seed_everything(config.seed)
    torch_reproducible()

    _, unseen_didxs = load_seen_unseen_didxs(config.dataset.upsample)

    log = path.joinpath("log")
    log.mkdir(parents=True, exist_ok=True)
    log.joinpath("eval").mkdir(parents=True, exist_ok=True)
    log_name = log.joinpath("eval").joinpath("eval.log")
    logging.basicConfig(filename=log_name, level=logging.INFO)

    eval_dataset = SONICOMMultiInferenceWithSP(config.dataset)

    model = getattr(neural_field, config.model.name)(**config.model.config)
    model = model.to(config.device)
    model.load_state_dict(torch.load(path.joinpath("adaptation.ckpt"), map_location=config.device))

    logging.info("Start evaluation...")
    model.eval()
    pred_dbs, pred_itds = [], []
    for data in eval_dataset:
        sofa_file, hrir = data[:2]
        tgt_loc, ret_specs, ret_itds, ret_locs, tgt_sidx, ret_sidxs = (
            torch.tensor(x).to(config.device) for x in data[2:]
        )
        ret_specs_db = linear2db(ret_specs)
        pred_db, pred_itd = model(ret_specs_db, ret_itds, ret_locs, tgt_loc, tgt_sidx, ret_sidxs)

        pred = db2linear(pred_db).detach().to("cpu").numpy()
        pred_itd = pred_itd.detach().to("cpu").numpy()
        pred_hrir = hrtf2hrir_minph(pred, itd=pred_itd, nfft=hrir.shape[-1])

        target_path = log.joinpath("eval").joinpath(f"target_p{data[-2][0]+1:04}.sofa")
        pred_path = log.joinpath("eval").joinpath(f"pred_p{data[-2][0]+1:04}.sofa")

        sf.write_sofa(target_path, sofa_file)

        sofa_file.Data_IR[unseen_didxs, ...] = pred_hrir.astype(np.float64)[unseen_didxs, ...]
        sf.write_sofa(pred_path, sofa_file)

        metrics = lap.calculate_task_two_metrics(str(target_path), str(pred_path))[0]

        logging.info(f"P{data[-2][0]+1:04} evaluation")
        logging.info(f"ITD difference (µs): {metrics[0]}")
        logging.info(f"ILD difference (dB): {metrics[1]}")
        logging.info(f"LSD (dB): {metrics[2]}")

        pred_dbs.append(pred_db)
        pred_itds.append(pred_itd)

    if args.remove_hrtf_files:
        os.remove(target_path)
        os.remove(pred_path)


if __name__ == "__main__":
    main()
