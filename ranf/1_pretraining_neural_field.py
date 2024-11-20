# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ranf.utils import neural_field_icassp as neural_field
from ranf.utils.loss_functions import ild_diff_loss, itd_diff_loss, lsd_loss
from ranf.utils.sonicom_dataset_retrieval import SONICOMMulti
from ranf.utils.util import count_parameters, db2linear, linear2db, plot_hrtf, seed_everything


def forward(data, model, config):
    tgt_spec, tgt_ild, tgt_itd, tgt_loc, ret_specs, ret_itds, _, tgt_sidx, ret_sidxs = data
    spec_db, ret_specs_db = linear2db(tgt_spec), linear2db(ret_specs)
    pred_db, pred_itd = model(ret_specs_db, ret_itds, tgt_loc, tgt_sidx, ret_sidxs)
    pred = db2linear(pred_db)

    if model.training:
        threshold = config.threshold_itd
    else:
        threshold = 0.0

    loss_val = torch.mean(lsd_loss(spec_db, pred_db, use_index=False))
    itd_diff_loss_val = torch.mean(itd_diff_loss(tgt_itd, pred_itd[:, 0], threshold=threshold))
    loss_val = loss_val + config.weight_itd * itd_diff_loss_val
    if model.training:
        return loss_val

    # These metrcis are only for validation
    lsd_loss_val = torch.mean(lsd_loss(spec_db, pred_db, use_index=True))
    ild_diff_loss_val = torch.mean(ild_diff_loss(tgt_spec, pred, tgt_ild))
    return loss_val, lsd_loss_val, ild_diff_loss_val, itd_diff_loss_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    path = Path(args.config_path)
    config = OmegaConf.load(path.joinpath("config.yaml"))
    seed_everything(config.seed)

    log = path.joinpath("log")
    log.mkdir(parents=True, exist_ok=True)

    log_name = log.joinpath("exp.log")
    fig_name = log.joinpath("hrtf.png")
    logging.basicConfig(filename=log_name, level=logging.INFO)

    # Prepare dataset and dataloader
    tr_dataset = SONICOMMulti(
        config.dataset,
        stage="pretrain",
        mode="train",
    )
    dev_dataset = SONICOMMulti(
        config.dataset,
        stage="pretrain",
        mode="valid",
    )

    tr_data_loader = DataLoader(
        tr_dataset,
        batch_size=config.learning.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.learning.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=config.learning.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.learning.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers,
    )
    assert len(tr_dataset) > 0 and len(dev_dataset) > 0, len(dev_dataset)

    # Prepare model
    model = getattr(neural_field, config.model.name)(**config.model.config)
    model = model.to(config.device)
    logging.info(f"Number of trainable parameters: {count_parameters(model)}")

    if hasattr(config.model, "init_path"):
        model.load_state_dict(torch.load(config.init_path, map_location=config.device))

    # Prepare the optimizer and scheduler
    optimizer = getattr(optim, config.optimizer.name)(model.parameters(), **config.optimizer.config)

    if hasattr(config, "scheduler"):
        scheduler = getattr(optim.lr_scheduler, config.scheduler.name)(optimizer, **config.scheduler.config)
    else:
        scheduler = None

    tr_loss, dev_loss = [], []
    dev_loss_min = 1.0e15
    early_stop = 0

    logging.info("Start training...")
    for epoch in range(config.learning.num_epoch):

        # Training
        running_loss = []
        model.train()
        for data in tqdm(tr_data_loader):
            data = [x.to(config.device) for x in data]
            loss_val = forward(data, model, config.loss)
            optimizer.zero_grad()
            loss_val.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.learning.clip)
            optimizer.step()
            running_loss.append(loss_val.item())

        tr_loss.append(np.mean(running_loss))

        # Validation
        running_loss, running_lsd = [], []
        running_ild_diff, running_itd_diff = [], []
        model.eval()
        for data in tqdm(dev_data_loader):
            data = [x.to(config.device) for x in data]
            loss_val, lsd_val, ild_diff_val, itd_diff_val = forward(data, model, config.loss)
            running_loss.append(loss_val.item())
            running_lsd.append(lsd_val.item())
            running_itd_diff.append(itd_diff_val.item())
            running_ild_diff.append(ild_diff_val.item())

        # Visualization
        with torch.no_grad():
            data = dev_dataset[epoch % len(dev_dataset)]
            data = [torch.tensor(x)[None, ...].to(config.device) for x in data]
            tgt_spec, _, _, tgt_locs, ret_specs, ret_itds, _, tgt_sidx, ret_sidxs = data
            spec_db, ret_specs_db = linear2db(tgt_spec), linear2db(ret_specs)
            pred_db, _ = model(ret_specs_db, ret_itds, tgt_locs, tgt_sidx, ret_sidxs)
            lsd = torch.mean(lsd_loss(spec_db, pred_db)).item()
            plot_hrtf(fig_name, spec_db, pred_db, lsd)

        dev_loss.append(np.mean(running_loss))

        if scheduler is not None:
            scheduler.step(dev_loss[-1])

        logging.info(f"Epoch {epoch}")
        logging.info(f"tr_loss: {tr_loss[-1]}, dev_loss: {dev_loss[-1]}")
        logging.info(f"dev lsd: {np.mean(running_lsd)}")
        logging.info(f"dev ild diff: {np.mean(running_ild_diff)}")
        logging.info(f"dev itd diff: {np.mean(running_itd_diff)}")

        if dev_loss[-1] <= dev_loss_min:
            dev_loss_min = dev_loss[-1]
            early_stop = 0
            torch.save(model.state_dict(), path.joinpath("best.ckpt"))
        else:
            early_stop += 1

        if early_stop == config.learning.patience:
            logging.info(f"Early stopping at epoch {epoch}")
            break

        if np.isnan(dev_loss[-1]):
            logging.info("Loss is Nan. Training is stopped")
            break


if __name__ == "__main__":
    main()
