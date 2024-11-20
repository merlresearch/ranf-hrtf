# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
import pathlib

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
from ranf.utils.util import count_parameters, db2linear, linear2db, seed_everything


def freeze_model_for_peft(model):
    for name, param in model.named_parameters():
        if "embed_layer" not in name:
            # This param is independent of the target subject and should be frozen
            param.requires_grad = False
            continue

        if "embed_layer_bitfit" in name:
            # In BitFit, the subject-dependent bias is computed by an FC layer
            if "bias" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                logging.info(f"{name} is trainable")

        elif "embed_layer_lorau" in name:
            # One low-rank matrix for LoRA "u" always depends on the target subject
            if "bias" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                logging.info(f"{name} is trainable")

        elif "embed_layer_lorav" in name:
            # Another low-rank matrix for LoRA "v" depends on the retrieved subject in the core-block of RANF
            if "bias" in name:
                param.requires_grad = False
            else:
                if isinstance(model, neural_field.RANF):
                    if int(name.split(".")[1]) < model.hidden_layers:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                        logging.info(f"{name} is trainable")
                else:
                    param.requires_grad = True
                    logging.info(f"{name} is trainable")

        else:
            raise ValueError("Invalid parameter name")

    logging.info(f"Number of trainable parameters: {count_parameters(model)}")


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

    path = pathlib.Path(args.config_path)
    config = OmegaConf.load(path.joinpath("config.yaml"))
    seed_everything(config.seed)

    log = path.joinpath("log")
    log.mkdir(parents=True, exist_ok=True)
    log.joinpath("adaptation").mkdir(parents=True, exist_ok=True)
    log_name = log.joinpath("adaptation").joinpath("adaptation.log")
    logging.basicConfig(filename=log_name, level=logging.INFO)

    # Prepare dataset and dataloader
    tr_dataset = SONICOMMulti(
        config.dataset,
        stage="adaptation",
        mode="train",
    )
    dev_dataset = SONICOMMulti(
        config.dataset,
        stage="adaptation",
        mode="valid",
    )
    tr_data_loader = DataLoader(
        tr_dataset,
        batch_size=config.adaptation.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.adaptation.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=config.adaptation.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.adaptation.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers,
    )

    # Prepare model
    model = getattr(neural_field, config.model.name)(**config.model.config)
    model = model.to(config.device)
    model.load_state_dict(torch.load(path.joinpath("best.ckpt"), map_location=config.device))

    # Freezing the model except for the subject-specific parameters
    freeze_model_for_peft(model)

    # Prepare the optimizer and scheduler
    optimizer = getattr(optim, config.optimizer.name)(model.parameters(), **config.optimizer.config)

    tr_loss, tr_loss_min = [], 1.0e15
    logging.info("Start adaptation...")
    for epoch in range(config.adaptation.num_epoch):

        # Adaptation
        model.train()
        for data in tqdm(tr_data_loader):
            data = [x.to(config.device) for x in data]
            loss_val = forward(data, model, config.loss)
            optimizer.zero_grad()
            loss_val.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.adaptation.clip)
            optimizer.step()

        # Validation but on adaptation data itself
        running_loss = []
        running_lsd, running_ild_diff, running_itd_diff = [], [], []
        model.eval()
        for data in tqdm(dev_data_loader):
            data = [x.to(config.device) for x in data]
            loss_val, lsd_val, ild_diff_val, itd_diff_val = forward(data, model, config.loss)
            running_loss.append(loss_val.item())
            running_lsd.append(lsd_val.item())
            running_itd_diff.append(itd_diff_val.item())
            running_ild_diff.append(ild_diff_val.item())

        tr_loss.append(np.mean(running_loss))

        logging.info(f"Epoch {epoch}")
        logging.info(f"tr_loss: {tr_loss[-1]}")
        logging.info(f"dev lsd: {np.mean(running_lsd)}")
        logging.info(f"dev ild diff: {np.mean(running_ild_diff)}")
        logging.info(f"dev itd diff: {np.mean(running_itd_diff)}")

        if tr_loss[-1] <= tr_loss_min:
            tr_loss_min = tr_loss[-1]
            torch.save(model.state_dict(), path.joinpath("adaptation.ckpt"))

        if np.isnan(tr_loss[-1]):
            logging.info("Loss is Nan. Training is stopped")
            break


if __name__ == "__main__":
    main()
