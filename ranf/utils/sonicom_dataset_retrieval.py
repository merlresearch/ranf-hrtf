# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from copy import deepcopy
from pathlib import Path

import numpy as np
import sofar as sf
import torch

from ranf.utils.config import TGTDIDXS003, TGTDIDXS005, TGTDIDXS019, TGTDIDXS100


class SONICOMMulti(torch.utils.data.Dataset):
    def __init__(self, config, stage="pretrain", mode="train"):
        self.nretrieval = config.nretrieval
        self.mode = mode

        upsample = config.upsample
        assert upsample in {3, 5, 19, 100}

        if upsample == 3:
            seen_didxs = TGTDIDXS003

        elif upsample == 5:
            seen_didxs = TGTDIDXS005

        elif upsample == 19:
            seen_didxs = TGTDIDXS019

        elif upsample == 100:
            seen_didxs = TGTDIDXS100
        else:
            raise ValueError(f"config.upsample should be in (3, 5, 19, 100) but is {config.upsample}.")

        self.seen_didxs = seen_didxs
        self.unseen_didxs = sorted(list(set(range(793)) - set(seen_didxs)))

        npz = np.load(config.features)
        self.specs = npz["specs"]
        self.ilds = npz["ilds"]
        self.itds = npz["itds"]
        self.locs = np.deg2rad(npz["locs"])
        self.locs[:, :, 0] -= np.pi

        npz = np.load(config.retrieval)
        lsd_mat = npz["lsd_mat"]
        itdd_mat = npz["itdd_mat"]

        self.retrieved_subjects = []
        for sidx in range(lsd_mat.shape[0]):
            sidxs = []
            if config.retrieval_priority == "itdd":
                for itdd in sorted(list(set(itdd_mat[sidx, :]))):
                    _sidxs = np.where(itdd_mat[sidx, :] == itdd)[0]
                    order = np.argsort(lsd_mat[sidx, _sidxs])
                    sidxs += _sidxs[order].tolist()

            elif config.retrieval_priority == "lsd":
                for lsd in sorted(list(set(lsd_mat[sidx, :]))):
                    _sidxs = np.where(lsd_mat[sidx, :] == lsd)[0]
                    sidxs += _sidxs.tolist()

            else:
                raise NameError(f"{config.retrieval_priority} is not supported")

            self.retrieved_subjects.append(sidxs[: config.npool])

        self.sidxs, self.didxs = [], []
        if stage == "pretrain" and mode == "train":
            for sidx in config.train_subjects:
                for didx in range(793):
                    self.sidxs.append(sidx)
                    self.didxs.append(didx)

            for sidx in config.valid_subjects:
                for didx in self.seen_didxs:
                    self.sidxs.append(sidx)
                    self.didxs.append(didx)

        elif stage == "pretrain" and mode == "valid":
            for sidx in config.valid_subjects:
                for didx in self.unseen_didxs:
                    self.sidxs.append(sidx)
                    self.didxs.append(didx)

        elif stage == "adaptation":
            for sidx in config.test_subjects:
                for didx in self.seen_didxs:
                    self.sidxs.append(sidx)
                    self.didxs.append(didx)

    def __len__(self):
        return len(self.sidxs)

    def __getitem__(self, idx):
        tgt_sidx, tgt_didx = self.sidxs[idx], self.didxs[idx]
        tgt_spec = self.specs[tgt_sidx, tgt_didx, :, :]
        tgt_ild = self.ilds[tgt_sidx, tgt_didx]
        tgt_itd = self.itds[tgt_sidx, tgt_didx].astype(np.float32)
        tgt_loc = self.locs[tgt_sidx, tgt_didx, :]

        if self.mode == "train":
            rng = np.random.default_rng(idx)
            ret_sidxs = rng.choice(self.retrieved_subjects[tgt_sidx], self.nretrieval)
        else:
            ret_sidxs = np.array(self.retrieved_subjects[tgt_sidx][: self.nretrieval])

        ret_specs = self.specs[ret_sidxs, tgt_didx, :, :]
        ret_itds = self.itds[ret_sidxs, tgt_didx].astype(np.float32)
        ret_locs = self.locs[ret_sidxs, tgt_didx, :]

        return tgt_spec, tgt_ild, tgt_itd, tgt_loc, ret_specs, ret_itds, ret_locs, tgt_sidx, ret_sidxs


class SONICOMMultiInference(torch.utils.data.Dataset):
    def __init__(self, config):
        if hasattr(config, "inference_sampling"):
            self.sampling = config.inference_sampling
        else:
            self.sampling = False

        self.fs = 48000
        self.hrtf_type = "FreeFieldCompMinPhase_48kHz"
        self.upsample = config.upsample
        self.nretrieval = config.nretrieval
        self.sonicom_path = Path(config.features).parent.parent.joinpath("subjects")

        npz = np.load(config.features)
        self.specs = npz["specs"]
        self.ilds = npz["ilds"]
        self.itds = npz["itds"]
        self.locs = np.deg2rad(npz["locs"])
        self.locs[:, :, 0] -= np.pi

        npz = np.load(config.retrieval)
        lsd_mat = npz["lsd_mat"]
        itdd_mat = npz["itdd_mat"]

        self.retrieved_subjects = []
        for sidx in range(lsd_mat.shape[0]):
            sidxs = []
            if config.retrieval_priority == "itdd":
                for itdd in sorted(list(set(itdd_mat[sidx, :]))):
                    _sidxs = np.where(itdd_mat[sidx, :] == itdd)[0]
                    order = np.argsort(lsd_mat[sidx, _sidxs])
                    sidxs += _sidxs[order].tolist()

            elif config.retrieval_priority == "lsd":
                for lsd in sorted(list(set(lsd_mat[sidx, :]))):
                    _sidxs = np.where(lsd_mat[sidx, :] == lsd)[0]
                    sidxs += _sidxs.tolist()

            else:
                raise NameError(f"{config.retrieval_priority} is not supported")

            self.retrieved_subjects.append(sidxs[: config.npool])

        self.fnames, self.sidxs = [], []
        for sidx in config.test_subjects:
            if sidx < 200:
                pidx = sidx + 1
                fname = self.sonicom_path.joinpath(f"P{pidx:04}_{self.hrtf_type}.sofa")
                self.fnames.append(fname)
                self.sidxs.append(sidx)
            else:
                fname = (
                    Path(config.features).parent.parent.joinpath("lap-task2-upsampled").joinpath(f"{self.upsample:003}")
                )
                fname = fname.joinpath(f"LAPtask2_{config.upsample}_{sidx+1-200}.sofa")
                self.fnames.append(fname)
                self.sidxs.append(sidx)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        tgt_sidx = self.sidxs[idx]
        tgt_sofa_file = sf.read_sofa(self.fnames[idx])
        tgt_hrir = tgt_sofa_file.Data_IR.astype(np.float32)

        loc = deepcopy(tgt_sofa_file.SourcePosition)

        loc = np.deg2rad(loc).astype(np.float32)
        loc[:, 0] -= np.pi
        retval = [tgt_sofa_file, tgt_hrir, loc]

        if self.sampling:
            rng = np.random.default_rng(tgt_sidx)
            ret_sidxs = rng.choice(self.retrieved_subjects[tgt_sidx], self.nretrieval)
        else:
            ret_sidxs = np.array(self.retrieved_subjects[tgt_sidx][: self.nretrieval])

        ret_specs = self.specs[ret_sidxs, :, :, :].transpose(1, 0, 2, 3)
        ret_itds = self.itds[ret_sidxs, :].transpose(1, 0).astype(np.float32)
        ret_locs = self.locs[ret_sidxs, :, :].transpose(1, 0, 2)

        retval += [ret_specs, ret_itds, ret_locs]
        retval += [np.tile(tgt_sidx, (793)), np.tile(ret_sidxs, (793, 1))]
        return retval
