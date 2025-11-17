# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from copy import deepcopy
from pathlib import Path

import numpy as np
import sofar as sf
import torch
from spatialaudiometrics import hrtf_metrics as hf

from ranf.utils.config import NUMSONICOMDIRECTIONS
from ranf.utils.util import load_seen_unseen_didxs


class SONICOMMultiWithSP(torch.utils.data.Dataset):
    def __init__(self, config, stage="pretrain", mode="train"):
        self.nretrieval = config.nretrieval
        self.mode = mode
        self.seen_didxs, self.unseen_didxs = load_seen_unseen_didxs(config.upsample)

        # Loading pre-processed features
        npz = np.load(config.features)
        self.specs = npz["specs"]
        self.ilds = npz["ilds"]
        self.itds = npz["itds"]
        self.locs = np.deg2rad(npz["locs"])
        self.locs[:, :, 0] -= np.pi

        # Loading the distance matrices for retrieval
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

        # Preparing pairs of subject and direction indices.
        self.sidxs, self.didxs = [], []
        if stage == "pretrain" and mode == "train":
            for sidx in config.train_subjects:
                for didx in range(NUMSONICOMDIRECTIONS):
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

        # Loading signal-processing results
        self.signal = []
        self.specs_signal = []
        self.itds_signal = []

        if hasattr(config.signal, "spvbap_path"):
            self.signal.append("spvbap")
            npz = np.load(config.signal.spvbap_path)
            self.specs_signal.append(npz["specs"])
            self.itds_signal.append(npz["itds"])

        if len(self.signal) > 0:
            self.specs_signal = np.concatenate(self.specs_signal, axis=-2)
            self.itds_signal = np.stack(self.itds_signal, axis=-1)

            # This option changes whether the signal processing results for the target subject will be used or not.
            self.concat_tgt_signal = config.signal.concat_tgt_signal

    def __len__(self):
        return len(self.sidxs)

    def __getitem__(self, idx):
        tgt_sidx, tgt_didx = self.sidxs[idx], self.didxs[idx]
        tgt_spec = self.specs[tgt_sidx, tgt_didx, :, :]
        tgt_itd = self.itds[tgt_sidx, tgt_didx].astype(np.float32)
        tgt_loc = self.locs[tgt_sidx, tgt_didx, :]

        if self.mode == "train":
            rng = np.random.default_rng(idx)
            ret_sidxs = rng.choice(self.retrieved_subjects[tgt_sidx], self.nretrieval)
        else:
            ret_sidxs = np.array(self.retrieved_subjects[tgt_sidx][: self.nretrieval])

        # Retrieving subjects
        input_specs = self.specs[ret_sidxs, tgt_didx, :, :]
        input_itds = np.expand_dims(self.itds[ret_sidxs, tgt_didx], axis=-1).astype(np.float32)
        input_locs = self.locs[ret_sidxs, tgt_didx, :]

        if len(self.signal) > 0:
            tmp_specs = [input_specs, self.specs_signal[ret_sidxs, tgt_didx, :, :]]
            tmp_itds = [input_itds, self.itds_signal[ret_sidxs, tgt_didx, :]]

            if self.concat_tgt_signal:
                tmp_specs += [np.tile(self.specs_signal[tgt_sidx, tgt_didx, :, :], (len(ret_sidxs), 1, 1))]
                tmp_itds += [np.tile(self.itds_signal[tgt_sidx, tgt_didx, :], (len(ret_sidxs), 1))]

            input_specs = np.concatenate(tmp_specs, axis=1)
            input_itds = np.concatenate(tmp_itds, axis=-1)

        return tgt_spec, tgt_itd, tgt_loc, input_specs, input_itds, input_locs, tgt_sidx, ret_sidxs


class SONICOMMultiInferenceWithSP(torch.utils.data.Dataset):
    def __init__(self, config):
        if hasattr(config, "inference_sampling"):
            self.sampling = config.inference_sampling
        else:
            self.sampling = False

        if hasattr(config, "azimuth_calibration"):
            self.azimuth_calibration = config.azimuth_calibration

        else:
            self.azimuth_calibration = True

        self.fs = 48000
        self.hrtf_type = "FreeFieldCompMinPhase_48kHz"
        self.upsample = config.upsample
        self.nretrieval = config.nretrieval
        self.sonicom_path = Path(config.features).parent.parent.joinpath("sonicom").joinpath("subjects")

        npz = np.load(config.features)
        self.specs = npz["specs"].transpose(1, 0, 2, 3)
        self.itds = npz["itds"].astype(np.float32).transpose(1, 0)
        self.locs = np.deg2rad(npz["locs"]).transpose(1, 0, 2)
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
            pidx = sidx + 1
            fname = self.sonicom_path.joinpath(f"P{pidx:04}_{self.hrtf_type}.sofa")
            self.fnames.append(fname)
            self.sidxs.append(sidx)

        self.signal = []
        self.specs_signal = []
        self.itds_signal = []

        if hasattr(config.signal, "spvbap_path"):
            self.signal.append("spvbap")
            npz = np.load(config.signal.spvbap_path)
            self.specs_signal.append(npz["specs"].transpose(1, 0, 2, 3))
            self.itds_signal.append(npz["itds"].transpose(1, 0))

        if len(self.signal) > 0:
            self.specs_signal = np.concatenate(self.specs_signal, axis=-2)
            self.itds_signal = np.stack(self.itds_signal, axis=-1)
            self.concat_tgt_signal = config.signal.concat_tgt_signal

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        tgt_sidx = self.sidxs[idx]
        tgt_sofa_file = sf.read_sofa(self.fnames[idx])
        tgt_hrir = tgt_sofa_file.Data_IR.astype(np.float32)

        loc = deepcopy(tgt_sofa_file.SourcePosition)

        if self.azimuth_calibration:
            _, itd, _ = hf.itd_estimator_maxiacce(tgt_hrir, np.float64(self.fs))
            if self.upsample == 3:
                alpha = (itd[414] - itd[4]) / 90
                beta = itd[4] / alpha
                loc[:, 0] = (loc[:, 0] + beta + 360) % 360

            elif self.upsample == 5:
                alpha = (itd[612] - itd[203]) / 90
                beta = itd[4] / alpha
                loc[:, 0] = (loc[:, 0] + beta + 360) % 360

            elif self.upsample == 19:
                alpha = (itd[269] - itd[546]) / 120
                beta = itd[4] / alpha
                loc[:, 0] = (loc[:, 0] + beta + 360) % 360

            elif self.upsample == 100:
                alpha = (itd[436] - itd[788]) / 80
                beta = (itd[788] - alpha * 5.0) / alpha
                loc[:, 0] = (loc[:, 0] + beta + 360) % 360
            else:
                raise ValueError(f"upsample should be in (3, 5, 19, 100) but is {self.upsample}.")

        loc = np.deg2rad(loc).astype(np.float32)
        loc[:, 0] -= np.pi
        retval = [tgt_sofa_file, tgt_hrir, loc]

        if self.sampling:
            rng = np.random.default_rng(tgt_sidx)
            ret_sidxs = rng.choice(self.retrieved_subjects[tgt_sidx], self.nretrieval)
        else:
            ret_sidxs = np.array(self.retrieved_subjects[tgt_sidx][: self.nretrieval])

        input_specs = self.specs[:, ret_sidxs, :, :]
        input_itds = np.expand_dims(self.itds[:, ret_sidxs], axis=-1).astype(np.float32)
        input_locs = self.locs[:, ret_sidxs, :]

        if len(self.signal) > 0:
            tmp_specs = [input_specs, self.specs_signal[:, ret_sidxs, :, :]]
            tmp_itds = [input_itds, self.itds_signal[:, ret_sidxs, :]]
            if self.concat_tgt_signal:
                tmp_specs += [
                    np.repeat(np.expand_dims(self.specs_signal[:, tgt_sidx, :, :], axis=1), len(ret_sidxs), axis=1)
                ]
                tmp_itds += [np.repeat(np.expand_dims(self.itds_signal[:, tgt_sidx], axis=1), len(ret_sidxs), axis=1)]

            input_specs = np.concatenate(tmp_specs, axis=2)
            input_itds = np.concatenate(tmp_itds, axis=-1)

        retval += [input_specs, input_itds, input_locs]
        retval += [np.tile(tgt_sidx, (NUMSONICOMDIRECTIONS)), np.tile(ret_sidxs, (NUMSONICOMDIRECTIONS, 1))]
        return retval
