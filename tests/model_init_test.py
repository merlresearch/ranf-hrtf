# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch

from ranf.utils.neural_field_ojsp import RANF, CbCNeuralField, PEFTNeuralField


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("f_nyquist", [129, 257])
@pytest.mark.parametrize("hidden_features", [16, 64])
@pytest.mark.parametrize("hidden_layers", [1, 4])
@pytest.mark.parametrize("scale", [1, 3])
@pytest.mark.parametrize("dropout", [0, 0.1])
@pytest.mark.parametrize("n_listeners", [100, 200])
@pytest.mark.parametrize("activation", ["PReLU", "GELU"])
@pytest.mark.parametrize("peft", ["bitfit", "lora"])
@pytest.mark.parametrize("itd_skip_connection", [False, True])
def test_peft_neural_field(
    batch_size,
    f_nyquist,
    hidden_features,
    hidden_layers,
    scale,
    dropout,
    n_listeners,
    activation,
    peft,
    itd_skip_connection,
):

    out_features = 2 * f_nyquist
    model = PEFTNeuralField(
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        scale=scale,
        dropout=dropout,
        n_listeners=n_listeners,
        activation=activation,
        peft=peft,
        itd_skip_connection=itd_skip_connection,
    )

    model.train()

    input_specs_db = None
    input_itds = None
    input_locs = None
    ret_sidxs = None
    tgt_loc = torch.rand(batch_size, 2)
    tgt_sidx = torch.randint(low=0, high=n_listeners, size=(batch_size,))

    mag, itd = model(input_specs_db, input_itds, input_locs, tgt_loc, tgt_sidx, ret_sidxs)
    assert list(mag.shape) == [batch_size, 2, f_nyquist]
    assert list(itd.shape) == [batch_size, 1]


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("f_nyquist", [129, 257])
@pytest.mark.parametrize("hidden_features", [16, 64])
@pytest.mark.parametrize("hidden_layers", [1, 4])
@pytest.mark.parametrize("scale", [1, 3])
@pytest.mark.parametrize("dropout", [0, 0.1])
@pytest.mark.parametrize("n_listeners", [100, 200])
@pytest.mark.parametrize("activation", ["PReLU", "GELU"])
@pytest.mark.parametrize("itd_skip_connection", [False, True])
def test_cbc_neural_field(
    batch_size,
    f_nyquist,
    hidden_features,
    hidden_layers,
    scale,
    dropout,
    n_listeners,
    activation,
    itd_skip_connection,
):

    out_features = 2 * f_nyquist
    model = CbCNeuralField(
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        scale=scale,
        dropout=dropout,
        n_listeners=n_listeners,
        activation=activation,
        itd_skip_connection=itd_skip_connection,
    )

    model.train()

    input_specs_db = None
    input_itds = None
    input_locs = None
    ret_sidxs = None
    tgt_loc = torch.rand(batch_size, 2)
    tgt_sidx = torch.randint(low=0, high=n_listeners, size=(batch_size,))

    mag, itd = model(input_specs_db, input_itds, input_locs, tgt_loc, tgt_sidx, ret_sidxs)

    assert list(mag.shape) == [batch_size, 2, f_nyquist]
    assert list(itd.shape) == [batch_size, 1]


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("f_nyquist", [257])
@pytest.mark.parametrize("n_retrievals", [1, 3, 5])
@pytest.mark.parametrize("hidden_features", [64])
@pytest.mark.parametrize("hidden_layers", [1, 4])
@pytest.mark.parametrize("spec_hidden_layers", [1, 2])
@pytest.mark.parametrize("itd_hidden_layers", [1, 2])
@pytest.mark.parametrize("conv_layers", [4])
@pytest.mark.parametrize("scale", [1])
@pytest.mark.parametrize("dropout", [0])
@pytest.mark.parametrize("n_listeners", [200])
@pytest.mark.parametrize("rnn", ["GRU", "LSTM"])
@pytest.mark.parametrize("activation", ["GELU"])
@pytest.mark.parametrize("norm", ["Identity", "LayerNorm"])
@pytest.mark.parametrize("spec_res", [False, True])
@pytest.mark.parametrize("itd_res", [False, True])
@pytest.mark.parametrize("lora_retrieval", ["alter", "diff", "target"])
@pytest.mark.parametrize("itd_skip_connection", [False, True])
def test_ranf(
    batch_size,
    f_nyquist,
    n_retrievals,
    hidden_features,
    hidden_layers,
    spec_hidden_layers,
    itd_hidden_layers,
    conv_layers,
    scale,
    dropout,
    n_listeners,
    rnn,
    activation,
    norm,
    spec_res,
    itd_res,
    lora_retrieval,
    itd_skip_connection,
):

    out_features = 2 * f_nyquist
    model = RANF(
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        spec_hidden_layers=spec_hidden_layers,
        itd_hidden_layers=itd_hidden_layers,
        conv_layers=conv_layers,
        out_features=out_features,
        scale=scale,
        dropout=dropout,
        n_listeners=n_listeners,
        rnn=rnn,
        activation=activation,
        norm=norm,
        spec_res=spec_res,
        itd_res=itd_res,
        lora_retrieval=lora_retrieval,
        itd_skip_connection=itd_skip_connection,
    )

    model.train()

    input_specs_db = torch.rand(batch_size, n_retrievals, 2, f_nyquist)
    input_itds = torch.rand(batch_size, n_retrievals, 1)
    input_locs = torch.rand(batch_size, n_retrievals, 2)
    ret_sidxs = torch.randint(low=0, high=n_listeners, size=(batch_size, n_retrievals))
    tgt_loc = torch.rand(batch_size, 2)
    tgt_sidx = torch.randint(low=0, high=n_listeners, size=(batch_size,))

    mag, itd = model(input_specs_db, input_itds, input_locs, tgt_loc, tgt_sidx, ret_sidxs)
    assert list(mag.shape) == [batch_size, 2, f_nyquist]
    assert list(itd.shape) == [batch_size, 1]
