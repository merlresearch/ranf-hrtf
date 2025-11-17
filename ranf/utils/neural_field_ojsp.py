# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, inch, outch, dropout=0.0, activation="GELU", bias=True):
        super().__init__()
        self.fc = nn.Linear(inch, outch, bias=bias)
        self.activatin = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activatin(self.fc(x))
        x = self.dropout(x)
        return x


class LoRAMLP(nn.Module):
    def __init__(self, inch, outch, dropout, activation="GELU", bias=True):
        super().__init__()
        self.fc = nn.Linear(inch, outch, bias=bias)
        self.activatin = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, u=0.0, v=0.0, b=0.0):
        """Forward
        Args:
            x (torch.Tensor): Feature tensor of [batch, inch]
            u (torch.Tensor | float, optional): Vector to construct rank-1 matrix for LoRA [batch, inch] or 0.0
            v (torch.Tensor | float, optional): Vector to construct rank-1 matrix for LoRA [batch, outch] or 0.0
            b (torch.Tensor | float, optional): Additional bias for BitFit [batch, outch] or 0.0

        Returns:
            x (torch.Tensor)
        """
        z = u * torch.mean(v * x, -1, keepdim=True)
        x = self.fc(x) + z + b
        x = self.dropout(self.activatin(x))
        return x


class LoRAMLP4RANF(nn.Module):
    def __init__(self, inch, outch, dropout, activation="GELU", bias=True):
        super().__init__()
        self.fc = nn.Linear(inch, outch, bias=bias)
        self.activatin = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, u=0.0, v=0.0, b=0.0):
        """Forward
        Args:
            x (torch.Tensor): Feature tensor of [batch, freqs (downsampled), inch]
            u (torch.Tensor | float, optional): Vector to construct rank-R matrix for LoRA [batch, inch, R] or 0.0
            v (torch.Tensor | float, optional): Vector to construct rank-R matrix for LoRA [batch, outch, R] or 0.0
            b (torch.Tensor | float, optional): Additional bias for BitFit [batch, outch] or 0.0

        Returns:
            x (torch.Tensor)
        """
        z = torch.mean(v * x[..., None], -2)
        z = torch.mean(u * z[..., None, :], -1)
        x = self.fc(x) + z + b
        x = self.dropout(self.activatin(x))
        return x


class LoRATACMLP4(nn.Module):
    def __init__(self, inch, outch, dropout, activation="GELU", bias=True):
        super().__init__()
        self.in_linear_pass = nn.Linear(inch, inch // 2)
        self.in_linear_ave = nn.Linear(inch, inch // 2)
        self.in_activatin = getattr(nn, activation)()
        self.in_dropout = nn.Dropout(dropout)

        self.out_linear_lora = nn.Linear(inch // 2 * 2, outch, bias=bias)
        self.out_activatin = getattr(nn, activation)()
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, u=0, v=0, b=0):
        """Forward
        Args:
            x (torch.Tensor): feature tensor of [batch, K, freqs (downsampled), inch] where K is # of retrievals
            u (torch.Tensor | float, optional): vector to construct rank-R matrix for LoRA [batch, K, inch, R] or 0.0
            v (torch.Tensor | float, optional): vector to construct rank-R matrix for LoRA [batch, K, outch, R] or 0.0
            b (torch.Tensor | float, optional): additional bias for BitFit [batch, outch] or 0.0

        Returns:
            x (torch.Tensor)
        """
        K = x.shape[1]

        # Modified transform-average-concatenation (TAC)
        y = self.in_linear_ave(x)
        y = torch.tile(torch.mean(y, dim=1, keepdim=True), (1, K, 1, 1))
        x = self.in_linear_pass(x)
        x = torch.cat([x, y], dim=-1)
        x = self.in_dropout(self.in_activatin(x))

        # MLP with LoRA
        z = torch.mean(v[..., None, :, :] * x[..., None], -2)
        z = torch.mean(u[..., None, :, :] * z[..., None, :], -1)
        x = self.out_linear_lora(x) + z + b
        x = self.out_dropout(self.out_activatin(x))
        return x


class LSTMLoRATAC(nn.Module):
    def __init__(
        self,
        hidden_features,
        dropout=0.0,
        rnn="LSTM",
        activation="GELU",
        norm="LayerNorm",
    ):
        super().__init__()
        self.freq_blstm = getattr(nn, rnn)(
            hidden_features, hidden_features // 2, bidirectional=True, num_layers=1, batch_first=True
        )
        self.lora_mlp = LoRATACMLP4(hidden_features, hidden_features, dropout=dropout, bias=True, activation=activation)
        self.norm = getattr(nn, norm)(hidden_features)

    def forward(self, x, u=0.0, v=0.0, b=0.0):
        """Forward
        Args:
            x (torch.Tensor): feature tensor of [batch, K, freqs (downsampled), inch] where K is # of retrievals
            u (torch.Tensor | float, optional): vector to construct rank-R matrix for LoRA [batch, K, inch, R] or 0.0
            v (torch.Tensor | float, optional): vector to construct rank-R matrix for LoRA [batch, K, outch, R] or 0.0
            b (torch.Tensor | float, optional): additional bias for BitFit [batch, outch] or 0.0

        Returns:
            x (torch.Tensor)
        """
        batch, nretrieval, _, hidden_features = x.shape
        x = x.reshape(batch * nretrieval, -1, hidden_features)
        x = self.freq_blstm(x)[0]
        x = x.reshape(batch, nretrieval, -1, hidden_features)
        x = self.lora_mlp(x, u=u, v=v, b=b)
        return self.norm(x)


class PEFTNeuralField(nn.Module):
    def __init__(
        self,
        hidden_features=128,
        hidden_layers=1,
        out_features=258,
        scale=1,
        dropout=0.1,
        n_listeners=210,
        activation="GELU",
        peft="lora",
        itd_skip_connection=False,
    ):
        super().__init__()

        assert hidden_features % 2 == 0
        assert peft in {"lora", "bitfit"}

        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.itd_skip_connection = itd_skip_connection
        self.peft = peft

        # For random Fourier feature
        self.rng = np.random.default_rng(0)
        bmat = scale * self.rng.normal(0.0, 1.0, (hidden_features // 2, 4))
        self.bmat = torch.nn.Parameter(torch.tensor(bmat.astype(np.float32)), requires_grad=False)

        # For MLP
        self.hidden_loramlps = nn.ModuleList(
            [
                LoRAMLP(hidden_features, hidden_features, dropout=dropout, bias=peft != "bitfit", activation=activation)
                for _ in range(hidden_layers)
            ]
        )
        self.out_linear = LoRAMLP(
            hidden_features, out_features, dropout=0.0, bias=peft != "bitfit", activation="Identity"
        )

        self.itd_net = nn.Sequential(
            MLP(hidden_features, hidden_features // 2, dropout), MLP(hidden_features // 2, 1, activation="Identity")
        )

        # For PEFT
        self.n_listeners = n_listeners

        if peft == "bitfit":
            bitfit = []
            for _ in range(hidden_layers):
                bitfit.append(nn.Linear(n_listeners, hidden_features, bias=False))

            bitfit.append(nn.Linear(n_listeners, out_features, bias=False))
            self.embed_layer_bitfit = nn.ModuleList(bitfit)

        if peft == "lora":
            lorau, lorav = [], []
            for _ in range(hidden_layers):
                lorau.append(nn.Linear(n_listeners, hidden_features, bias=False))
                lorav.append(nn.Linear(n_listeners, hidden_features, bias=False))
                nn.init.zeros_(lorau[-1].weight)

            lorau.append(nn.Linear(n_listeners, out_features, bias=False))
            lorav.append(nn.Linear(n_listeners, hidden_features, bias=False))
            nn.init.zeros_(lorau[-1].weight)

            self.embed_layer_lorau = nn.ModuleList(lorau)
            self.embed_layer_lorav = nn.ModuleList(lorav)

    def forward(self, ret_specs_db, ret_itds, ret_locs, tgt_loc, tgt_sidx, ret_sidxs):
        """Forward

        Args:
            ret_specs_db: Unused in neural field without retrievals
            ret_itds: Unused in neural field without retrievals
            tgt_loc (torch.Tensor): Sound source location (azimuth, elevation, distance) in [batch, 3]
            tgt_sidx (torch.Tensor): Indices of the target subject in integer
            ret_sidxs: Unused in neural field without retrievals

        Returns:
            estimate (torch.Tensor): Estimated magnitude in [batch, 2, nfreqs]
            itd (torch.Tensor): Estimated ITD in [batch, 1]
        """
        batch = tgt_loc.shape[0]
        azimuth, elevation = tgt_loc[:, 0], tgt_loc[:, 1]
        sidxs = tgt_sidx

        onehot = F.one_hot(sidxs, self.n_listeners).type(torch.float32)

        emb = [azimuth.sin(), azimuth.cos(), elevation.sin(), elevation.cos()]
        emb = torch.stack(emb, -1) @ self.bmat.T
        emb = torch.concatenate([emb.sin(), emb.cos()], axis=-1)

        x = emb
        for n in range(self.hidden_layers):
            x = self.hidden_loramlps[n](
                x,
                u=self.embed_layer_lorau[n](onehot) if self.peft == "lora" else 0,
                v=self.embed_layer_lorav[n](onehot) if self.peft == "lora" else 0,
                b=self.embed_layer_bitfit[n](onehot) if self.peft == "bitfit" else 0,
            )

        estimate = self.out_linear(
            x,
            u=self.embed_layer_lorau[-1](onehot) if self.peft == "lora" else 0,
            v=self.embed_layer_lorav[-1](onehot) if self.peft == "lora" else 0,
            b=self.embed_layer_bitfit[-1](onehot) if self.peft == "bitfit" else 0,
        )
        estimate = estimate.reshape(batch, 2, -1)

        if self.itd_skip_connection:
            x = x + emb

        itd = self.itd_net(x)
        return estimate, itd


class CbCNeuralField(nn.Module):
    def __init__(
        self,
        hidden_features=128,
        embed_features=32,
        hidden_layers=1,
        out_features=258,
        scale=1,
        dropout=0.1,
        n_listeners=210,
        activation="GELU",
        itd_skip_connection=False,
    ):
        super().__init__()

        assert hidden_features % 2 == 0

        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.itd_skip_connection = itd_skip_connection

        # For random Fourier feature
        self.rng = np.random.default_rng(0)
        bmat = scale * self.rng.normal(0.0, 1.0, (hidden_features // 2, 4))
        self.bmat = torch.nn.Parameter(torch.tensor(bmat.astype(np.float32)), requires_grad=False)

        # For MLP
        self.n_listeners = n_listeners

        # Latent vector for PEFT
        self.embed_layer_lorau = nn.Linear(
            n_listeners,
            embed_features,
            bias=False,
        )
        self.hidden_mlps = nn.ModuleList(
            [
                MLP(
                    hidden_features + embed_features if n == 0 else hidden_features,
                    hidden_features,
                    dropout=dropout,
                    bias=True,
                    activation=activation,
                )
                for n in range(hidden_layers)
            ]
        )
        self.out_linear = MLP(hidden_features, out_features, dropout=0.0, bias=True, activation="Identity")

        self.itd_net = nn.Sequential(
            MLP(hidden_features, hidden_features // 2, dropout), MLP(hidden_features // 2, 1, activation="Identity")
        )

    def forward(self, ret_specs_db, ret_itds, ret_locs, tgt_loc, tgt_sidx, ret_sidxs):
        """Forward

        Args:
            ret_specs_db: Unused in neural field without retrievals
            ret_itds: Unused in neural field without retrievals
            tgt_loc (torch.Tensor): Sound source location (azimuth, elevation, distance) in [batch, 3]
            tgt_sidx (torch.Tensor): Indices of the target subject in integer
            ret_sidxs: Unused in neural field without retrievals

        Returns:
            estimate (torch.Tensor): Estimated magnitude in [batch, 2, nfreqs]
            itd (torch.Tensor): Estimated ITD in [batch, 1]
        """
        azimuth, elevation = tgt_loc[:, 0], tgt_loc[:, 1]
        sidxs = tgt_sidx

        batch = azimuth.shape[0]
        onehot = F.one_hot(sidxs, self.n_listeners).type(torch.float32)
        listener_emb = self.embed_layer_lorau(onehot)

        emb = [azimuth.sin(), azimuth.cos(), elevation.sin(), elevation.cos()]
        emb = torch.stack(emb, -1) @ self.bmat.T
        emb = torch.concatenate([emb.sin(), emb.cos()], axis=-1)

        x = torch.concatenate([emb, listener_emb], axis=-1)
        for n in range(self.hidden_layers):
            x = self.hidden_mlps[n](x)

        estimate = self.out_linear(x)
        estimate = estimate.reshape(batch, 2, -1)

        if self.itd_skip_connection:
            x = x + emb

        itd = self.itd_net(x)
        return estimate, itd


class RANF(nn.Module):
    def __init__(
        self,
        hidden_features=128,
        hidden_layers=1,
        spec_hidden_layers=1,
        itd_hidden_layers=1,
        conv_layers=3,
        scale=1,
        dropout=0.1,
        n_listeners=200,
        rnn="LSTM",
        activation="GELU",
        norm="LayerNorm",
        peft="lora",
        lora_rank=1,
        itd_scale=np.pi / 45,
        conv_in=2,
        emb_in=3,
        spec_res=False,
        itd_res=False,
        itd_activation="Identity",
        itd_skip_connection=True,
        azimuth_calibration=False,
        **kwargs,
    ):
        super().__init__()

        assert hidden_features % 2 == 0
        assert conv_in % 2 == 0 and conv_in > 1
        assert conv_layers > 2
        assert peft == "lora", "Our current implementation supports only LoRA"
        assert lora_rank == 1, "Our current implementation supports only the rank-1 case of LoRA"

        self.hidden_layers = hidden_layers
        self.spec_hidden_layers = spec_hidden_layers
        self.itd_hidden_layers = itd_hidden_layers
        self.hidden_features = hidden_features
        self.n_listeners = n_listeners
        self.spec_res = spec_res
        self.itd_res = itd_res
        self.itd_skip_connection = itd_skip_connection
        self.conv_in = conv_in
        self.azimuth_calibration = azimuth_calibration

        # For random Fourier feature
        self.itd_scale = itd_scale
        self.rng = np.random.default_rng(0)

        loc_emb_in = 4 if emb_in == 3 else 6
        if azimuth_calibration:
            emb_in += 2
        bmat = scale * self.rng.normal(0.0, 1.0, (hidden_features // 2, emb_in * 2))
        self.bmat = torch.nn.Parameter(torch.tensor(bmat.astype(np.float32)), requires_grad=False)
        loc_bmat = scale * self.rng.normal(0.0, 1.0, (hidden_features // 2, loc_emb_in))
        self.loc_bmat = torch.nn.Parameter(torch.tensor(loc_bmat.astype(np.float32)), requires_grad=False)

        self.emb_mlp = MLP(hidden_features, hidden_features * 2, dropout, activation=activation)
        self.itd_net = nn.Sequential(
            MLP(hidden_features, hidden_features // 2, dropout, activation=activation),
            MLP(hidden_features // 2, 1, dropout=0.0, activation=itd_activation),
        )

        # Spec input and output layers
        layer = [
            nn.Conv1d(conv_in, hidden_features // 2, 3, stride=1, padding=1),
            nn.PReLU(),
        ]
        for _ in range(conv_layers - 2):
            layer += [
                nn.Conv1d(hidden_features // 2, hidden_features // 2, 5, stride=2, padding=2),
                nn.PReLU(),
            ]
        layer += [
            nn.Conv1d(hidden_features // 2, hidden_features, 5, stride=2, padding=2),
            nn.PReLU(),
        ]
        self.spec_enc = nn.Sequential(*layer)

        layer = [
            nn.ConvTranspose1d(hidden_features, hidden_features // 2, 6, stride=2, padding=2),
            nn.PReLU(),
        ]
        for _ in range(conv_layers - 2):
            layer += [
                nn.ConvTranspose1d(hidden_features // 2, hidden_features // 2, 6, stride=2, padding=2),
                nn.PReLU(),
            ]
        layer += [
            nn.ConvTranspose1d(hidden_features // 2, 2, 3, stride=1, padding=1),
        ]
        self.spec_dec = nn.Sequential(*layer)

        self.hidden_blocks = nn.ModuleList(
            [
                LSTMLoRATAC(hidden_features, dropout=dropout, rnn=rnn, activation=activation, norm=norm)
                for _ in range(hidden_layers)
            ]
        )

        self.spec_hidden_blocks = nn.ModuleList(
            [
                LoRAMLP4RANF(
                    hidden_features,
                    hidden_features,
                    dropout=dropout,
                    bias=True,
                    activation=activation,
                )
                for _ in range(spec_hidden_layers)
            ]
        )
        self.itd_hidden_blocks = nn.ModuleList(
            [
                LoRAMLP4RANF(
                    hidden_features,
                    hidden_features,
                    dropout=dropout,
                    bias=True,
                    activation=activation,
                )
                for n in range(itd_hidden_layers)
            ]
        )

        # For PEFT
        lorau, lorav = [], []

        for _ in range(hidden_layers + spec_hidden_layers + itd_hidden_layers):
            lorau.append(nn.Linear(n_listeners, hidden_features * lora_rank, bias=False))
            lorav.append(nn.Linear(n_listeners, hidden_features * lora_rank, bias=False))
            nn.init.zeros_(lorau[-1].weight)

        self.embed_layer_lorau = nn.ModuleList(lorau)
        self.embed_layer_lorav = nn.ModuleList(lorav)

    def _compute_uv(self, tgt_onehot, ret_onehot, idx, mode):
        if mode == "target":
            a = self.embed_layer_lorau[idx](tgt_onehot)
            b = self.embed_layer_lorav[idx](tgt_onehot)

        elif mode == "alter":
            a = self.embed_layer_lorau[idx](tgt_onehot)
            b = self.embed_layer_lorav[idx](ret_onehot)

        else:
            raise ValueError(f"Invalid mode: {mode}")

        if a.shape[-1] < self.hidden_features:
            u = torch.stack(a.split(1, -1), -1)
        else:
            u = torch.stack(a.split(self.hidden_features, -1), -1)

        if b.shape[-1] < self.hidden_features:
            v = torch.stack(b.split(1, -1), -1)
        else:
            v = torch.stack(b.split(self.hidden_features, -1), -1)
        return u, v

    def forward(self, ret_specs_db, ret_itds, ret_locs, tgt_loc, tgt_sidx, ret_sidxs):
        batch, nretrieval, nch, _ = ret_specs_db.shape

        tgt_onehot = F.one_hot(tgt_sidx.unsqueeze(-1), self.n_listeners)
        tgt_onehot = tgt_onehot.tile(1, nretrieval, 1).type(torch.float32)
        ret_onehot = F.one_hot(ret_sidxs, self.n_listeners).type(torch.float32)

        tgt_azimuth = tgt_loc[:, 0, None].tile(1, nretrieval)  # (Batch, 3) -> (Batch, K)
        tgt_elevation = tgt_loc[:, 1, None].tile(1, nretrieval)
        ret_itds_scaled = self.itd_scale * ret_itds

        emb = torch.concatenate([tgt_azimuth.unsqueeze(-1), tgt_elevation.unsqueeze(-1), ret_itds_scaled], -1)

        if self.azimuth_calibration:
            emb = torch.concatenate([emb, ret_locs[..., 0, None], ret_locs[..., 1, None]], -1)

        ave_itd = torch.mean(ret_itds[:, :, :1], 1)
        ave_spec = torch.mean(ret_specs_db[:, :, :2, :], 1)

        emb = torch.cat([emb.sin(), emb.cos()], -1)
        emb = torch.einsum("bkn,mn->bkm", emb, self.bmat)
        emb = torch.concatenate([emb.sin(), emb.cos()], axis=-1)
        x_embs = torch.split(self.emb_mlp(emb), self.hidden_features, dim=-1)

        x_spec = ret_specs_db[..., :-1].reshape(batch * nretrieval, nch, -1)
        x_spec = self.spec_enc(x_spec).reshape(batch, nretrieval, self.hidden_features, -1)
        x = torch.concatenate([x_embs[0][..., None], x_spec, x_embs[1][..., None]], -1).transpose(-1, -2)

        # Core-processing
        for n in range(self.hidden_layers):
            u, v = self._compute_uv(tgt_onehot, ret_onehot, n, "alter")
            x = x + self.hidden_blocks[n](x, u=u, v=v)

        x = torch.mean(x, dim=1, keepdims=False)
        x_itd1, x_itd2, x_spec = x[:, :1, :], x[:, -1:, :], x[:, 1:-1, :]

        # Spectra post-processing
        for n in range(self.spec_hidden_layers):
            u, v = self._compute_uv(tgt_onehot[:, :1, :], None, n + self.hidden_layers, "target")
            x_spec = self.spec_hidden_blocks[n](x_spec, u=u, v=v)

        estimate = self.spec_dec(x_spec.transpose(-1, -2))
        estimate = F.pad(estimate, (0, 1), mode="replicate")

        if self.spec_res:
            estimate = estimate + ave_spec

        # ITD post-processing
        x_itd = x_itd1 + x_itd2
        for n in range(self.itd_hidden_layers):
            m = n + self.hidden_layers + self.spec_hidden_layers
            u, v = self._compute_uv(tgt_onehot[:, :1, :], None, m, "target")
            x_itd = self.itd_hidden_blocks[n](x_itd, u=u, v=v)

        if self.itd_skip_connection:
            if self.conv_in == 2:
                locs = [tgt_loc[:, 0].sin(), tgt_loc[:, 0].cos(), tgt_loc[:, 1].sin(), tgt_loc[:, 1].cos()]
                loc_emb = torch.stack(locs, -1) @ self.loc_bmat.T
                loc_emb = torch.concatenate([loc_emb.sin(), loc_emb.cos()], axis=-1)
            else:
                loc_emb = torch.concatenate(
                    [tgt_azimuth[:, 0].unsqueeze(-1), tgt_elevation[:, 0].unsqueeze(-1), ret_itds_scaled[:, 0, -1:]], -1
                )
                loc_emb = torch.cat([loc_emb.sin(), loc_emb.cos()], -1) @ self.loc_bmat.T
                loc_emb = torch.concatenate([loc_emb.sin(), loc_emb.cos()], axis=-1)

            x_itd = x_itd[:, 0, :] + loc_emb
        else:
            x_itd = x_itd[:, 0, :]

        itd = self.itd_net(x_itd)

        if self.itd_res:
            itd = itd + ave_itd

        return estimate, itd
