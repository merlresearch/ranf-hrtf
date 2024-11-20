<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Retrieval-Augmented Neural Field for HRTF Upsampling and Personalization

This repository includes source code for training and evaluating the retrieval-augmented neural field (RANF) proposed in the following ICASSP 2025 submission:

    @InProceedings{Masuyama2024ICASSP_ranf,
      author    =  {Masuyama, Yoshiki and Wichern, Gordon and Germain, Fran\c{c}ois G. and Ick, Christopher and {Le Roux}, Jonathan},
      title     =  {Retrieval-Augmented Neural Field for HRTF Upsampling and Personalization},
      booktitle =  {Submitted to IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      year      =  2025,
      month     =  apr
    }

## Table of contents

1. [Environment setup](#environment-setup)
2. [Supported sparsity levels and models](#supported-sparsity-levels-and-models)
3. [Training and evaluating RANF](#training-and-evaluating-ranf)
4. [Evaluating learning-free baseline methods](#evaluating-learning-free-baseline-methods)
5. [Contributing](#contributing)
6. [Copyright and license](#copyright-and-license)

## Environment setup

The code has been tested using `python 3.10.0` on Linux.
Necessary dependencies can be installed using the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Supported sparsity levels and models
- Our HRTF upsampling experiments were performed on [the SONICOM dataset](https://www.sonicom.eu/tools-and-resources/hrtf-dataset/) that is [released under the MIT license](https://www.axdesign.co.uk/tools-and-devices/sonicom-hrtf-dataset).
- We performed HRTF upsampling with four sparsity levels following [Task 2 of the Listener Acoustic Personalization Challenge 2024](https://www.sonicom.eu/lap-challenge/). The number of measured directions, `sp_level` in `run_example.sh`, should be selected from `{3, 5, 19, 100}`, where smaller is more challenging to upsample.
- We currently support three NF-based methods. Please refer to our paper for their details.
  - NF with conditioning by concatenation (CbC): NF takes a subject-specific latent vector as an auxiliary input in addition to the sound source direction.
  - NF with low-rank adaptation (LoRA): The model weights will be updated by adding a subject-specific low-rank matrix.
  - RANF: NF takes HRTF magnitude and ITDs of the retrieved subjects in addition to the sound source direction. LoRA is also used to adapt the model.

## Training and evaluating RANF
In order to train and evaluate RANF and the existing NF-based methods on the SONICOM dataset, please execute `run_example.sh` after following Stage 0. Then, `run_example.sh` consists of five stages. You can run each stage one by one by changing `stage` and `stop_stage` in the script.


- **Stage 0:**
   - Before starting the training and evaluation, download the SONICOM dataset into a directory specified in `original_path` in `run_example.sh` and unzip the dataset.
      - The directory is assumed to contain `KEMAR`, `P0001-P0005`, ..., `P0196-P0200`.
      - If you find `P0050_FreeFieldCompMinPhase_48kHz.sofa` instead of `P0051_FreeFieldCompMinPhase_48kHz.sofa` in `$original_path/P0051-P0055/P0051/HRTF/HRTF/48kHz`, please copy it as follows:
      ```bash
      cp $original_path/P0051-P0055/P0051/HRTF/HRTF/48kHz/P0050_FreeFieldCompMinPhase_48kHz.sofa $original_path/P0051-P0055/P0051/HRTF/HRTF/48kHz/P0051_FreeFieldCompMinPhase_48kHz.sofa
      ```
   - `preprocessed_dataset_path` should be specified to save the preprocessed SONICOM dataset.
   - Model checkpoints and log files will be stored in subdirectories under `exp_base_path`
   - You can select a model and a sparsity level by `config_path` and `sp_level`, respectively.

- **Stage 1:**
   - This stage extracts features (spectra and ITDs) and stores the features as `$sonicom_path/npzs/features_and_locs_wo_azimuth_calibration.npz`
   - This stage is required only once, and you can start from Stage 2 if you want to train a new model.

- **Stage 2:**
   - This stage splits the datasets (train, valid, and test), where the option `--skip_78` enforces that the training set excludes a subject with atypical ITD measurements.
   - Distance matrices between subjects in terms of the spectra and ITDs are computed based on the measured HRTFs, where the number of measurements depends on the given sparsity level.
   - The configuration file in `$config_path/original_config.yaml` will be modified based on the sparsity level and the data split, and then the updated cconfiguration file will be saved in `$exp_path/config.yaml`.

- **Stage 3:**
   - This stage trains the model specified by `$exp_path/original_config.yaml` on the multi-subject training dataset.
   - The log file will be stored in `$exp_path/log/exp.log`, while the checkpoint with the best validation loss will be `$exp_path/best.ckpt`

- **Stage 4:**
   - This stage adapts the pre-trained model to the target subject by fine-tuning a few parameters in the model.
   - The log file will be stored in `$exp_path/log/adaptation/adaptation.log`, while the checkpoint with the best adaptation loss will be `$exp_path/adaptation.ckpt`
   - Currently, this stage simultaneously optimizes the subject-specific parameters of all target subjects since the parameters of each subject are independent of other subjects.

- **Stage 5:**
   - This stage runs inference and evaluates the results.
   - The metrics used in the LAP challenge for each subject will be in `$exp_path/log/eval/eval.log`, and the summarized result will be shown in the CLI.
   - We note that the performance may vary from the results reported in the paper depending on your specific environment, especially when `sp_level = 3`, and we used Pytorch 1.13.0 for the paper while the current default in `requirements.txt` is 2.2.2.


## Evaluating learning-free baseline methods
In order to evaluate the learning-free methods, HRTF selection and nearest neighbor neighbor, please execute `run_learningfree_methods.sh` after specifying the paths as explained for RANF above. Stages 1 and 2 are the same as for RANF, and both inference and evaluation are performed in Stage 3.


## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.


## Copyright and license
Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:
```
Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```
