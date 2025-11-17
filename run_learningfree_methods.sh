#!/usr/bin/env bash
# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -e
set -u
set -o pipefail

stage=1
stop_stage=3

original_path="YOUR_SONICOM_DATASET_PATH"
lap_challenge_path="YOUR_CHALLENGE_EVALUATION_SET_PATH"
preprocessed_dataset_path="PATH_TO_STORE_PREPROCESSED_SONICOM_DATA"
exp_base_path="PATH_TO_STORE_CHECKPOINTS_AND_LOG_FILES"
config_path="config_template/hrtf_selection"
sp_level=3
valid_size=9

dump_dir="${preprocessed_dataset_path}/sp_level_$(printf "%03d" "$sp_level")"
sonicom_path="${preprocessed_dataset_path}/sonicom"
exp_path="${exp_base_path}/$(basename "$config_path")_splevel_$(printf "%03d" "$sp_level")"

# The test subjects are the same as the ones for the LAP challenge.
if [ $sp_level -eq 3 ]; then
  test_subjects="204,208,213"
elif [ $sp_level -eq 5 ]; then
  test_subjects="203,207,212"
elif [ $sp_level -eq 19 ]; then
  test_subjects="202,206,211"
elif [ $sp_level -eq 100 ]; then
  test_subjects="201,205,210"
else
  echo "sp_level should be one of {3, 5, 19, 100} but the given value is ${sp_level}."
  exit 1
fi

if [ $stage -le  1 ] && [ ${stop_stage} -ge 1 ]; then
    # Stage 1: Copying the required HRTF files to local and extracting features
    # This stage is required only once regardless of the sparsity levels

    echo "Stage 1 ..."
    bash preprocess_sonicom.sh $original_path $lap_challenge_path $sonicom_path
fi

if [ $stage -le  2 ] && [ ${stop_stage} -ge 2 ]; then
    # Stage 2: Splitting the dataset and writing the split into the configuration
    # This stage is required only once for each sparsity level
    # `skip_78` removes an outlier from the training dataset

    echo "Stage 2 ..."
    python -m ranf.compute_spec_ild_itd_for_sonicom_datasets \
        "${sonicom_path}/subjects" $dump_dir --upsample $sp_level --calibrate_itdoffset

    mkdir -p $dump_dir
    python -m ranf.compute_distance_matrices_for_spec_itd \
        $dump_dir $sp_level --skip_78 --calibrate_itdoffset
fi

if [ $stage -le  3 ] && [ ${stop_stage} -ge 3 ]; then
    # Stage 3: Performing inference and evaluation on the test set
    echo "Stage 3 ..."
    python -m ranf.prepare_single_fold \
        $dump_dir $config_path $exp_path $sonicom_path $sp_level $valid_size $test_subjects --skip_78 --calibrate_itdoffset

    if [ "$(echo $exp_path | grep 'hrtf_selection')" ]; then
        python -m ranf.evaluating_hrtf_selection $exp_path
    elif [ "$(echo $exp_path | grep 'nearest_neighbor')" ]; then
        python -m ranf.evaluating_nearest_neighbor $exp_path
    elif [ "$(echo $exp_path | grep 'sparse_vbap')" ]; then
        python -m ranf.evaluating_sparse_vbap $exp_path
    else
        echo "Invalid exp_path"
    fi
    python -m ranf.summarize_evaluation_result $exp_path
fi
