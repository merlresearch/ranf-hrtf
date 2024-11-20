#!/usr/bin/env bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -e
set -u
set -o pipefail

stage=1
stop_stage=3

original_path="YOUR_SONICOM_DATASET_PATH/"
preprocessed_dataset_path="PATH_TO_STORE_PREPROCESSED_SONICOM_DATA"
exp_base_path="PATH_TO_STORE_LOG_FILES"
config_path="config_template/hrtf_selection"
sp_level=19
valid_size=19
test_size=20

dump_dir="${preprocessed_dataset_path}/sp_level_$(printf "%03d" "$sp_level")_no_azimuth_calibration"
sonicom_path="${preprocessed_dataset_path}/sonicom"
exp_path="${exp_base_path}/$(basename "$config_path")_splevel$(printf "%03d" "$sp_level")"


if [ $stage -le  1 ] && [ ${stop_stage} -ge 1 ]; then
    # Stage 1: Copying the required HRTF files to local and extracting features
    # This stage is required only once

    echo "Stage 1 ..."
    bash preprocess_sonicom.sh $original_path $sonicom_path
    python -m ranf.compute_spec_ild_itd_for_sonicom_datasets \
        "${sonicom_path}/subjects" "${sonicom_path}/npzs"
fi

if [ $stage -le  2 ] && [ ${stop_stage} -ge 2 ]; then
    # Stage 2: Splitting the dataset and writing the split into the configuration
    # `skip_78` excludes an outlier from the training dataset

    echo "Stage 2 ..."
    mkdir -p $dump_dir
    python -m ranf.compute_distance_matrices_for_spec_itd \
        "${sonicom_path}/npzs" $dump_dir $sp_level $test_size --skip_78

    python -m ranf.prepare_single_fold \
        $dump_dir $config_path $exp_path $sonicom_path $sp_level $valid_size $test_size --skip_78

    echo "The config file in $config_path has been modified for the specified sparsity level and saved in $exp_path"
fi

if [ $stage -le  3 ] && [ ${stop_stage} -ge 3 ]; then
    # Stage 3: Performing inference and evaluation on the test set
    echo "Stage 3 ..."
    if [ "$(echo $exp_path | grep 'hrtf_selection')" ]; then
        python -m ranf.evaluating_hrtf_selection $exp_path
    elif [ "$(echo $exp_path | grep 'nearest_neighbor')" ]; then
        python -m ranf.evaluating_nearest_neighbor $exp_path
    else
        echo "Invalid exp_path"
    fi
    python -m ranf.summarize_evaluation_result $exp_path
fi
