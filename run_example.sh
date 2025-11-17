#!/usr/bin/env bash
# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -e
set -u
set -o pipefail

stage=1
stop_stage=5

original_path="YOUR_SONICOM_DATASET_PATH"
lap_challenge_path="YOUR_CHALLENGE_EVALUATION_SET_PATH"
preprocessed_dataset_path="PATH_TO_STORE_PREPROCESSED_SONICOM_DATA"
exp_base_path="PATH_TO_STORE_CHECKPOINTS_AND_LOG_FILES"
config_path="config_template/ranf"
sp_level=3
valid_size=9
maximum_subject_index=213

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
    # Stage 3: Pretraining a neural field
    echo "Stage 3 ..."
    python -m ranf.prepare_single_fold \
        $dump_dir $config_path $exp_path $sonicom_path $sp_level $valid_size $test_subjects --skip_78 --calibrate_itdoffset

    # We dump the results of the panning-based method because RANF+ takes them as auxiliary inputs.
    if [[ "${config_path}" == *ranf_plus ]]; then
        python -m ranf.sparse_vbap_dump_original $sonicom_path $sp_level $maximum_subject_index
        spvbap_feature_path="${dump_dir}/spvbap_features_original_$(printf "%03d" "$sp_level").npz"
        sed -i'' -E 's|^([[:space:]]*spvbap_path:).*|\1 "'"${spvbap_feature_path}"'"|g' "${exp_path}/config.yaml"
    fi

    python -m ranf.1_pretraining_neural_field $exp_path
fi

if [ $stage -le  4 ] && [ ${stop_stage} -ge 4 ]; then
    # Stage 4: Adapting the pre-trained neural field to the target subjects
    echo "Stage 4 ..."
    python -m ranf.2_adapting_neural_field $exp_path
fi


if [ $stage -le  5 ] && [ ${stop_stage} -ge 5 ]; then
    # Stage 5: Performing the inference and evaluation on the test set
    echo "Stage 5 ..."
    python -m ranf.3_evaluating_neural_field $exp_path
    python -m ranf.summarize_evaluation_result $exp_path
fi
