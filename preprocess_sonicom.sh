#!/usr/bin/env bash
# Copyright (C) 2024-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

original_path=$1
lap_challenge_path=$2
sonicom_path=$3

mkdir -p "${sonicom_path}/kemar"
mkdir -p "${sonicom_path}/myhrtf"
mkdir -p "${sonicom_path}/subjects"

fnames=`find -L $original_path -type f -name *_FreeFieldCompMinPhase_48kHz.sofa`

for fname in $fnames; do
    fbase=`basename "$fname"`

    if [ "`echo $fname | grep 'KEMAR'`" ]; then
        cp -n $fname "${sonicom_path}/kemar/${fbase}"

    elif [ "`echo $fname | grep 'MyHRTF'`" ]; then
        cp -n $fname "${sonicom_path}/myhrtf/${fbase}"

    else
        cp -n $fname "${sonicom_path}/subjects/${fbase}"
    fi
done

fnames=`find -L $lap_challenge_path -type f -name *_FreeFieldCompMinPhase_48kHz.sofa`
for fname in $fnames; do
    fbase=`basename "$fname"`
    cp -n $fname "${sonicom_path}/subjects/${fbase}"
done
