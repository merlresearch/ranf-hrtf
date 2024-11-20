#!/usr/bin/env bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

original_path=$1
sonicom_path=$2

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
