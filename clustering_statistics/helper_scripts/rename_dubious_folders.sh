#!/bin/bash
# script to copy rename measurements folders. Known dubious mocks in particular.
# bash rename_dubious_folders.sh

# version=holi-v3-altmtl
version=glam-uchuu-v2-altmtl
base_dir=/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base/${version}/
mapfile -t mocks_list < dubious_${version}.txt
for imock in "${mocks_list[@]}"; do
    echo "Renaming $imock"
    mv $base_dir/mock$imock/ $base_dir/dubious_mock$imock
done