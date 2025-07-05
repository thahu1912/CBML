#!/bin/bash
set -e

CUB_ROOT='resource/datasets/CUB_200_2011/'
CUB_DATA='https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'


if [[ ! -d "${CUB_ROOT}" ]]; then
    mkdir -p resource/datasets
    pushd resource/datasets
    echo "Downloading CUB_200_2011 data-set..."
    wget ${CUB_DATA}
    tar -zxf CUB_200_2011.tgz
    rm CUB_200_2011.tgz
    popd
fi
# Generate train.txt and test.txt splits
echo "Generating the train.txt/test.txt split files"
python scripts/split_cub_for_cbml_loss.py


