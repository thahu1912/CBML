#!/bin/bash
set -e


CARS196_ROOT='resource/datasets/Cars196/'
CARS196_DATA='https://www.dropbox.com/scl/fi/0a7bi36c55qb9bhm4pzzu/cars196.tar?rlkey=8rld9uelq39iw8r5moootq19g&dl=0'
CARS196_DEVKIT='https://drive.google.com/file/d/1t2CzZy3oEpZnb7vPwp68Z7AwyI43osuS/view?usp=drive_link'


if [[ ! -d "${CARS196_ROOT}" ]]; then
    mkdir -p resource/datasets
    pushd resource/datasets
    echo "Downloading Cars196 data-set..."
    wget ${CARS196_DATA}
    wget ${CARS196_DEVKIT}
    tar -xvf cars196.tar
    tar -xvf cars_devkit.tgz
    popd
fi

rm cars196.tar
rm cars_devkit.tgz



echo "Preparing Cars196 dataset using torchvision..."

python scripts/split_cars196_for_cbml_loss.py

echo "Cars196 dataset prepared successfully!"