#!/bin/bash
set -e


CUB_ROOT='resource/datasets/Cars196/'

echo "Preparing Cars196 dataset using torchvision..."

python scripts/split_cars196_for_cbml_loss.py

echo "Cars196 dataset prepared successfully!"