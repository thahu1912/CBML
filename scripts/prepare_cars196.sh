#!/bin/bash
set -e

echo "Preparing Cars196 dataset using torchvision..."

python scripts/split_cars196_for_cbml_loss.py

echo "Cars196 dataset prepared successfully!"