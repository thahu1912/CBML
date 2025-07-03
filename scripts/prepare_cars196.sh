#!/bin/bash

# Prepare Cars196 dataset for CBML training
CARS196_ROOT='resource/datasets/Cars196/'

echo "Preparing Cars196 dataset..."

# Create directories
mkdir -p resource/datasets
pushd resource/datasets

# Download Cars196 dataset
if [ ! -d "Cars196" ]; then
    echo "Downloading Cars196 dataset..."
    wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
    wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
    wget http://imagenet.stanford.edu/internal/car196/car_devkit.tgz
    
    # Extract files
    tar -xzf cars_train.tgz
    tar -xzf cars_test.tgz
    tar -xzf car_devkit.tgz
    
    # Create Cars196 directory structure
    mkdir -p Cars196
    mv cars_train Cars196/
    mv cars_test Cars196/
    mv devkit Cars196/
    
    # Clean up
    rm cars_train.tgz cars_test.tgz car_devkit.tgz
fi

popd

echo "Cars196 dataset prepared successfully!"
echo "Please run the data splitting script to generate train.txt and test.txt files." 