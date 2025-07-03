#!/bin/bash

# Prepare Stanford Online Products (SOP) dataset for CBML training
SOP_ROOT='resource/datasets/Stanford_Online_Products/'

echo "Preparing Stanford Online Products dataset..."

# Create directories
mkdir -p resource/datasets
pushd resource/datasets

# Download SOP dataset
if [ ! -d "Stanford_Online_Products" ]; then
    echo "Downloading Stanford Online Products dataset..."
    wget ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
    
    # Extract files
    unzip Stanford_Online_Products.zip
    
    # Clean up
    rm Stanford_Online_Products.zip
fi

popd

echo "Stanford Online Products dataset prepared successfully!"
echo "Please run the data splitting script to generate train.txt and test.txt files." 