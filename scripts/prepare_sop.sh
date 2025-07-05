#!/bin/bash
set -e

# Prepare Stanford Online Products (SOP) dataset for CBML training
SOP_ROOT='resource/datasets/Stanford_Online_Products/'

echo "Preparing Stanford Online Products dataset..."

# Create directories
mkdir -p resource/datasets
pushd resource/datasets

# Download SOP dataset
if [ ! -d "Stanford_Online_Products" ]; then
    echo "Downloading Stanford Online Products dataset..."
    # Updated download link for SOP dataset
    gdown 1AtTQv-5ATXYHbVO-dVuQqoUqbVZT1auV
    
    # Extract files
    unzip Stanford_Online_Products.zip
    
    # Clean up
    rm Stanford_Online_Products.zip
fi

popd

echo "Stanford Online Products dataset prepared successfully!"

# Generate train.txt and test.txt splits
echo "Generating the train.txt/test.txt split files"
python scripts/split_sop_for_cbml_loss.py 