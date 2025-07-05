#!/bin/bash
set -e

# Prepare Stanford Online Products (SOP) dataset for CBML training
SOP_ROOT='resource/datasets/Stanford_Online_Products/'
SOP_DATA='https://www.dropbox.com/scl/fi/7icj466ds04ex7rd7kxxs/online_products.tar?rlkey=c2tp644h3uzui38tpu3l8z2uq&e=1&dl=0'

echo "Preparing Stanford Online Products dataset..."

# Create directories
mkdir -p resource/datasets
pushd resource/datasets

# Download SOP dataset
if [ ! -d "Stanford_Online_Products" ]; then
    echo "Downloading Stanford Online Products dataset..."
    # Updated download link for SOP dataset
    wget ${SOP_DATA}
    
    # Extract files
    tar -xvf online_products.tar
    
    # Clean up
    rm online_products.tar
fi

popd

echo "Stanford Online Products dataset prepared successfully!"

# Generate train.txt and test.txt splits
echo "Generating the train.txt/test.txt split files"
python scripts/split_sop_for_cbml_loss.py 