#!/bin/bash

echo "=========================================="
echo "Running CUB200-2011 with all backbones"
echo "=========================================="

# Function to run training and testing for a backbone
run_backbone() {
    local backbone=$1
    local config_file=$2
    local test_config_file=$3
    
    echo ""
    echo "------------------------------------------"
    echo "Running ${backbone} for CUB200-2011"
    echo "------------------------------------------"
    
    # Create output directory
    OUT_DIR="output-${backbone}-cub"
    if [[ ! -d "${OUT_DIR}" ]]; then
        echo "Creating output dir for training : ${OUT_DIR}"
        mkdir -p ${OUT_DIR}
    fi
    
    # Run training
    echo "Starting training with ${backbone}..."
    CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg ${config_file}
    
    # Run testing
    echo "Starting testing with ${backbone}..."
    CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg ${test_config_file} --phase test
    
    echo "Completed ${backbone} for CUB200-2011"
    echo ""
}

# Run all three backbones
echo "Starting ResNet50..."
run_backbone "resnet50" "configs/example_resnet50.yaml" "configs/example_resnet50_test.yaml"

echo "Starting GoogLeNet..."
run_backbone "googlenet" "configs/example_googlenet.yaml" "configs/example_googlenet_test.yaml"

echo "Starting BN-Inception..."
run_backbone "bninception" "configs/example_bninception.yaml" "configs/example_bninception_test.yaml"

echo "=========================================="
echo "All CUB200-2011 experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - output-resnet50-cub/"
echo "  - output-googlenet-cub/"
echo "  - output-bninception-cub/" 