#!/usr/bin/env python3
"""
Simple script to start training with wandb using just a project name.
This automatically creates the config file and starts training.

Usage:
    python scripts/train_with_wandb.py --project "my-experiment"
    python scripts/train_with_wandb.py --project "cbml-cub-resnet50" --dataset cub --phase test
    python scripts/train_with_wandb.py --project "cbml-cars196-resnet50" --dataset cars196
"""

import argparse
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.main import create_config_from_project, train, test
from cbml_benchmark.config import cfg


def main():
    parser = argparse.ArgumentParser(description='Start CBML training with wandb')
    parser.add_argument(
        '--project',
        required=True,
        help='wandb project name (will auto-generate config)'
    )
    parser.add_argument(
        '--dataset',
        choices=['cub', 'cars196', 'sop'],
        default='cub',
        help='dataset to use (cub, cars196, sop)'
    )
    parser.add_argument(
        '--phase',
        choices=['train', 'test'],
        default='train',
        help='train or test'
    )
    parser.add_argument(
        '--entity',
        help='wandb username or team name (optional)'
    )
    
    args = parser.parse_args()
    
    print(f"Starting CBML training with project: {args.project}")
    print(f"Dataset: {args.dataset.upper()}")
    
    # Create config file
    config_path = create_config_from_project(args.project, args.dataset)
    
    # Load config
    cfg.merge_from_file(config_path)
    
    # Set entity if provided
    if args.entity:
        cfg.WANDB.ENTITY = args.entity
    
    print(f"Config file: {config_path}")
    print(f"Phase: {args.phase}")
    print(f"Wandb project: {cfg.WANDB.PROJECT_NAME}")
    
    # Start training or testing
    if args.phase == 'train':
        train(cfg)
    else:
        test(cfg)
    
    print("Training completed!")


if __name__ == '__main__':
    main() 