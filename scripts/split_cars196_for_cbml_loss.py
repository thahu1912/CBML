#!/usr/bin/env python3
"""
Split Cars196 dataset for CBML loss training.
Cars196 has 196 classes, typically split as:
- Training: classes 1-98 (98 classes)
- Testing: classes 99-196 (98 classes)
"""

import os
import glob

cars_root = 'resource/datasets/Cars196/'
images_dir = os.path.join(cars_root, 'images')
train_file = os.path.join(cars_root, 'train.txt')
test_file = os.path.join(cars_root, 'test.txt')


def main():
    train = []
    test = []
    
    # Get all class directories
    class_dirs = sorted(glob.glob(os.path.join(images_dir, '*')))
    
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        # Extract class number from directory name (e.g., "001" from "001.BMW_1_Series")
        class_num = int(class_name.split('.')[0])
        
        # Get all images in this class
        image_files = glob.glob(os.path.join(class_dir, '*.jpg'))
        image_files.extend(glob.glob(os.path.join(class_dir, '*.jpeg')))
        image_files.extend(glob.glob(os.path.join(class_dir, '*.png')))
        
        for img_path in image_files:
            # Get relative path from images directory
            rel_path = os.path.relpath(img_path, images_dir)
            # Convert to forward slashes for consistency
            rel_path = rel_path.replace('\\', '/')
            
            # Cars196 uses 0-based labels for training
            label = class_num - 1
            
            if class_num <= 98:
                train.append((rel_path, label))  # Training: classes 1-98 (labels 0-97)
            else:
                test.append((rel_path, label))   # Testing: classes 99-196 (labels 98-195)

    # Write train.txt
    with open(train_file, 'w') as tf:
        for fname, label in train:
            print(f"{fname},{label}", file=tf)
    
    # Write test.txt
    with open(test_file, 'w') as tf:
        for fname, label in test:
            print(f"{fname},{label}", file=tf)
    
    print(f"Cars196 split completed:")
    print(f"  Training: {len(train)} images from {len(set([label for _, label in train]))} classes")
    print(f"  Testing: {len(test)} images from {len(set([label for _, label in test]))} classes")


if __name__ == '__main__':
    main()
