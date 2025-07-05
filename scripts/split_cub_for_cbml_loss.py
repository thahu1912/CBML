#!/usr/bin/env python3
"""
Split CUB200-2011 dataset for CBML loss training.
CUB200 dataset has separate files for images, labels, and train/test splits.
"""

import os

cub_root = 'resource/datasets/CUB_200_2011/'
images_txt = os.path.join(cub_root, 'images.txt')
labels_txt = os.path.join(cub_root, 'image_class_labels.txt')
split_txt = os.path.join(cub_root, 'train_test_split.txt')
train_file = os.path.join(cub_root, 'train.txt')
test_file = os.path.join(cub_root, 'test.txt')


def main():
    # Read image paths
    with open(images_txt) as f:
        image_id_to_path = dict(line.strip().split() for line in f)
    
    # Read class labels
    with open(labels_txt) as f:
        image_id_to_label = dict(line.strip().split() for line in f)
    
    # Read train/test splits
    with open(split_txt) as f:
        image_id_to_split = dict(line.strip().split() for line in f)
    
    train_lines = []
    test_lines = []
    
    for image_id in image_id_to_path:
        path = image_id_to_path[image_id]
        label = int(image_id_to_label[image_id]) - 1  # 0-based label
        is_train = int(image_id_to_split[image_id])
        
        line = f"{path},{label}\n"
        if is_train == 1:
            train_lines.append(line)
        else:
            test_lines.append(line)
    
    # Save train/test splits
    with open(train_file, "w") as f:
        f.writelines(train_lines)
    
    with open(test_file, "w") as f:
        f.writelines(test_lines)
    
    print("CUB200-2011 split completed!")
    print(f"  Training: {len(train_lines)} images")
    print(f"  Testing: {len(test_lines)} images")


if __name__ == '__main__':
    main()
