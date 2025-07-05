#!/usr/bin/env python3

import os

sop_root = 'resource/datasets/Stanford_Online_Products/'
train_file = sop_root + 'train.txt'
test_file = sop_root + 'test.txt'

def main():
    train = []
    test = []
    
    # Read Ebay_train.txt
    train_txt = os.path.join(sop_root, 'Ebay_train.txt')
    if os.path.exists(train_txt):
        with open(train_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    class_id = int(parts[0]) - 1  # Convert to 0-based indexing
                    image_path = parts[2]  # Full path from root
                    
                    # Extract relative path from images directory
                    if image_path.startswith('images/'):
                        relative_path = image_path[7:]  # Remove 'images/' prefix
                    else:
                        relative_path = image_path
                    
                    train.append((relative_path, class_id))
    
    # Read Ebay_test.txt
    test_txt = os.path.join(sop_root, 'Ebay_test.txt')
    if os.path.exists(test_txt):
        with open(test_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    class_id = int(parts[0]) - 1  # Convert to 0-based indexing
                    image_path = parts[2]  # Full path from root
                    
                    # Extract relative path from images directory
                    if image_path.startswith('images/'):
                        relative_path = image_path[7:]  # Remove 'images/' prefix
                    else:
                        relative_path = image_path
                    
                    test.append((relative_path, class_id))
    
    # Write train.txt
    with open(train_file, 'w') as tf:
        for fname, label in train:
            print("{},{}".format(fname, label), file=tf)
    
    # Write test.txt
    with open(test_file, 'w') as tf:
        for fname, label in test:
            print("{},{}".format(fname, label), file=tf)
    
    print(f"SOP split completed:")
    print(f"Training samples: {len(train)}")
    print(f"Testing samples: {len(test)}")

if __name__ == '__main__':
    main() 