#!/usr/bin/env python3
"""
Split Stanford Online Products (SOP) dataset for CBML loss training.
SOP dataset comes with pre-defined train/test splits in Ebay_train.txt and Ebay_test.txt files.
"""

import os

sop_root = 'resource/datasets/Stanford_Online_Products/'
train_source = os.path.join(sop_root, 'Ebay_train.txt')
test_source = os.path.join(sop_root, 'Ebay_test.txt')
train_file = os.path.join(sop_root, 'train.txt')
test_file = os.path.join(sop_root, 'test.txt')


def process_sop_file(source_file, output_file):
    """Process SOP dataset file and convert to CBML format"""
    processed_data = []
    
    with open(source_file, 'r') as f:
        for line in f:
            # SOP format: class_id image_id image_path
            parts = line.strip().split()
            if len(parts) >= 3:
                class_id = parts[0]
                image_path = parts[2]  # Full path from root
                
                # Extract relative path from images directory
                if image_path.startswith('images/'):
                    relative_path = image_path[7:]  # Remove 'images/' prefix
                else:
                    relative_path = image_path
                
                # Convert class_id to integer (0-based for training)
                label = int(class_id) - 1
                
                processed_data.append((relative_path, label))
    
    # Write to output file
    with open(output_file, 'w') as f:
        for fname, label in processed_data:
            print(f"{fname},{label}", file=f)
    
    return processed_data


def main():
    print("Processing Stanford Online Products dataset...")
    
    # Process training data
    if os.path.exists(train_source):
        train_data = process_sop_file(train_source, train_file)
        print(f"Training: {len(train_data)} images from {len(set([label for _, label in train_data]))} classes")
    else:
        print(f"Warning: {train_source} not found!")
    
    # Process testing data
    if os.path.exists(test_source):
        test_data = process_sop_file(test_source, test_file)
        print(f"Testing: {len(test_data)} images from {len(set([label for _, label in test_data]))} classes")
    else:
        print(f"Warning: {test_source} not found!")
    
    print("SOP dataset split completed!")


if __name__ == '__main__':
    main()
