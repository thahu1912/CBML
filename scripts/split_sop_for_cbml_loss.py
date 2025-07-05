import os
from collections import defaultdict

sop_root = 'resource/datasets/Stanford_Online_Products/'
ebay_train_file = sop_root + 'Ebay_train.txt'
ebay_test_file = sop_root + 'Ebay_test.txt'
train_file = sop_root + 'train.txt'
test_file = sop_root + 'test.txt'

def load_sop_data(file_path):
    """Load SOP data from Ebay_train.txt or Ebay_test.txt"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                class_id = int(parts[0])
                image_id = int(parts[1])
                image_path = parts[2]
                data.append((class_id, image_id, image_path))
    return data

def main():
    print("Processing Stanford Online Products dataset...")
    
    # Load training and test data
    train_data_raw = load_sop_data(ebay_train_file)
    test_data_raw = load_sop_data(ebay_test_file)
    
    # Get unique classes from training data
    train_classes = set(item[0] for item in train_data_raw)
    test_classes = set(item[0] for item in test_data_raw)
    
    # Create class mapping (original class_id -> 0-based label)
    train_class_mapping = {cls: idx for idx, cls in enumerate(sorted(train_classes))}
    test_class_mapping = {cls: idx for idx, cls in enumerate(sorted(test_classes))}
    
    # Process training data
    train_data = []
    for class_id, image_id, image_path in train_data_raw:
        # Remove 'images/' prefix if present
        if image_path.startswith('images/'):
            image_path = image_path[7:]
        
        new_label = train_class_mapping[class_id]
        train_data.append((image_path, new_label))
    
    # Process test data
    test_data = []
    for class_id, image_id, image_path in test_data_raw:
        # Remove 'images/' prefix if present
        if image_path.startswith('images/'):
            image_path = image_path[7:]
        
        new_label = test_class_mapping[class_id]
        test_data.append((image_path, new_label))
    
    # Write training file
    with open(train_file, 'w') as f:
        for image_path, label in train_data:
            print(f"images/{image_path},{label}", file=f)
    
    # Write test file
    with open(test_file, 'w') as f:
        for image_path, label in test_data:
            print(f"images/{image_path},{label}", file=f)
    
    print(f"Created {train_file} with {len(train_data)} training samples")
    print(f"Created {test_file} with {len(test_data)} test samples")
    print(f"Training classes: {len(train_classes)} (0-{len(train_classes)-1})")
    print(f"Test classes: {len(test_classes)} (0-{len(test_classes)-1})")

if __name__ == '__main__':
    main()