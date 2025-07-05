import os
from collections import defaultdict

# Try to use torchvision's Cars196 dataset implementation
try:
    from torchvision.datasets import StanfordCars
    USE_TORCHVISION = True
except ImportError:
    USE_TORCHVISION = False

# Try to use scipy for .mat files
try:
    import scipy.io as sio
    USE_SCIPY = True
except ImportError:
    USE_SCIPY = False

cars196_root = 'resource/datasets/Cars196/'
train_file = cars196_root + 'train.txt'
test_file = cars196_root + 'test.txt'

def load_cars_with_scipy():
    """Load cars metadata using scipy"""
    devkit_path = cars196_root + 'devkit/'
    
    # Load training annotations
    cars_train_annos = sio.loadmat(devkit_path + 'cars_train_annos.mat')
    annotations = cars_train_annos['annotations'][0]
    
    data = []
    for anno in annotations:
        fname = anno[5][0]  # filename
        class_id = anno[4][0][0]  # class (1-based)
        data.append((fname, class_id - 1))  # Convert to 0-based
    
    return data

def load_cars_with_torchvision():
    """Load cars metadata using torchvision"""
    # Use torchvision's built-in Cars196 dataset
    train_dataset = StanfordCars(root=cars196_root, split='train', download=False)
    
    data = []
    for i in range(len(train_dataset)):
        # Get the image path and label
        img_path = train_dataset._samples[i][0]
        label = train_dataset._samples[i][1]
        
        # Extract just the filename
        fname = os.path.basename(img_path)
        data.append((fname, label))
    
    return data

def create_manual_split():
    """Create a manual split based on available image files"""
    cars_train_dir = cars196_root + 'cars_train/'
    
    if not os.path.exists(cars_train_dir):
        print(f"Error: {cars_train_dir} does not exist.")
        print("Please make sure the Cars196 dataset is properly downloaded.")
        return None
    
    # Get all image files
    image_files = []
    for fname in os.listdir(cars_train_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(fname)
    
    if not image_files:
        print("No image files found in the cars_train directory.")
        return None
    
    # Sort to ensure consistent ordering
    image_files.sort()
    
    # Create a simple split based on filename patterns
    # Cars196 filenames typically follow a pattern that includes class info
    class_to_images = defaultdict(list)
    
    for fname in image_files:
        # Extract class from filename (assuming format like "00001.jpg" where first digits represent class)
        # This is a simplified approach - adjust based on actual filename pattern
        try:
            # Try to extract class from filename
            class_id = int(fname.split('_')[0]) if '_' in fname else int(fname.split('.')[0][:5])
            class_to_images[class_id].append(fname)
        except (ValueError, IndexError):
            # If we can't parse the class, assign to a default class
            class_to_images[0].append(fname)
    
    # Split classes: first half for training, second half for testing
    all_classes = sorted(class_to_images.keys())
    split_point = len(all_classes) // 2
    
    train_data = []
    test_data = []
    
    for i, class_id in enumerate(all_classes):
        if i < split_point:
            # Training classes
            for fname in class_to_images[class_id]:
                train_data.append((fname, i))
        else:
            # Test classes (relabel from 0)
            for fname in class_to_images[class_id]:
                test_data.append((fname, i - split_point))
    
    return train_data, test_data

def main():
    global USE_SCIPY, USE_TORCHVISION
    
    print("Processing Cars196 dataset...")
    
    train_data = []
    test_data = []
    
    # Try different approaches in order of preference
    if USE_SCIPY:
        print("Using scipy to load .mat files...")
        try:
            all_data = load_cars_with_scipy()
            
            # Split by class: first 98 classes for training, rest for testing
            class_to_images = defaultdict(list)
            for fname, class_id in all_data:
                class_to_images[class_id].append(fname)
            
            for class_id in range(196):
                if class_id < 98:  # First 98 classes for training
                    for fname in class_to_images[class_id]:
                        train_data.append((fname, class_id))
                else:  # Rest for testing, relabel from 0
                    for fname in class_to_images[class_id]:
                        test_data.append((fname, class_id - 98))
                        
        except Exception as e:
            print(f"Error loading with scipy: {e}")
            USE_SCIPY = False
    
    if USE_TORCHVISION and not train_data:
        print("Using torchvision to load dataset...")
        try:
            all_data = load_cars_with_torchvision()
            
            # Split by class: first 98 classes for training, rest for testing
            class_to_images = defaultdict(list)
            for fname, class_id in all_data:
                class_to_images[class_id].append(fname)
            
            for class_id in range(196):
                if class_id < 98:  # First 98 classes for training
                    for fname in class_to_images[class_id]:
                        train_data.append((fname, class_id))
                else:  # Rest for testing, relabel from 0
                    for fname in class_to_images[class_id]:
                        test_data.append((fname, class_id - 98))
                        
        except Exception as e:
            print(f"Error loading with torchvision: {e}")
            USE_TORCHVISION = False
    
    if not train_data:
        print("Falling back to manual split...")
        result = create_manual_split()
        if result:
            train_data, test_data = result
    
    if not train_data:
        print("Error: Could not load Cars196 dataset.")
        print("Please ensure:")
        print("1. The dataset is properly downloaded")
        print("2. Install scipy: pip install scipy")
        print("3. Or use torchvision's built-in Cars196 dataset")
        return
    
    # Write training file
    with open(train_file, 'w') as f:
        for fname, label in train_data:
            print(f"cars_train/{fname},{label}", file=f)
    
    # Write test file
    with open(test_file, 'w') as f:
        for fname, label in test_data:
            print(f"cars_train/{fname},{label}", file=f)
    
    print(f"Created {train_file} with {len(train_data)} training samples")
    print(f"Created {test_file} with {len(test_data)} test samples")
    print(f"Training classes: 0-{len(set(label for _, label in train_data))-1}")
    print(f"Test classes: 0-{len(set(label for _, label in test_data))-1}")

if __name__ == '__main__':
    main()