#!/usr/bin/env python3
import os
import sys
from cbml_benchmark.data.datasets.cars196 import Cars196Dataset

cars196_root = 'resource/datasets/Cars196/'
train_file = os.path.join(cars196_root, 'train.txt')
test_file = os.path.join(cars196_root, 'test.txt')


def write_split(dataset, out_file):
    with open(out_file, 'w') as f:
        for idx in range(len(dataset)):
            # The dataset __getitem__ returns (img, label), but we want the path and label
            # We'll reconstruct the path as in the dataset logic
            label = dataset.label_list[idx]
            img_path = dataset.path_list[idx]
            print(f"{img_path},{label}", file=f)


def main():
    # Train split
    train_dataset = Cars196Dataset(root_dir=cars196_root, train=True, transforms=None, download=True)
    write_split(train_dataset, train_file)
    print(f"Wrote {len(train_dataset)} samples to {train_file}")

    # Test split
    test_dataset = Cars196Dataset(root_dir=cars196_root, train=False, transforms=None, download=True)
    write_split(test_dataset, test_file)
    print(f"Wrote {len(test_dataset)} samples to {test_file}")

if __name__ == '__main__':
    main() 