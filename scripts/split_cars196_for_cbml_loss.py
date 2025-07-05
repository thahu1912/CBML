#!/usr/bin/env python3
import os
import torchvision

cars196_root = 'resource/datasets/Cars196/'
train_file = os.path.join(cars196_root, 'train.txt')
test_file = os.path.join(cars196_root, 'test.txt')


def write_split(dataset, out_file):
    with open(out_file, 'w') as f:
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            label_str = f"{label:03d}"
            img_path = f"cars196_{label_str}_{idx:06d}.jpg"
            print(f"{img_path},{label_str}", file=f)


def main():
    # Train split
    train_dataset = torchvision.datasets.StanfordCars(
        root=cars196_root,
        split='train',
        download=True
    )
    write_split(train_dataset, train_file)
    print(f"Wrote {len(train_dataset)} samples to {train_file}")

    # Test split
    test_dataset = torchvision.datasets.StanfordCars(
        root=cars196_root,
        split='test',
        download=True
    )
    write_split(test_dataset, test_file)
    print(f"Wrote {len(test_dataset)} samples to {test_file}")

if __name__ == '__main__':
    main() 