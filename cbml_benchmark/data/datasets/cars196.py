import os
import torch
from torch.utils.data import Dataset
from cbml_benchmark.utils.img_reader import read_image


class Cars196Dataset(Dataset):
    """
    Cars196 Dataset
    Manual download required: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    Expected directory structure:
    Cars196/
    ├── images/
    │   ├── 001.BMW_1_Series/
    │   │   ├── 001001.jpg
    │   │   ├── 001002.jpg
    │   │   └── ...
    │   ├── 002.BMW_3_Series/
    │   └── ...
    ├── train.txt
    └── test.txt
    """

    def __init__(self, root_dir, train=True, transforms=None, mode="RGB"):
        self.root_dir = root_dir
        self.train = train
        self.transforms = transforms
        self.mode = mode
        
        self.images_dir = os.path.join(root_dir, "images")
        self.train_txt = os.path.join(root_dir, "train.txt")
        self.test_txt = os.path.join(root_dir, "test.txt")
        
        assert os.path.exists(self.images_dir), f"Images directory not found: {self.images_dir}"
        assert os.path.exists(self.train_txt), f"train.txt not found: {self.train_txt}"
        assert os.path.exists(self.test_txt), f"test.txt not found: {self.test_txt}"
        
        self.path_list = []
        self.label_list = []
        self._load_data()
        
    def _load_data(self):
        """Load image paths and labels from Cars196 dataset files"""
        # Choose the appropriate text file based on train/test split
        txt_file = self.train_txt if self.train else self.test_txt
        
        with open(txt_file, 'r') as f:
            for line in f:
                # Cars196 format: image_path,label
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    image_path = parts[0]
                    label = int(parts[1])
                    
                    self.path_list.append(image_path)
                    self.label_list.append(label)
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.path_list[index])
        label = self.label_list[index]
        
        img = read_image(img_path, mode=self.mode)
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, label
    
    def __str__(self):
        return f"Cars196 Dataset | train: {self.train} | datasize: {self.__len__()} | num_labels: {len(set(self.label_list))}" 