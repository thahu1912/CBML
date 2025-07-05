import os
import torch
from torch.utils.data import Dataset
from cbml_benchmark.utils.img_reader import read_image


class CUB200Dataset(Dataset):
    """
    CUB200-2011 Dataset
    Manual download required: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    Expected directory structure:
    CUB_200_2011/
    ├── images/
    │   ├── 001.Black_footed_Albatross/
    │   ├── 002.Laysan_Albatross/
    │   └── ...
    ├── images.txt
    ├── train_test_split.txt
    └── classes.txt
    """

    def __init__(self, root_dir, train=True, transforms=None, mode="RGB"):
        self.root_dir = root_dir
        self.train = train
        self.transforms = transforms
        self.mode = mode
        
        self.images_dir = os.path.join(root_dir, "images")
        self.images_txt = os.path.join(root_dir, "images.txt")
        self.train_test_split_txt = os.path.join(root_dir, "train_test_split.txt")
        
        assert os.path.exists(self.images_dir), f"Images directory not found: {self.images_dir}"
        assert os.path.exists(self.images_txt), f"images.txt not found: {self.images_txt}"
        assert os.path.exists(self.train_test_split_txt), f"train_test_split.txt not found: {self.train_test_split_txt}"
        
        self.path_list = []
        self.label_list = []
        self._load_data()
        
    def _load_data(self):
        """Load image paths and labels from CUB200 dataset files"""
        # Read image paths and class labels
        image_data = {}
        with open(self.images_txt, 'r') as f:
            for line in f:
                img_id, img_path = line.strip().split(' ', 1)
                image_data[img_id] = img_path
        
        # Read train/test split
        split_data = {}
        with open(self.train_test_split_txt, 'r') as f:
            for line in f:
                img_id, is_training = line.strip().split(' ', 1)
                split_data[img_id] = int(is_training)
        
        # Filter images based on train/test split
        for img_id, img_path in image_data.items():
            if split_data[img_id] == 1 and self.train:  # Training image
                self.path_list.append(img_path)
                # Extract class label from path (e.g., "001.Black_footed_Albatross" -> "001")
                class_label = img_path.split('/')[0].split('.')[0]
                self.label_list.append(class_label)
            elif split_data[img_id] == 0 and not self.train:  # Test image
                self.path_list.append(img_path)
                class_label = img_path.split('/')[0].split('.')[0]
                self.label_list.append(class_label)
    
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
        return f"CUB200 Dataset | train: {self.train} | datasize: {self.__len__()} | num_labels: {len(set(self.label_list))}" 