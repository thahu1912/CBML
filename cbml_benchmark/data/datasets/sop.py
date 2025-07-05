import os
import torch
from torch.utils.data import Dataset
from cbml_benchmark.utils.img_reader import read_image


class SOPDataset(Dataset):
    """
    Stanford Online Products (SOP) Dataset
    Manual download required: https://cvgl.stanford.edu/projects/lifted_struct/
    Expected directory structure:
    Stanford_Online_Products/
    ├── Ebay_train.txt
    ├── Ebay_test.txt
    └── images/
        ├── bicycle_final/
        │   ├── 111085122871_0.JPG
        │   ├── 111085122871_1.JPG
        │   └── ...
        ├── chair_final/
        └── ...
    """

    def __init__(self, root_dir, train=True, transforms=None, mode="RGB"):
        self.root_dir = root_dir
        self.train = train
        self.transforms = transforms
        self.mode = mode
        
        self.images_dir = os.path.join(root_dir, "images")
        self.train_txt = os.path.join(root_dir, "Ebay_train.txt")
        self.test_txt = os.path.join(root_dir, "Ebay_test.txt")
        
        assert os.path.exists(self.images_dir), f"Images directory not found: {self.images_dir}"
        assert os.path.exists(self.train_txt), f"Ebay_train.txt not found: {self.train_txt}"
        assert os.path.exists(self.test_txt), f"Ebay_test.txt not found: {self.test_txt}"
        
        self.path_list = []
        self.label_list = []
        self._load_data()
        
    def _load_data(self):
        """Load image paths and labels from SOP dataset files"""
        # Choose the appropriate text file based on train/test split
        txt_file = self.train_txt if self.train else self.test_txt
        
        with open(txt_file, 'r') as f:
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
                    
                    self.path_list.append(relative_path)
                    self.label_list.append(class_id)
    
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
        return f"SOP Dataset | train: {self.train} | datasize: {self.__len__()} | num_labels: {len(set(self.label_list))}" 