import os
import torch
import torchvision
from torch.utils.data import Dataset
from cbml_benchmark.utils.img_reader import read_image


class Cars196Dataset(Dataset):
    """
    Cars196 Dataset using torchvision.datasets.StanfordCars
    Automatically downloads and manages the dataset
    """

    def __init__(self, root_dir, train=True, transforms=None, mode="RGB", download=True):
        self.root_dir = root_dir
        self.train = train
        self.transforms = transforms
        self.mode = mode
        
        # Use torchvision's StanfordCars dataset
        self.dataset = torchvision.datasets.StanfordCars(
            root=root_dir,
            split='train' if train else 'test',
            download=download
        )
        
        # Convert labels to string format for consistency with CBML framework
        self.path_list = []
        self.label_list = []
        self._process_data()
        
    def _process_data(self):
        """Process torchvision dataset to match CBML format"""
        for idx in range(len(self.dataset)):
            img, label = self.dataset[idx]
            
            # Convert label to string format (e.g., "001", "002", etc.)
            label_str = f"{label:03d}"
            
            # Create a virtual path for consistency
            img_path = f"cars196_{label_str}_{idx:06d}.jpg"
            
            self.path_list.append(img_path)
            self.label_list.append(label_str)
            
        # Store original dataset for image access
        self._original_dataset = self.dataset
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        # Get image and label from original torchvision dataset
        img, label = self._original_dataset[index]
        
        # Convert label to string format
        label_str = f"{label:03d}"
        
        # Apply transforms if provided
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, label_str
    
    def __str__(self):
        return f"Cars196 Dataset | train: {self.train} | datasize: {self.__len__()} | num_labels: {len(set(self.label_list))}" 