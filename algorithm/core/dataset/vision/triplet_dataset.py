import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets.folder import default_loader
import random
from typing import Callable, Optional

class TripleFaceDataset(Dataset):
    """
    A dataset class for triplet loss, compatible with ImageFolder structure.
    It returns (anchor_img, positive_img, negative_img)
    """
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform
        self.loader = default_loader
        
        # Use torchvision.datasets.ImageFolder to parse the directory
        # We don't need its __getitem__, just its internal lists
        img_folder = datasets.ImageFolder(root)
        
        self.samples = img_folder.samples  # List of (filepath, class_idx)
        self.targets = img_folder.targets  # List of class_idx for each sample
        self.classes = img_folder.classes  # List of class names
        
        # Build a map for fast lookup: class_idx -> list of sample_indices
        # This is the most important part for triplet sampling
        self.class_to_indices = {}
        for idx, target in enumerate(self.targets):
            if target not in self.class_to_indices:
                self.class_to_indices[target] = []
            self.class_to_indices[target].append(idx)
            
        # We need a list of all *class indices* to pick a negative class
        self.class_indices_list = list(self.class_to_indices.keys())
        
        # Ensure all classes have at least 2 images, otherwise sampling will fail
        for class_idx, indices in self.class_to_indices.items():
            if len(indices) < 2:
                print(f"Warning: Class {self.classes[class_idx]} (ID: {class_idx}) has "
                      f"only {len(indices)} images. This may cause issues "
                      "with positive sampling.")
    def __len__(self):
        # The length of the dataset is the total number of images (anchors)
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the anchor image.
        Returns:
            tuple: (anchor_img, positive_img, negative_img)
        """
        
        # 1. Get Anchor
        anchor_path, anchor_class_idx = self.samples[index]
        
        # 2. Get Positive
        # Get all indices for the same class
        positive_indices = self.class_to_indices[anchor_class_idx]
        
        positive_idx = index
        # Loop to ensure the positive image is *not* the same as the anchor image
        # (This is safe since you have 50 images per class)
        while positive_idx == index:
            positive_idx = random.choice(positive_indices)
        
        positive_path, _ = self.samples[positive_idx]

        # 3. Get Negative
        negative_class_idx = anchor_class_idx
        # Loop to ensure the negative class is *not* the same as the anchor class
        while negative_class_idx == anchor_class_idx:
            negative_class_idx = random.choice(self.class_indices_list)
        
        # Pick a random image index from that different class
        negative_indices = self.class_to_indices[negative_class_idx]
        negative_idx = random.choice(negative_indices)
        
        negative_path, _ = self.samples[negative_idx]

        # 4. Load images
        try:
            anchor_img = self.loader(anchor_path)
            positive_img = self.loader(positive_path)
            negative_img = self.loader(negative_path)
        except Exception as e:
            print(f"Error loading images for index {index}:")
            print(f"  Anchor: {anchor_path}")
            print(f"  Positive: {positive_path}")
            print(f"  Negative: {negative_path}")
            print(f"Error: {e}")
            # Return a fallback (e.g., the anchor 3 times) to prevent a crash
            # You might want to handle this more gracefully
            anchor_img = self.loader(anchor_path)
            positive_img = anchor_img
            negative_img = anchor_img


        # 5. Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return anchor_img, positive_img, negative_img