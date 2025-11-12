import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random
import numpy as np

class PKSampler(Sampler):
    """
    Randomly samples N identities, then samples K instances for each identity.
    Total batch size = P * K
    
    Args:
    - dataset: Assumes a dataset (like ImageFolder) has a 'targets' attribute.
    - p: Number of identities (classes) per batch.
    - k: Number of instances (images) per identity.
    """
    def __init__(self, dataset, p, k, num_replicas=1, rank=0):
        super().__init__(dataset)
        self.dataset = dataset
        self.p = p
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        # Get all class IDs
        self.all_targets = np.array(self.dataset.targets)
        self.unique_classes = sorted(np.unique(self.all_targets))
        
        # Build a map from class_id -> list of indices
        self.class_to_indices = defaultdict(list)
        for idx, target in enumerate(self.all_targets):
            self.class_to_indices[target].append(idx)
            
        # Filter out classes with less than K images
        self.valid_classes = [
            cls for cls in self.unique_classes 
            if len(self.class_to_indices[cls]) >= self.k
        ]
        
        # Total number of classes for this rank
        self.num_classes_per_rank = len(self.valid_classes) // self.num_replicas
        
        # Approximate number of batches per epoch
        self.num_batches = self.num_classes_per_rank // self.p

    def __iter__(self):
        # Set seed for this epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        # Shuffle valid classes
        indices = torch.randperm(len(self.valid_classes), generator=g).tolist()
        
        # Get class indices for this rank
        rank_indices = indices[
            self.rank * self.num_classes_per_rank : (self.rank + 1) * self.num_classes_per_rank
        ]
        
        batch = []
        # Group classes into batches of size P
        for i in range(0, len(rank_indices) - self.p + 1, self.p):
            p_classes = rank_indices[i : i + self.p]
            
            for cls_idx in p_classes:
                class_id = self.valid_classes[cls_idx]
                
                # Get all indices for this class
                all_class_indices = self.class_to_indices[class_id]
                
                # Sample K indices from this class
                # We use torch.randperm for fast sampling
                perm = torch.randperm(len(all_class_indices), generator=g)
                selected_indices = perm[:self.k].tolist()
                
                for idx in selected_indices:
                    batch.append(all_class_indices[idx])
            
            yield batch
            batch = []

    def __len__(self):
        return self.num_batches
        
    def set_epoch(self, epoch):
        self.epoch = epoch