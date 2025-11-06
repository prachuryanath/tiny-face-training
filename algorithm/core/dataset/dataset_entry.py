from .vision import *
from ..utils.config import configs
from .vision.transform import *
import torchvision
import os

# Import the new sampler
try:
    from .pk_sampler import PKSampler
except ImportError:
    print("Warning: PKSampler not found. Make sure pk_sampler.py is in algorithm/core/dataset/")
    PKSampler = None


__all__ = ['build_dataset', 'build_pk_sampler']


def build_dataset():
    # Get a new config option. We'll default to 'image_folder'
    dataset_type = configs.data_provider.get('dataset_type', 'image_folder')
    transform = ImageTransform()

    if dataset_type == 'image_folder':
        if configs.data_provider.dataset == 'image_folder':
            dataset = ImageFolder(
                root=configs.data_provider.root,
                transforms=transform,
            )
        # ... (other non-triplet datasets) ...
        elif configs.data_provider.dataset == 'cifar100':
            dataset = {
                'train': torchvision.datasets.CIFAR100(configs.data_provider.root, train=True,
                                                    transform=transform['train'], download=True),
                'val': torchvision.datasets.CIFAR100(configs.data_provider.root, train=False,
                                                    transform=transform['val'], download=True),
            }
        else:
            raise NotImplementedError(configs.data_provider.dataset)
            
    elif dataset_type == 'triplet':
        # --- THIS IS THE OLD SLOW METHOD ---
        dataset = {
            'train': TripleFaceDataset(
                root=configs.data_provider.root,
                transform=transform['train']
            ),
        }
        
    elif dataset_type == 'pk_sampler':
        # --- NEW FAST METHOD ---
        # For PK sampling, we just need a standard ImageFolder dataset for 'train'
        # The sampler will handle the batch creation.
        dataset = {
            'train': torchvision.datasets.ImageFolder(
                root=configs.data_provider.root,
                transform=transform['train']
            ),
        }
        # Validation is handled by .bin files, so no 'val' dataset is needed here
        # --- END OF NEW BLOCK ---
        
    else:
        raise NotImplementedError(configs.data_provider.dataset)

    return dataset

def build_pk_sampler(dataset, world_size, rank):
    """
    Builds the PKSampler for training.
    """
    if PKSampler is None:
        raise ImportError("PKSampler not available.")
        
    p = configs.data_provider.get('p_identities')
    k = configs.data_provider.get('k_images')
    
    if not p or not k:
        raise ValueError("data_provider.p_identities and data_provider.k_images must be set in config for pk_sampler.")
    
    return PKSampler(
        dataset=dataset,
        p=p,
        k=k,
        num_replicas=world_size,
        rank=rank
    )
