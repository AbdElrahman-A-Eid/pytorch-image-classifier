import os
from typing import Dict, Tuple
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_data(data_dir: str) -> Tuple[Dict[str, ImageFolder], Dict[str, DataLoader]]:
    """
    Prepares the datasets and dataloaders for training, validation, and testing.

    Args:
        data_dir (str): The directory containing the 'train', 'valid', and 'test' subdirectories with images.

    Returns:
        Tuple[Dict[str, ImageFolder], Dict[str, DataLoader]]: A tuple containing:
            - image_datasets (dict): A dictionary with keys 'train', 'valid', and 'test' mapping to the corresponding datasets.
            - dataloaders (dict): A dictionary with keys 'train', 'valid', and 'test' mapping to the corresponding dataloaders.
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    data_transforms = {
        'train': transforms.Compose([
            transforms.AutoAugment(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['valid']),
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    }

    return image_datasets, dataloaders
