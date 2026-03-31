"""
Updated patch classifier dataset with augmentation support
"""

import torch
from PIL import Image
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PatchClassifierDatasetAugmented(Dataset):
    """
    Binary patch classification dataset with optional augmentation
    Supports domain shift evaluation (train on one scanner, test on others)
    """
    
    def __init__(self, root, scanner, augmentation=None, split='train'):
        """
        Args:
            root: Path to dataset root (e.g., classifier_patches_large or patches_multi_scanner)
            scanner: Scanner name (e.g., "Hamamatsu_XR")
            augmentation: Augmentation transform (from augmentation_strategies.py)
            split: 'train', 'val', or 'test'
        """
        self.root = Path(root)
        self.scanner = scanner
        self.split = split
        self.augmentation = augmentation
        
        # Default normalization for ImageNet pretrained models
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Find images and masks
        split_dir = self.root / scanner / split
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Get list of image files
        self.image_files = sorted([
            f for f in images_dir.iterdir() 
            if f.suffix.lower() in ['.png', '.jpg', '.tif']
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        self.masks_dir = masks_dir
        
        # Pre-compute labels
        self.labels = []
        for image_file in self.image_files:
            mask_file = self.masks_dir / image_file.name
            if mask_file.exists():
                mask_array = np.array(Image.open(mask_file).convert('L'))
                label = 1.0 if np.sum(mask_array > 0) > 0 else 0.0
            else:
                label = 0.0
            self.labels.append(label)
        
        print(f"Loaded {len(self.image_files)} patches from {scanner}/{split}")
        pos = sum(self.labels)
        print(f"  Positive: {int(pos)}, Negative: {len(self.labels) - int(pos)}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(image_file).convert('RGB')
        
        # Apply augmentation if provided
        if self.augmentation is not None and self.split == 'train':
            # Convert to tensor first
            img = transforms.ToTensor()(img)
            # Apply augmentation
            img = self.augmentation(img)
        else:
            # Just normalize
            img = self.normalize(img)
        
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)
        
        return img, label
