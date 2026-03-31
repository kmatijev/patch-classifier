"""
Data augmentation strategies for patch classifier
=====================================================================
Defines 4 augmentation approaches for domain shift robustness testing
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np


class PosterizeFloat32(torch.nn.Module):
    """Custom posterize that handles float32 tensors (0-1 range)"""
    def __init__(self, bits=4, p=0.3):
        super().__init__()
        self.bits = bits
        self.p = p
    
    def forward(self, img):
        if torch.rand(1).item() < self.p:
            # Convert float32 (0-1) to uint8 (0-255)
            if img.dtype == torch.float32:
                img_uint8 = (img * 255).to(torch.uint8)
                # Apply posterize
                img_uint8 = F.posterize(img_uint8, self.bits)
                # Convert back to float32 (0-1)
                img = img_uint8.to(torch.float32) / 255.0
            else:
                img = F.posterize(img, self.bits)
        return img


class AugmentationStrategy:
    """Base augmentation strategy"""
    def get_transforms(self):
        raise NotImplementedError


class StandardAugmentation(AugmentationStrategy):
    """
    Standard augmentation: Basic geometric and color transforms
    - Random horizontal and vertical flips
    - Random rotation (-15 to +15 degrees)
    - Color jitter (brightness, contrast)
    - Normalization for ImageNet
    """
    def get_transforms(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __repr__(self):
        return "StandardAugmentation"


class StrongAugmentation(AugmentationStrategy):
    """
    Strong augmentation: More aggressive transforms
    - Stronger rotations
    - GaussianBlur
    - Stronger color jitter
    - Random affine transforms
    """
    def get_transforms(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __repr__(self):
        return "StrongAugmentation"


class MediumAugmentation(AugmentationStrategy):
    """
    Medium augmentation: Balanced between Standard and Strong
    - Moderate rotations (20 degrees)
    - Moderate affine transforms
    - Light GaussianBlur
    - Medium color jitter
    """
    def get_transforms(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __repr__(self):
        return "MediumAugmentation"


class HistologyAugmentation(AugmentationStrategy):
    """
    Histology-specific augmentation: Targets staining variation and domain shift
    - Strong color jitter (mimics stain variation)
    - Elastic deformations (tissue distortion)
    - Stronger geometric transforms
    - Blur and noise (scanner artifacts)
    """
    def get_transforms(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            # Extreme color jitter for stain variation
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.5)),
            # Custom posterize that handles float32 images
            PosterizeFloat32(bits=4, p=0.3),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __repr__(self):
        return "HistologyAugmentation"


def get_augmentation(strategy_name):
    """
    Get augmentation strategy by name
    
    Args:
        strategy_name: "standard", "medium", "strong", or "histology"
    
    Returns:
        Augmentation transform object
    """
    strategies = {
        "standard": StandardAugmentation(),
        "medium": MediumAugmentation(),
        "strong": StrongAugmentation(),
        "histology": HistologyAugmentation(),
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown augmentation strategy: {strategy_name}")
    
    return strategies[strategy_name].get_transforms()


# Available augmentation names
AUGMENTATION_STRATEGIES = ["standard", "medium", "strong", "histology"]


class AugmentationInfo:
    """Information about available augmentations for logging"""
    
    @staticmethod
    def get_all_info():
        return {
            "standard": {
                "name": "Standard Augmentation",
                "description": "Basic geometric and color transforms",
                "strategy": StandardAugmentation()
            },
            "medium": {
                "name": "Medium Augmentation",
                "description": "Balanced transforms between Standard and Strong",
                "strategy": MediumAugmentation()
            },
            "strong": {
                "name": "Strong Augmentation",
                "description": "Aggressive transforms including affine and blur",
                "strategy": StrongAugmentation()
            },
            "histology": {
                "name": "Histology-Specific Augmentation",
                "description": "Targets staining variation and domain shift",
                "strategy": HistologyAugmentation()
            }
        }
    
    @staticmethod
    def print_info():
        """Print information about all augmentations"""
        info = AugmentationInfo.get_all_info()
        print("\n" + "="*80)
        print("Available Augmentation Strategies")
        print("="*80)
        for key, val in info.items():
            print(f"\n{val['name']} ({key}):")
            print(f"  {val['description']}")
        print("="*80 + "\n")
