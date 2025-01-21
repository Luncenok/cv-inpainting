"""
Data loading and preprocessing utilities.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
from pathlib import Path

class MaskGenerator:
    def __init__(self, height: int, width: int, channels: int = 1):
        self.height = height
        self.width = width
        self.channels = channels
    
    def random_rectangle(self) -> np.ndarray:
        """Generate a random rectangular mask."""
        mask = np.zeros((self.height, self.width, self.channels), np.float32)
        
        # Random dimensions
        h = np.random.randint(self.height // 4, self.height // 2)
        w = np.random.randint(self.width // 4, self.width // 2)
        
        # Random position
        x = np.random.randint(0, self.width - w)
        y = np.random.randint(0, self.height - h)
        
        mask[y:y+h, x:x+w] = 1
        return mask
    
    def random_ellipse(self) -> np.ndarray:
        """Generate a random elliptical mask."""
        mask = np.zeros((self.height, self.width, self.channels), np.float32)
        
        # Random center and axes
        center = (np.random.randint(self.width // 4, 3 * self.width // 4),
                 np.random.randint(self.height // 4, 3 * self.height // 4))
        axes = (np.random.randint(self.width // 8, self.width // 3),
               np.random.randint(self.height // 8, self.height // 3))
        angle = np.random.randint(0, 180)
        
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
        return mask

class InpaintingDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 mask_generator: Optional[MaskGenerator] = None,
                 debug: bool = False):
        """
        Args:
            root_dir: Path to CelebA dataset root
            split: One of ['train', 'val', 'test']
            transform: Optional transform to be applied on images
            mask_generator: Optional custom mask generator
            debug: If True, limit dataset size
        """
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise ValueError(f"Split directory {self.split_dir} does not exist")
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.mask_generator = mask_generator or MaskGenerator(256, 256)
        
        # Load image paths
        all_image_paths = list(self.split_dir.glob('*.jpg'))
        if not all_image_paths:
            raise ValueError(f"No images found in {self.split_dir}")
            
        # Limit dataset size in debug mode
        if debug:
            limit = 100 if split == 'val' else 100
            self.image_paths = all_image_paths[:limit]
            print(f"Debug mode: Using {len(self.image_paths)} images for {split}")
        else:
            self.image_paths = all_image_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Generate mask (randomly choose between rectangle and ellipse)
        if np.random.random() > 0.5:
            mask = torch.from_numpy(self.mask_generator.random_rectangle()).permute(2, 0, 1)
        else:
            mask = torch.from_numpy(self.mask_generator.random_ellipse()).permute(2, 0, 1)
        
        # Apply mask to image
        masked_image = image * (1 - mask)
        
        return {
            'image': image,
            'masked_image': masked_image,
            'mask': mask
        }

def get_data_loaders(root_dir: str,
                    batch_size: int = 8,
                    num_workers: int = 4,
                    debug: bool = False) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.
    
    Args:
        root_dir: Path to dataset root directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        debug: If True, limit dataset size to 1000 training and 100 validation samples
    """
    # Create datasets
    train_dataset = InpaintingDataset(root_dir, split='train', debug=debug)
    val_dataset = InpaintingDataset(
        root_dir,
        split='val',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        debug=debug
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
