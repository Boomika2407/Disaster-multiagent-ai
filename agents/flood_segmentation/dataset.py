import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FloodNetDataset(Dataset):
    def __init__(self, data_dir: Path, split: str = "train", img_size: int = 512):
        self.data_dir = Path(data_dir) / split
        self.img_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"
        self.split = split
        self.img_size = img_size
        
        # Verify directories exist
        if not self.img_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
            
        self.images = sorted([f.name for f in self.img_dir.iterdir() if f.suffix in ['.png', '.jpg']])
        
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        if self.split == "train":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ])
        else:
            return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.img_dir / img_name
        mask_path = self.mask_dir / img_name  # masks have same name in processed data
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Convert image to float tensor normalized to ImageNet stats
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor
