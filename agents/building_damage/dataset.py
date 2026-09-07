import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class XBDDataset(Dataset):
    """
    Dataset loader for xBD / Building Damage assessment.
    Expects data formatted in splits:
        data_dir / split / pre / <filename>.png
        data_dir / split / post / <filename>.png
        data_dir / split / masks / <filename>.png
    """
    def __init__(self, data_dir: Path, split: str = "train", img_size: int = 512):
        self.data_dir = Path(data_dir) / split
        self.pre_dir = self.data_dir / "pre"
        self.post_dir = self.data_dir / "post"
        self.mask_dir = self.data_dir / "masks"
        self.split = split
        self.img_size = img_size

        if not self.pre_dir.exists() or not self.post_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing pre, post, or masks folder in: {self.data_dir}")

        self.images = sorted([
            f.name for f in self.pre_dir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif"]
        ])

        self.transform = self._get_transforms()
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.normalize_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _get_transforms(self):
        if self.split == "train":
            # Joint augmentations for both pre and post images
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                ],
                additional_targets={"post_image": "image"}
            )
        else:
            return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        pre_path = self.pre_dir / fname
        post_path = self.post_dir / fname
        mask_path = self.mask_dir / fname

        pre_img = cv2.imread(str(pre_path))
        if pre_img is None:
            raise ValueError(f"Failed to read {pre_path}")
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

        post_img = cv2.imread(str(post_path))
        if post_img is None:
            raise ValueError(f"Failed to read {post_path}")
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read {mask_path}")

        # Ensure correct size
        if pre_img.shape[0] != self.img_size or pre_img.shape[1] != self.img_size:
            pre_img = cv2.resize(pre_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            post_img = cv2.resize(post_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=pre_img, post_image=post_img, mask=mask)
            pre_img = augmented["image"]
            post_img = augmented["post_image"]
            mask = augmented["mask"]

        # Convert to normalized float tensors
        pre_tensor = torch.from_numpy(pre_img).permute(2, 0, 1).float() / 255.0
        pre_tensor = (pre_tensor - self.normalize_mean) / self.normalize_std

        post_tensor = torch.from_numpy(post_img).permute(2, 0, 1).float() / 255.0
        post_tensor = (post_tensor - self.normalize_mean) / self.normalize_std

        mask_tensor = torch.from_numpy(mask).long()

        return pre_tensor, post_tensor, mask_tensor
