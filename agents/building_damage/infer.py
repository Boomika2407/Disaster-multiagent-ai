import os
import yaml
from pathlib import Path
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from agents.building_damage.model import build_model
except ImportError:
    from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# Palette for 5 xBD classes:
# 0: Background (Black/Transparent)
# 1: No Damage (Green)
# 2: Minor Damage (Yellow)
# 3: Major Damage (Orange)
# 4: Destroyed (Red)
DAMAGE_COLORMAP = {
    0: (0, 0, 0),        # Background
    1: (46, 204, 113),   # No Damage (Green)
    2: (241, 196, 15),   # Minor Damage (Yellow)
    3: (230, 126, 34),   # Major Damage (Orange)
    4: (231, 76, 60),    # Destroyed (Red)
}


class BuildingDamageAgent:
    """
    Building Damage Assessment Agent (Agent 2).
    Evaluates building structures across pre- and post-disaster imagery pairs,
    classifying per-pixel damage levels and providing structured summary statistics.
    """
    def __init__(self, checkpoint_path: str = None):
        self.config = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_cfg = self.config["training"]["building_damage"]
        self.num_classes = self.config["xbd"]["num_classes"]
        self.class_names = self.config["xbd"]["classes"]

        self.model = build_model(
            encoder_name=train_cfg.get("encoder", "resnet34"),
            encoder_weights=None,
            classes=self.num_classes,
        )

        model_path = Path(checkpoint_path) if checkpoint_path else PROJECT_ROOT / self.config["models"]["building_damage"]
        if model_path.exists():
            self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        else:
            print(f"Warning: Model checkpoint not found at {model_path}. Using uninitialized weights.")

        self.model.to(self.device)
        self.model.eval()

        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.normalize_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.img_size = self.config["preprocessing"]["image_size"]

    def _preprocess_image(self, img_path: Path) -> tuple[torch.Tensor, np.ndarray]:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image at {img_path}")
        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.normalize_mean) / self.normalize_std
        return tensor.unsqueeze(0), img_rgb

    def predict(self, pre_image_path: str, post_image_path: str) -> dict:
        """
        Runs building damage assessment inference on pre/post image pair.
        Returns:
            {
                "mask": <np.ndarray [H, W] of class IDs>,
                "damage_percentages": {class_name: pct_of_image},
                "building_damage_distribution": {class_name: pct_of_building_area},
                "total_building_pixels": int,
                "dominant_damage_level": str,
                "damage_severity_score": float (0-100 scale),
            }
        """
        pre_path = Path(pre_image_path)
        post_path = Path(post_image_path)

        if not pre_path.exists():
            raise FileNotFoundError(f"Pre image not found: {pre_path}")
        if not post_path.exists():
            raise FileNotFoundError(f"Post image not found: {post_path}")

        pre_tensor, _ = self._preprocess_image(pre_path)
        post_tensor, post_rgb = self._preprocess_image(post_path)

        pre_tensor = pre_tensor.to(self.device)
        post_tensor = post_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(pre_tensor, post_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Compute statistics
        total_pixels = self.img_size * self.img_size
        unique, counts = np.unique(pred_mask, return_counts=True)
        counts_dict = {cls_id: 0 for cls_id in range(self.num_classes)}
        for cls_id, cnt in zip(unique, counts):
            if cls_id < self.num_classes:
                counts_dict[cls_id] = int(cnt)

        # Image percentages
        damage_percentages = {
            self.class_names[c]: (counts_dict[c] / total_pixels) * 100
            for c in range(self.num_classes)
        }

        # Building-only distribution (classes 1..4)
        total_building_pixels = sum(counts_dict[c] for c in range(1, 5))
        building_damage_dist = {}
        for c in range(1, 5):
            c_name = self.class_names[c]
            if total_building_pixels > 0:
                building_damage_dist[c_name] = (counts_dict[c] / total_building_pixels) * 100
            else:
                building_damage_dist[c_name] = 0.0

        # Weighted severity score (0 = No Damage, 100 = 100% Destroyed)
        # Weights: No-Damage=0, Minor=0.33, Major=0.66, Destroyed=1.0
        if total_building_pixels > 0:
            severity_score = (
                counts_dict[1] * 0.0 +
                counts_dict[2] * 33.3 +
                counts_dict[3] * 66.6 +
                counts_dict[4] * 100.0
            ) / total_building_pixels
        else:
            severity_score = 0.0

        # Dominant non-background damage class
        active_damages = {k: v for k, v in building_damage_dist.items() if v > 0}
        dominant_damage = max(active_damages, key=active_damages.get) if active_damages else "None"

        return {
            "mask": pred_mask,
            "damage_percentages": damage_percentages,
            "building_damage_distribution": building_damage_dist,
            "total_building_pixels": total_building_pixels,
            "dominant_damage_level": dominant_damage,
            "damage_severity_score": round(severity_score, 2),
        }

    def generate_overlay(self, post_image_path: str, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Creates a colorized overlay of the damage mask on the post-disaster image."""
        post_img = cv2.imread(str(post_image_path))
        post_img = cv2.resize(post_img, (self.img_size, self.img_size))

        color_mask = np.zeros_like(post_img)
        for cls_id, color_rgb in DAMAGE_COLORMAP.items():
            if cls_id == 0:
                continue
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            color_mask[mask == cls_id] = color_bgr

        # Blend where mask > 0
        mask_binary = (mask > 0).astype(np.uint8)
        overlay = post_img.copy()
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_binary == 1,
                (1 - alpha) * post_img[:, :, c] + alpha * color_mask[:, :, c],
                post_img[:, :, c]
            )
        return overlay.astype(np.uint8)


if __name__ == "__main__":
    agent = BuildingDamageAgent()
    print("BuildingDamageAgent initialized successfully.")
