import os
import yaml
from pathlib import Path
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from agents.flood_segmentation.model import build_model
except ImportError:
    from model import build_model

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

class FloodSegmentationAgent:
    def __init__(self):
        self.config = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        train_cfg = self.config["training"]["flood_segmentation"]
        self.num_classes = self.config["floodnet"]["num_classes"]
        self.class_names = self.config["floodnet"]["classes"]
        
        self.model = build_model(
            encoder_name=train_cfg["encoder"],
            encoder_weights=None,  # Loading trained weights below
            classes=self.num_classes
        )
        
        model_path = PROJECT_ROOT / self.config["models"]["flood_segmentation"]
        if model_path.exists():
            self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        else:
            print(f"Warning: Model checkpoint not found at {model_path}. Using untrained weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        self.severity_thresholds = self.config["thresholds"]["flood_severity"]

    def predict(self, image_path: str) -> dict:
        """
        Runs flood segmentation inference on a single image.
        Returns:
            {
                "mask": <numpy array, per-pixel class ids>,
                "class_pixel_percentages": {class_name: percent, ...},
                "flood_severity": "none" | "partial" | "severe"
            }
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image at {image_path}")
            
        # Resize to expected input size
        img_size = self.config["preprocessing"]["image_size"]
        original_size = (image.shape[1], image.shape[0])
        image_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=image_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
        # Calculate percentages
        unique, counts = np.unique(pred_mask, return_counts=True)
        total_pixels = img_size * img_size
        
        percentages = {}
        for cls_id, count in zip(unique, counts):
            if cls_id < len(self.class_names):
                cls_name = self.class_names[cls_id]
                percentages[cls_name] = (count / total_pixels) * 100
                
        # Calculate derived flood severity
        # (building-flooded + road-flooded)
        building_flooded_pct = percentages.get("Building-Flooded", 0.0)
        road_flooded_pct = percentages.get("Road-Flooded", 0.0)
        critical_flooded_pct = building_flooded_pct + road_flooded_pct
        
        severity = "none"
        if critical_flooded_pct > self.severity_thresholds["severe_percent"]:
            severity = "severe"
        elif critical_flooded_pct > self.severity_thresholds["partial_percent"]:
            severity = "partial"
            
        # Optional: resize mask back to original size for visualization if needed later
        # pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
        return {
            "mask": pred_mask,
            "class_pixel_percentages": percentages,
            "flood_severity": severity
        }

if __name__ == "__main__":
    # Simple test if script is run directly
    agent = FloodSegmentationAgent()
    print("Agent initialized.")
