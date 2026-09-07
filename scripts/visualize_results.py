import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Fix encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.coordinator.agent import CoordinatorAgent
from agents.building_damage.infer import DAMAGE_COLORMAP

# FloodNet Color Map (BGR)
FLOODNET_COLORMAP = {
    0: (0, 0, 0),         # Background
    1: (0, 0, 255),       # Building-Flooded (Red)
    2: (0, 255, 0),       # Building-Non-Flooded (Green)
    3: (255, 165, 0),     # Road-Flooded (Orange)
    4: (128, 128, 128),   # Road-Non-Flooded (Gray)
    5: (255, 0, 0),       # Water (Blue)
    6: (0, 128, 0),       # Tree (Dark Green)
    7: (255, 255, 0),     # Vehicle (Yellow)
    8: (255, 192, 203),   # Pool (Pink)
    9: (144, 238, 144)    # Grass (Light Green)
}


def colorize_mask(mask: np.ndarray, colormap: dict) -> np.ndarray:
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in colormap.items():
        color_img[mask == class_id] = color
    return color_img


def generate_visual_assessment(
    pre_path: str,
    post_path: str,
    zone_id: str = "zone_001",
    output_dir: str = "outputs/demo"
):
    print(f"[Visualizer] Processing visual overlay for {zone_id}...")
    coordinator = CoordinatorAgent(use_llm=False)
    
    # 1. Run inference across specialist agents
    flood_results = coordinator.flood_agent.predict(post_path)
    damage_results = coordinator.building_agent.predict(pre_path, post_path)
    
    # 2. Read original images
    pre_img = cv2.cvtColor(cv2.imread(pre_path), cv2.COLOR_BGR2RGB)
    post_img = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB)
    
    # Resize to match mask size
    h, w = flood_results["mask"].shape
    pre_img = cv2.resize(pre_img, (w, h))
    post_img = cv2.resize(post_img, (w, h))
    
    # 3. Create color masks
    flood_color = colorize_mask(flood_results["mask"], FLOODNET_COLORMAP)
    damage_color = colorize_mask(damage_results["mask"], DAMAGE_COLORMAP)
    
    # Blended overlays
    flood_overlay = cv2.addWeighted(post_img, 0.6, cv2.cvtColor(flood_color, cv2.COLOR_BGR2RGB), 0.4, 0)
    damage_overlay = cv2.addWeighted(post_img, 0.6, cv2.cvtColor(damage_color, cv2.COLOR_BGR2RGB), 0.4, 0)
    
    # 4. Plot 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Disaster Assessment Dashboard — Zone: {zone_id}", fontsize=16, fontweight='bold')
    
    axes[0, 0].imshow(pre_img)
    axes[0, 0].set_title("1. Pre-Disaster Imagery", fontsize=12)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(post_img)
    axes[0, 1].set_title("2. Post-Disaster Imagery", fontsize=12)
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(flood_overlay)
    axes[1, 0].set_title(f"3. Flood Segmentation (Severity: {flood_results['flood_severity'].upper()})", fontsize=12)
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(damage_overlay)
    axes[1, 1].set_title(f"4. Building Damage Overlay (Score: {damage_results['damage_severity_score']:.1f}/100)", fontsize=12)
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    
    out_path = Path(output_dir) / f"{zone_id}_visual_assessment.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Visualizer] Successfully saved visual assessment map to: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Disaster Multi-Agent AI — Visual Overlay Dashboard Generator")
    parser.add_argument("--zone", type=str, default="zone_001", help="Zone identifier")
    parser.add_argument("--pre", type=str, default="data/sample/sample_pre.png", help="Path to pre-disaster image")
    parser.add_argument("--post", type=str, default="data/sample/sample_post.png", help="Path to post-disaster image")
    args = parser.parse_args()

    generate_visual_assessment(args.pre, args.post, zone_id=args.zone)

