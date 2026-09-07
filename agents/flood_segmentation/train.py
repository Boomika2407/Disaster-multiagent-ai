import argparse
import os
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

try:
    from agents.flood_segmentation.dataset import FloodNetDataset
    from agents.flood_segmentation.model import build_model
except ImportError:
    from dataset import FloodNetDataset
    from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def calculate_confusion_matrix(preds, labels, num_classes):
    """Compute confusion matrix on CPU to preserve GPU VRAM."""
    preds = torch.argmax(preds, dim=1).detach().cpu().view(-1)
    labels = labels.detach().cpu().view(-1)
    
    # Filter valid labels
    mask = (labels >= 0) & (labels < num_classes)
    preds = preds[mask]
    labels = labels[mask]
    
    bins = num_classes * labels + preds
    counts = torch.bincount(bins, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes)

def compute_ious_from_confusion_matrix(confusion_matrix):
    """Compute per-class IoU and mean IoU from global confusion matrix."""
    cm = confusion_matrix.float()
    intersection = torch.diag(cm)
    ground_truth_set = cm.sum(dim=1)
    predicted_set = cm.sum(dim=0)
    union = ground_truth_set + predicted_set - intersection
    
    ious = []
    for cls in range(cm.shape[0]):
        if union[cls] == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection[cls] / union[cls]).item())
            
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    return miou, ious

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    args = parser.parse_args()

    config = load_config()
    
    # Configuration
    raw_data_path = Path(config["data"]["processed"]["floodnet"])
    if raw_data_path.is_absolute():
        data_dir = raw_data_path
    else:
        data_dir = PROJECT_ROOT / raw_data_path
        
    train_cfg = config["training"]["flood_segmentation"]
    num_classes = config["floodnet"]["num_classes"]
    class_names = config["floodnet"]["classes"]
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 4)
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    learning_rate = train_cfg["learning_rate"]
    model_save_path = PROJECT_ROOT / config["models"]["flood_segmentation"]
    log_save_path = PROJECT_ROOT / config["outputs"]["flood_seg_log"]
    
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    log_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"FloodNet data directory: {data_dir}")
    logger.info(f"Batch size: {batch_size} | Epochs: {epochs}")
    
    # Datasets and Dataloaders
    train_dataset = FloodNetDataset(data_dir=data_dir, split="train")
    val_dataset = FloodNetDataset(data_dir=data_dir, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Model, Loss, Optimizer, Scaler
    model = build_model(
        encoder_name=train_cfg["encoder"], 
        encoder_weights=train_cfg["encoder_weights"], 
        classes=num_classes
    )
    model.to(device)
    
    criterion = smp.losses.FocalLoss(mode="multiclass", alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training Loop
    best_miou = 0.0
    history = []
    
    for epoch in range(epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Train
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validate
        model.eval()
        val_loss = 0.0
        total_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                images = images.to(device)
                masks = masks.to(device)
                
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                cm = calculate_confusion_matrix(outputs, masks, num_classes)
                total_cm += cm
                
        val_loss /= len(val_dataset)
        val_miou, per_class_ious = compute_ious_from_confusion_matrix(total_cm)
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou*100:.2f}%")
        logger.info("Per-Class IoU Breakdown:")
        for cls_id, iou_val in enumerate(per_class_ious):
            name = class_names[cls_id] if cls_id < len(class_names) else f"Class-{cls_id}"
            iou_str = f"{iou_val * 100:6.2f}%" if not np.isnan(iou_val) else "   N/A (No GT/Pred)"
            logger.info(f"  [{cls_id:02d}] {name:<22s}: {iou_str}")
        
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "lr": optimizer.param_groups[0]['lr']
        }
        for cls_id, iou_val in enumerate(per_class_ious):
            name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            epoch_record[f"iou_{name}"] = iou_val
            
        history.append(epoch_record)
        
        # Save best model
        if val_miou > best_miou or epoch == 0:
            best_miou = val_miou
            torch.save(model.state_dict(), str(model_save_path))
            logger.info(f"Saved best model checkpoint to {model_save_path} (mIoU: {val_miou*100:.2f}%)")
            
    # Save training logs
    pd.DataFrame(history).to_csv(str(log_save_path), index=False)
    logger.info(f"Training history successfully saved to {log_save_path}")

if __name__ == "__main__":
    main()
