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

try:
    from agents.building_damage.dataset import XBDDataset
    from agents.building_damage.model import build_model
except ImportError:
    from dataset import XBDDataset
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


def compute_metrics_from_cm(confusion_matrix):
    """Compute per-class IoU, F1 score, and macro averages."""
    cm = confusion_matrix.float()
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    # IoU = TP / (TP + FP + FN)
    union = tp + fp + fn
    ious = []
    for cls in range(cm.shape[0]):
        if union[cls] == 0:
            ious.append(float("nan"))
        else:
            ious.append((tp[cls] / union[cls]).item())

    # F1 = 2*TP / (2*TP + FP + FN)
    f1_denom = 2 * tp + fp + fn
    f1_scores = []
    for cls in range(cm.shape[0]):
        if f1_denom[cls] == 0:
            f1_scores.append(float("nan"))
        else:
            f1_scores.append(((2 * tp[cls]) / f1_denom[cls]).item())

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    valid_f1 = [f for f in f1_scores if not np.isnan(f)]
    mf1 = sum(valid_f1) / len(valid_f1) if valid_f1 else 0.0

    # Damage classes only (classes 1..4, excluding background 0)
    damage_f1 = [f1_scores[c] for c in range(1, cm.shape[0]) if not np.isnan(f1_scores[c])]
    mean_damage_f1 = sum(damage_f1) / len(damage_f1) if damage_f1 else 0.0

    return miou, ious, mf1, f1_scores, mean_damage_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    args = parser.parse_args()

    config = load_config()

    raw_data_path = Path(config["data"]["processed"]["xbd"])
    if raw_data_path.is_absolute():
        data_dir = raw_data_path
    else:
        data_dir = PROJECT_ROOT / raw_data_path

    train_cfg = config["training"]["building_damage"]
    num_classes = config["xbd"]["num_classes"]
    class_names = config["xbd"]["classes"]
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 4)
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    learning_rate = train_cfg["learning_rate"]
    model_save_path = PROJECT_ROOT / config["models"]["building_damage"]
    log_save_path = PROJECT_ROOT / config["outputs"]["building_dmg_log"]

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    log_save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"xBD data directory: {data_dir}")
    logger.info(f"Batch size: {batch_size} | Epochs: {epochs}")

    if not data_dir.exists() or not (data_dir / "train").exists():
        logger.error(f"Processed xBD data directory not found at: {data_dir}")
        logger.info("Please prepare data using scripts/prepare_data.py first.")
        return

    train_dataset = XBDDataset(data_dir=data_dir, split="train")
    val_dataset = XBDDataset(data_dir=data_dir, split="val")

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    model = build_model(
        encoder_name=train_cfg.get("encoder", "resnet34"),
        encoder_weights=train_cfg.get("encoder_weights", "imagenet"),
        classes=num_classes,
    ).to(device)

    # Loss: weighted cross entropy to handle high background / low minor-damage ratios
    # Class weights: Background=0.2, No-Damage=1.0, Minor=3.0, Major=3.0, Destroyed=2.0
    class_weights = torch.tensor([0.2, 1.0, 3.0, 3.0, 2.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_damage_f1 = -1.0
    training_logs = []

    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        train_loss = 0.0
        train_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for pre, post, masks in pbar:
            pre = pre.to(device)
            post = post.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(pre, post)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_cm += calculate_confusion_matrix(outputs, masks, num_classes)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)
        train_miou, _, train_mf1, _, train_dmg_f1 = compute_metrics_from_cm(train_cm)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        with torch.no_grad():
            for pre, post, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                pre = pre.to(device)
                post = post.to(device)
                masks = masks.to(device)

                outputs = model(pre, post)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                val_cm += calculate_confusion_matrix(outputs, masks, num_classes)

        val_loss /= len(val_loader)
        val_miou, val_ious, val_mf1, val_f1s, val_dmg_f1 = compute_metrics_from_cm(val_cm)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} mIoU: {train_miou:.4f} Dmg-F1: {train_dmg_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} mIoU: {val_miou:.4f} Dmg-F1: {val_dmg_f1:.4f}"
        )

        for c_idx, c_name in enumerate(class_names):
            logger.info(f"  Class [{c_name:<14s}]: IoU={val_ious[c_idx]:.4f} | F1={val_f1s[c_idx]:.4f}")

        # Checkpoint saving
        if val_dmg_f1 > best_damage_f1 or epoch == 1:
            best_damage_f1 = val_dmg_f1
            torch.save(model.state_dict(), str(model_save_path))
            logger.info(f"  --> Saved new best checkpoint to {model_save_path}")

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "train_dmg_f1": train_dmg_f1,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "val_dmg_f1": val_dmg_f1,
        }
        for c_idx, c_name in enumerate(class_names):
            log_row[f"val_iou_{c_name}"] = val_ious[c_idx]
            log_row[f"val_f1_{c_name}"] = val_f1s[c_idx]

        training_logs.append(log_row)
        pd.DataFrame(training_logs).to_csv(log_save_path, index=False)


if __name__ == "__main__":
    main()
