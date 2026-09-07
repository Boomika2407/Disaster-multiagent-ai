"""
Data Preparation Script for Disaster Multi-Agent AI
====================================================
Preprocesses FloodNet and xBD datasets into clean, model-ready format.

Usage:
    python scripts/prepare_data.py --dry-run          # Preview only, no files written
    python scripts/prepare_data.py                     # Full preprocessing
    python scripts/prepare_data.py --floodnet-only     # Process FloodNet only
    python scripts/prepare_data.py --xbd-only          # Process xBD only
"""

import argparse
import os
import sys
import json
import shutil
import logging
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
import yaml
from shapely import wkt as shapely_wkt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Resolve project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    """Load central configuration from config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# FloodNet Class Definitions
# ──────────────────────────────────────────────────────────────

FLOODNET_CLASSES = {
    0: "Background",
    1: "Building-Flooded",
    2: "Building-Non-Flooded",
    3: "Road-Flooded",
    4: "Road-Non-Flooded",
    5: "Water",
    6: "Tree",
    7: "Vehicle",
    8: "Pool",
    9: "Grass",
}

# ──────────────────────────────────────────────────────────────
# xBD Class Definitions
# ──────────────────────────────────────────────────────────────

XBD_CLASSES = {
    0: "Background",
    1: "No-Damage",
    2: "Minor-Damage",
    3: "Major-Damage",
    4: "Destroyed",
}

# Maps xBD JSON subtype strings → mask class IDs
SUBTYPE_TO_CLASS = {
    "no-damage":    1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed":    4,
}


# ══════════════════════════════════════════════════════════════
# FLOODNET PREPROCESSING
# ══════════════════════════════════════════════════════════════

def discover_floodnet_pairs(raw_dir: Path) -> list[dict]:
    """
    Discover image/mask pairs from the actual FloodNet directory structure.

    Expected structure:
        raw_dir/
        ├── Train/
        │   ├── Labeled/
        │   │   ├── Flooded/     {image/, mask/}
        │   │   └── Non-Flooded/ {image/, mask/}
        │   └── Unlabeled/       {image/}  ← skipped (no masks)
        ├── Test/                {image/}  ← skipped (no masks)
        └── Validation/          {image/}  ← skipped (no masks)

    Mask naming convention: image "10165.jpg" → mask "10165_lab.png"
    """
    pairs = []
    skipped_no_mask = 0
    skipped_unlabeled = 0

    # ── Labeled subdirectories (Flooded + Non-Flooded) ──
    labeled_dir = raw_dir / "Train" / "Labeled"
    if not labeled_dir.exists():
        # Fallback: try flat structure (images/ + masks/)
        logger.info("Labeled dir not found, trying flat structure (images/ + masks/)...")
        return _discover_floodnet_flat(raw_dir)

    for category in ["Flooded", "Non-Flooded"]:
        cat_dir = labeled_dir / category
        img_dir = cat_dir / "image"
        mask_dir = cat_dir / "mask"

        if not img_dir.exists():
            logger.warning(f"Image directory not found: {img_dir}")
            continue

        for img_file in sorted(img_dir.iterdir()):
            if not img_file.is_file() or img_file.suffix.lower() not in (".jpg", ".png", ".jpeg", ".tif"):
                continue

            # Derive mask filename: "10165.jpg" → "10165_lab.png"
            stem = img_file.stem
            mask_file = mask_dir / f"{stem}_lab.png"

            if not mask_file.exists():
                # Try other extensions
                found = False
                for ext in [".png", ".jpg", ".tif"]:
                    alt = mask_dir / f"{stem}_lab{ext}"
                    if alt.exists():
                        mask_file = alt
                        found = True
                        break
                if not found:
                    skipped_no_mask += 1
                    continue

            pairs.append({
                "image": img_file,
                "mask": mask_file,
                "category": category.lower(),  # "flooded" or "non-flooded"
            })

    # ── Count unlabeled/test/val images that we skip ──
    for skip_dir in [
        raw_dir / "Train" / "Unlabeled" / "image",
        raw_dir / "Test" / "image",
        raw_dir / "Validation" / "image",
    ]:
        if skip_dir.exists():
            count = sum(1 for f in skip_dir.iterdir() if f.is_file())
            skipped_unlabeled += count
            logger.info(f"Skipping {count} images from {skip_dir.relative_to(raw_dir)} (no masks)")

    logger.info(f"FloodNet: Found {len(pairs)} labeled image/mask pairs")
    logger.info(f"FloodNet: Skipped {skipped_no_mask} images with missing masks")
    logger.info(f"FloodNet: Skipped {skipped_unlabeled} unlabeled/test/val images (no masks available)")

    return pairs


def _discover_floodnet_flat(raw_dir: Path) -> list[dict]:
    """Fallback: discover pairs from flat images/ + masks/ structure."""
    pairs = []
    img_dir = raw_dir / "images"
    mask_dir = raw_dir / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        logger.error(f"Cannot find FloodNet data in {raw_dir}")
        return pairs

    for img_file in sorted(img_dir.iterdir()):
        if not img_file.is_file():
            continue
        stem = img_file.stem
        # Try common mask naming patterns
        for pattern in [f"{stem}_lab.png", f"{stem}.png", f"{stem}_mask.png"]:
            mask_file = mask_dir / pattern
            if mask_file.exists():
                pairs.append({"image": img_file, "mask": mask_file, "category": "unknown"})
                break

    return pairs


def compute_class_distribution(pairs: list[dict], num_classes: int = 10,
                                sample_limit: int = 50) -> dict[int, int]:
    """
    Compute pixel-level class distribution across masks.
    Samples up to `sample_limit` masks for speed during dry-run.
    """
    class_pixels = Counter()
    sample = pairs if len(pairs) <= sample_limit else \
        [pairs[i] for i in np.linspace(0, len(pairs) - 1, sample_limit, dtype=int)]

    for item in tqdm(sample, desc="Analyzing class distribution", leave=False):
        mask = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        unique, counts = np.unique(mask, return_counts=True)
        for cls_id, count in zip(unique, counts):
            if cls_id < num_classes:
                class_pixels[cls_id] += int(count)

    return dict(class_pixels)


def split_dataset(pairs: list[dict], train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                  random_state=42) -> dict[str, list]:
    """
    Split pairs into train/val/test.
    Stratifies by category (flooded vs non-flooded) when available.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Extract stratification labels
    categories = [p["category"] for p in pairs]
    has_strat = len(set(categories)) > 1 and all(c != "unknown" for c in categories)

    if has_strat:
        logger.info("Using stratified split by flood category")
        strat = categories
    else:
        logger.info("Using random split (no stratification labels)")
        strat = None

    # First split: train vs (val+test)
    train, temp, strat_train, strat_temp = train_test_split(
        pairs, categories if strat else [0]*len(pairs),
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=strat,
    )

    # Second split: val vs test (from the temp set)
    val_frac = val_ratio / (val_ratio + test_ratio)
    strat2 = strat_temp if has_strat else None
    val, test = train_test_split(
        temp,
        test_size=(1 - val_frac),
        random_state=random_state,
        stratify=strat2,
    )

    return {"train": train, "val": val, "test": test}


def process_floodnet(config: dict, dry_run: bool = True):
    """Main FloodNet preprocessing pipeline."""
    raw_dir = Path(config["data"]["raw"]["floodnet"])
    out_dir = Path(config["data"]["processed"]["floodnet"])
    img_size = config["preprocessing"]["image_size"]
    ratios = config["preprocessing"]["split_ratios"]

    logger.info("=" * 60)
    logger.info("FLOODNET PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"Raw data directory : {raw_dir}")
    logger.info(f"Output directory   : {out_dir}")
    logger.info(f"Target image size  : {img_size}x{img_size}")

    # ── Step 1: Discover pairs ──
    if not raw_dir.exists():
        logger.error(f"FloodNet raw directory not found: {raw_dir}")
        return None

    pairs = discover_floodnet_pairs(raw_dir)
    if not pairs:
        logger.error("No image/mask pairs found. Check directory structure.")
        return None

    # ── Step 2: Class distribution ──
    logger.info("\nAnalyzing class distribution...")
    class_dist = compute_class_distribution(pairs, num_classes=10)
    total_pixels = sum(class_dist.values())

    logger.info("\n┌─────────────────────────────────────────────────────┐")
    logger.info("│            FloodNet Class Distribution              │")
    logger.info("├──────┬───────────────────────┬──────────┬───────────┤")
    logger.info("│  ID  │ Class Name            │  Pixels  │ Percent   │")
    logger.info("├──────┼───────────────────────┼──────────┼───────────┤")
    for cls_id in range(10):
        name = FLOODNET_CLASSES.get(cls_id, f"Unknown-{cls_id}")
        px = class_dist.get(cls_id, 0)
        pct = (px / total_pixels * 100) if total_pixels > 0 else 0
        logger.info(f"│  {cls_id:2d}  │ {name:<21s} │ {px:>8,d} │ {pct:>7.2f}%  │")
    logger.info("└──────┴───────────────────────┴──────────┴───────────┘")

    # ── Step 3: Split ──
    splits = split_dataset(
        pairs,
        train_ratio=ratios["train"],
        val_ratio=ratios["val"],
        test_ratio=ratios["test"],
    )

    # Category breakdown per split
    logger.info("\n┌────────────────────────────────────────────────────┐")
    logger.info("│              FloodNet Split Summary                │")
    logger.info("├──────────┬────────┬─────────┬─────────────────────┤")
    logger.info("│  Split   │ Total  │ Flooded │ Non-Flooded         │")
    logger.info("├──────────┼────────┼─────────┼─────────────────────┤")
    for split_name in ["train", "val", "test"]:
        items = splits[split_name]
        n_flood = sum(1 for p in items if p["category"] == "flooded")
        n_nf = sum(1 for p in items if p["category"] == "non-flooded")
        logger.info(f"│  {split_name:<7s} │ {len(items):>5d}  │ {n_flood:>6d}  │ {n_nf:>6d}              │")
    logger.info("└──────────┴────────┴─────────┴─────────────────────┘")

    if dry_run:
        logger.info("\n[DRY RUN] No files written. Re-run without --dry-run to process.")
        return {
            "dataset": "FloodNet",
            "total_pairs": len(pairs),
            "splits": {k: len(v) for k, v in splits.items()},
            "class_distribution": class_dist,
        }

    # ── Step 4: Resize and save ──
    logger.info("\nWriting processed files...")
    for split_name, items in splits.items():
        img_out = out_dir / split_name / "images"
        mask_out = out_dir / split_name / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for item in tqdm(items, desc=f"Processing {split_name}", leave=True):
            # Read
            img = cv2.imread(str(item["image"]))
            mask = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                logger.warning(f"Failed to read: {item['image'].name}")
                continue

            # Resize
            img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

            # Save
            fname = item["image"].stem
            cv2.imwrite(str(img_out / f"{fname}.png"), img_resized)
            cv2.imwrite(str(mask_out / f"{fname}.png"), mask_resized)

    logger.info(f"FloodNet processing complete → {out_dir}")
    return {
        "dataset": "FloodNet",
        "total_pairs": len(pairs),
        "splits": {k: len(v) for k, v in splits.items()},
        "class_distribution": class_dist,
    }


# ══════════════════════════════════════════════════════════════
# XBD PREPROCESSING
# ══════════════════════════════════════════════════════════════

def discover_xbd_disasters(raw_dir: Path) -> dict[str, list[str]]:
    """
    Scan xBD labels to find all disaster event names.
    Returns {disaster_name: [list of image stems]}.
    """
    labels_dir = raw_dir / "labels"
    if not labels_dir.exists():
        # Try alternate: labels might be alongside images
        labels_dir = raw_dir / "labels"
        if not labels_dir.exists():
            logger.error(f"xBD labels directory not found: {labels_dir}")
            return {}

    disasters = defaultdict(list)
    for label_file in sorted(labels_dir.glob("*.json")):
        try:
            with open(label_file, "r") as f:
                meta = json.load(f)
            # xBD JSON structure: metadata.disaster
            disaster_name = meta.get("metadata", {}).get("disaster", "unknown")
            # Stem format: disaster_00000_post_disaster.json → get the base
            stem = label_file.stem  # e.g., "hurricane-florence_00000001_post_disaster"
            disasters[disaster_name].append(stem)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse {label_file.name}: {e}")
            # Fallback: extract disaster name from filename
            parts = label_file.stem.split("_")
            if len(parts) >= 2:
                disaster_name = parts[0]
                disasters[disaster_name].append(label_file.stem)

    return dict(disasters)


def get_xbd_image_pairs(raw_dir: Path, stems: list[str]) -> list[dict]:
    """
    From a list of label stems, find matching pre/post image pairs and post label JSON.

    xBD naming convention:
        Image : {disaster}_{id}_pre_disaster.png / {disaster}_{id}_post_disaster.png
        Label : {disaster}_{id}_post_disaster.json   ← polygon annotations

    Masks are NOT shipped as PNGs; they are rasterised on-the-fly from the
    polygon WKT coordinates stored in each post-disaster JSON label.
    """
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"
    pairs = []
    seen = set()

    for stem in stems:
        # Extract the base ID (strip _pre/_post suffix)
        if "_post_disaster" in stem:
            base = stem.replace("_post_disaster", "")
        elif "_pre_disaster" in stem:
            base = stem.replace("_pre_disaster", "")
        else:
            continue

        if base in seen:
            continue
        seen.add(base)

        pre_img    = images_dir / f"{base}_pre_disaster.png"
        post_img   = images_dir / f"{base}_post_disaster.png"
        post_label = labels_dir  / f"{base}_post_disaster.json"

        if pre_img.exists() and post_img.exists() and post_label.exists():
            pairs.append({
                "base_id":     base,
                "pre":         pre_img,
                "post":        post_img,
                "post_label":  post_label,   # JSON with building polygons
            })

    return pairs


def rasterize_xbd_mask(post_label_path: Path, img_size: int = 1024) -> np.ndarray:
    """
    Rasterise building-damage polygons from an xBD post-disaster JSON label
    into a single-channel uint8 mask.

    Class mapping (matches XBD_CLASSES / SUBTYPE_TO_CLASS):
        0 = Background
        1 = No-Damage
        2 = Minor-Damage
        3 = Major-Damage
        4 = Destroyed

    The JSON stores two coordinate sets:
        features.lng_lat  – geographic WGS-84 coords (NOT used here)
        features.xy       – pixel coords in the 1024×1024 image space  ← used
    """
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    try:
        with open(post_label_path, "r") as f:
            label_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as exc:
        logger.warning(f"Failed to load label {post_label_path.name}: {exc}")
        return mask

    features = label_data.get("features", {}).get("xy", [])
    if not features:
        return mask

    for feat in features:
        props   = feat.get("properties", {})
        subtype = props.get("subtype", "")
        class_id = SUBTYPE_TO_CLASS.get(subtype, 0)
        if class_id == 0:
            continue

        wkt_str = feat.get("wkt", "")
        if not wkt_str:
            continue

        try:
            geom   = shapely_wkt.loads(wkt_str)
            coords = np.array(geom.exterior.coords, dtype=np.int32)
            # Clip to image bounds
            coords[:, 0] = np.clip(coords[:, 0], 0, img_size - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, img_size - 1)
            cv2.fillPoly(mask, [coords], int(class_id))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Skipping polygon in {post_label_path.name}: {exc}")
            continue

    return mask


def patch_image(img: np.ndarray, patch_size: int = 512) -> list[np.ndarray]:
    """Crop a 1024x1024 image into 4 non-overlapping 512x512 patches."""
    h, w = img.shape[:2]
    patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if y + patch_size <= h and x + patch_size <= w:
                patches.append(img[y:y+patch_size, x:x+patch_size])
    return patches


def has_building_pixels(mask_patch: np.ndarray, min_pixels: int = 50) -> bool:
    """Return True if a mask patch has at least *min_pixels* building pixels (classes 1-4)."""
    return int(np.sum((mask_patch >= 1) & (mask_patch <= 4))) >= min_pixels


def process_xbd(config: dict, dry_run: bool = True, auto_confirm: bool = False):
    """Main xBD preprocessing pipeline.

    Reads raw xBD images + JSON polygon labels, rasterises damage masks,
    patches 1024×1024 tiles into 512×512 crops, filters empty patches, and
    writes the processed dataset to out_dir/{train,val,test}/{pre,post,masks}/.
    """
    raw_dir = Path(config["data"]["raw"]["xbd"])
    out_dir = Path(config["data"]["processed"]["xbd"])
    patch_size = config["preprocessing"]["image_size"]
    filter_keywords = config["xbd"]["disaster_filter_keywords"]
    ratios = config["preprocessing"]["split_ratios"]
    img_original_size = config["preprocessing"].get("xbd_original_size", 1024)

    logger.info("\n" + "=" * 60)
    logger.info("XBD PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"Raw data directory : {raw_dir}")
    logger.info(f"Output directory   : {out_dir}")
    logger.info(f"Patch size         : {patch_size}x{patch_size}")
    logger.info(f"Filter keywords    : {filter_keywords}")
    logger.info(f"Mask source        : rasterised from JSON polygon labels")

    # ── Step 1: Discover disaster events ──
    if not raw_dir.exists() or not (raw_dir / "images").exists():
        logger.warning(f"xBD raw directory not ready: {raw_dir}")
        logger.warning("Skipping xBD preprocessing. Re-run when data is available.")
        return None

    disasters = discover_xbd_disasters(raw_dir)
    if not disasters:
        logger.error("No disaster events found in xBD labels.")
        return None

    logger.info(f"\nFound {len(disasters)} disaster events:")
    logger.info("┌─────────────────────────────────────┬────────┐")
    logger.info("│ Disaster Name                       │ Images │")
    logger.info("├─────────────────────────────────────┼────────┤")
    for name, stems in sorted(disasters.items()):
        marker = " ✓" if any(kw in name.lower() for kw in filter_keywords) else ""
        logger.info(f"│ {name:<35s} │ {len(stems):>5d} │{marker}")
    logger.info("└─────────────────────────────────────┴────────┘")

    # ── Step 2: Filter to flood/hurricane events ──
    filtered_disasters = {
        name: stems for name, stems in disasters.items()
        if any(kw in name.lower() for kw in filter_keywords)
    }

    if not filtered_disasters:
        logger.error("No flood/hurricane disasters found after filtering.")
        return None

    logger.info(f"\nFiltered to {len(filtered_disasters)} disaster events: "
                f"{list(filtered_disasters.keys())}")

    if not auto_confirm and not dry_run:
        response = input("\nProceed with these disasters? [Y/n]: ").strip().lower()
        if response == "n":
            logger.info("Aborted by user.")
            return None

    # ── Step 3: Find pre/post pairs ──
    all_stems = []
    for stems in filtered_disasters.values():
        all_stems.extend(stems)

    pairs = get_xbd_image_pairs(raw_dir, all_stems)
    logger.info(f"\nFound {len(pairs)} pre/post image pairs with masks")

    # ── Step 4: Patch and filter ──
    logger.info("\nPatching images and filtering empty patches...")
    patched_data = []
    discarded = 0
    kept = 0

    limit = min(50, len(pairs)) if dry_run else len(pairs)

    for item in tqdm(pairs[:limit], desc="Patching & rasterising", leave=True):
        pre  = cv2.imread(str(item["pre"]))
        post = cv2.imread(str(item["post"]))

        if pre is None or post is None:
            logger.warning(f"Cannot read images for {item['base_id']}")
            continue

        # Rasterise mask from JSON polygon annotations
        mask = rasterize_xbd_mask(item["post_label"], img_size=img_original_size)

        pre_patches  = patch_image(pre,  patch_size)
        post_patches = patch_image(post, patch_size)
        mask_patches = patch_image(mask, patch_size)

        for i, (pp, pop, mp) in enumerate(zip(pre_patches, post_patches, mask_patches)):
            if has_building_pixels(mp):
                patched_data.append({
                    "base_id":    item["base_id"],
                    "patch_idx": i,
                    "pre":        pp   if not dry_run else None,
                    "post":       pop  if not dry_run else None,
                    "mask":       mp   if not dry_run else None,
                    # Keep source paths so we can re-rasterise in dry-run write path
                    "pre_path":        item["pre"],
                    "post_path":       item["post"],
                    "post_label_path": item["post_label"],
                })
                kept += 1
            else:
                discarded += 1

    logger.info(f"\nPatching results ({'sampled ' + str(limit) if dry_run else 'full'}):")
    logger.info(f"  Patches kept     : {kept}")
    logger.info(f"  Patches discarded: {discarded} (no building pixels)")
    logger.info(f"  Keep rate        : {kept/(kept+discarded)*100:.1f}%" if (kept+discarded) > 0 else "  N/A")

    # ── Step 5: Class distribution ──
    class_pixels = Counter()
    for item in tqdm(patched_data[:50], desc="Class distribution", leave=False):
        if dry_run:
            # Re-rasterise for distribution sampling (no disk mask available)
            mask = rasterize_xbd_mask(
                item["post_label_path"], img_size=img_original_size
            )
            # Sample only the relevant patch quadrant
            h, w = img_original_size, img_original_size
            row = (item["patch_idx"] // (w // patch_size)) * patch_size
            col = (item["patch_idx"] %  (w // patch_size)) * patch_size
            mask = mask[row:row+patch_size, col:col+patch_size]
        else:
            mask = item["mask"]
        if mask is None:
            continue
        unique, counts = np.unique(mask, return_counts=True)
        for cls_id, count in zip(unique, counts):
            if cls_id < 5:
                class_pixels[cls_id] += int(count)

    total_px = sum(class_pixels.values())
    logger.info("\n┌───────────────────────────────────────────────────┐")
    logger.info("│              xBD Class Distribution               │")
    logger.info("├──────┬───────────────────────┬─────────┬──────────┤")
    logger.info("│  ID  │ Class Name            │ Pixels  │ Percent  │")
    logger.info("├──────┼───────────────────────┼─────────┼──────────┤")
    for cls_id in range(5):
        name = XBD_CLASSES.get(cls_id, f"Unknown-{cls_id}")
        px = class_pixels.get(cls_id, 0)
        pct = (px / total_px * 100) if total_px > 0 else 0
        logger.info(f"│  {cls_id:2d}  │ {name:<21s} │ {px:>7,d} │ {pct:>6.2f}%  │")
    logger.info("└──────┴───────────────────────┴─────────┴──────────┘")

    if dry_run:
        logger.info("\n[DRY RUN] No files written. Re-run without --dry-run to process.")
        return {
            "dataset": "xBD",
            "total_pairs": len(pairs),
            "patches_kept": kept,
            "patches_discarded": discarded,
            "class_distribution": dict(class_pixels),
        }

    # ── Step 6: Split by disaster event (avoid data leakage!) ──
    logger.info("\nSplitting by disaster event (to prevent data leakage)...")

    # Group patches by disaster event
    event_groups = defaultdict(list)
    for item in patched_data:
        # Extract disaster name from base_id (e.g., "hurricane-florence_00000001")
        disaster = item["base_id"].rsplit("_", 1)[0] if "_" in item["base_id"] else item["base_id"]
        # Simpler: use the disaster name prefix
        for dname in filtered_disasters.keys():
            if item["base_id"].startswith(dname):
                disaster = dname
                break
        event_groups[disaster].append(item)

    # Split events (not individual patches)
    event_names = list(event_groups.keys())
    if len(event_names) >= 3:
        train_events, temp_events = train_test_split(
            event_names, test_size=(ratios["val"] + ratios["test"]),
            random_state=42
        )
        val_frac = ratios["val"] / (ratios["val"] + ratios["test"])
        val_events, test_events = train_test_split(
            temp_events, test_size=(1 - val_frac), random_state=42
        )
    else:
        # Too few events — fall back to random split of patches
        logger.warning("Too few disaster events for event-based split. Using random split.")
        train_events = event_names
        val_events = event_names
        test_events = event_names
        # Do a random split instead
        indices = list(range(len(patched_data)))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        splits = {
            "train": [patched_data[i] for i in train_idx],
            "val": [patched_data[i] for i in val_idx],
            "test": [patched_data[i] for i in test_idx],
        }
        # Jump to writing
        _write_xbd_splits(splits, out_dir, patch_size=patch_size,
                          img_original_size=img_original_size)
        return

    splits = {
        "train": [p for e in train_events for p in event_groups[e]],
        "val":   [p for e in val_events   for p in event_groups[e]],
        "test":  [p for e in test_events  for p in event_groups[e]],
    }

    logger.info(f"  Train events: {train_events} ({len(splits['train'])} patches)")
    logger.info(f"  Val events  : {val_events} ({len(splits['val'])} patches)")
    logger.info(f"  Test events : {test_events} ({len(splits['test'])} patches)")

    _write_xbd_splits(splits, out_dir, patch_size=patch_size,
                      img_original_size=img_original_size)

    return {
        "dataset": "xBD",
        "total_pairs": len(pairs),
        "patches_kept": kept,
        "patches_discarded": discarded,
        "class_distribution": dict(class_pixels),
    }


def _write_xbd_splits(splits: dict, out_dir: Path, patch_size: int = 512,
                      img_original_size: int = 1024):
    """Write xBD patches to disk.

    When array data is already in memory (non-dry-run path), it is written
    directly.  When only paths are stored (dry-run path), images are re-read
    and masks re-rasterised from the JSON labels.
    """
    for split_name, items in splits.items():
        pre_out  = out_dir / split_name / "pre"
        post_out = out_dir / split_name / "post"
        mask_out = out_dir / split_name / "masks"
        pre_out.mkdir(parents=True, exist_ok=True)
        post_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for item in tqdm(items, desc=f"Writing {split_name}", leave=True):
            fname = f"{item['base_id']}_p{item['patch_idx']}.png"
            idx   = item["patch_idx"]

            if item["pre"] is not None:
                # Arrays already in memory
                cv2.imwrite(str(pre_out  / fname), item["pre"])
                cv2.imwrite(str(post_out / fname), item["post"])
                cv2.imwrite(str(mask_out / fname), item["mask"])
            else:
                # Re-read images and re-rasterise mask from JSON label
                pre  = cv2.imread(str(item["pre_path"]))
                post = cv2.imread(str(item["post_path"]))
                mask = rasterize_xbd_mask(
                    item["post_label_path"], img_size=img_original_size
                )
                if pre is None or post is None:
                    continue

                pre_patches  = patch_image(pre,  patch_size)
                post_patches = patch_image(post, patch_size)
                mask_patches = patch_image(mask, patch_size)

                if idx < len(pre_patches):
                    cv2.imwrite(str(pre_out  / fname), pre_patches[idx])
                    cv2.imwrite(str(post_out / fname), post_patches[idx])
                    cv2.imwrite(str(mask_out / fname), mask_patches[idx])

    logger.info(f"xBD processing complete → {out_dir}")


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════

def print_final_summary(results: list[dict]):
    """Print a combined summary table for all datasets."""
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    for r in results:
        if r is None:
            continue
        logger.info(f"\n📊 {r['dataset']}:")
        if "total_pairs" in r:
            logger.info(f"   Total images (with masks): {r['total_pairs']}")
        if "patches_kept" in r:
            logger.info(f"   Patches kept / discarded : {r['patches_kept']} / {r['patches_discarded']}")
        if "splits" in r:
            for split, count in r["splits"].items():
                logger.info(f"   {split:<6s}: {count} images")
        if "class_distribution" in r:
            total = sum(r["class_distribution"].values())
            classes = FLOODNET_CLASSES if r["dataset"] == "FloodNet" else XBD_CLASSES
            logger.info(f"   Class distribution (sampled):")
            for cls_id, px in sorted(r["class_distribution"].items()):
                name = classes.get(cls_id, f"Class-{cls_id}")
                pct = px / total * 100 if total > 0 else 0
                logger.info(f"     {cls_id}: {name:<22s} {pct:>6.2f}%")


# ══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess FloodNet and xBD datasets for disaster-multiagent-ai"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing any files")
    parser.add_argument("--floodnet-only", action="store_true",
                        help="Process FloodNet only (skip xBD)")
    parser.add_argument("--xbd-only", action="store_true",
                        help="Process xBD only (skip FloodNet)")
    parser.add_argument("--auto-confirm", action="store_true",
                        help="Skip interactive confirmation prompts")
    args = parser.parse_args()

    config = load_config()
    results = []

    if not args.xbd_only:
        result = process_floodnet(config, dry_run=args.dry_run)
        results.append(result)

    if not args.floodnet_only:
        result = process_xbd(config, dry_run=args.dry_run, auto_confirm=args.auto_confirm)
        results.append(result)

    print_final_summary(results)

    if args.dry_run:
        logger.info("\n" + "─" * 60)
        logger.info("This was a DRY RUN. To write files, run again without --dry-run")
        logger.info("─" * 60)


if __name__ == "__main__":
    main()
