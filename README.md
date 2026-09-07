# 🌊 Disaster Multi-Agent AI

> A multi-agent AI system for post-disaster scene assessment, combining computer vision agents with an LLM-based coordinator for automated situational reporting.

---

## Overview

This project implements an **agentic multi-agent architecture** for disaster scene understanding. Three specialized AI agents analyze satellite/aerial imagery for flood extent, building damage, and road accessibility, while an LLM-based Coordinator Agent orchestrates their execution and synthesizes structured situational reports for emergency responders.

**Key capabilities:**
- 🏠 10-class flood scene segmentation (FloodNet)
- 🏗️ Pre/post-disaster building damage classification (xBD/xView2)
- 🛣️ Road accessibility assessment with transport recommendations
- 🤖 LLM-orchestrated pipeline with natural language querying

---

## Architecture Diagram

<!-- TODO: Embed architecture diagram image here -->

| Agent | Role | Model Type |
|-------|------|------------|
| **Flood Segmentation** | Per-pixel classification of flood scenes | U-Net (ResNet34 encoder) |
| **Building Damage** | Pre/post damage severity segmentation | Siamese ResNet34 |
| **Geospatial Reasoning** | Road access rating from flood masks | Rule-based heuristics |
| **Coordinator** | Orchestrates agents, synthesizes reports | Claude Sonnet (Anthropic API) |

---

## Datasets

### FloodNet
- **Paper:** [FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding](https://arxiv.org/abs/2012.02951)
- **Classes:** 10 (Background, Building-Flooded, Building-Non-Flooded, Road-Flooded, Road-Non-Flooded, Water, Tree, Vehicle, Pool, Grass)
- **Usage:** Training Agent 1 (Flood Segmentation)

### xBD (xView2)
- **Paper:** [xBD: A Dataset for Assessing Building Damage from Satellite Imagery](https://arxiv.org/abs/1911.09296)
- **Classes:** 5 damage levels (Background, No-Damage, Minor-Damage, Major-Damage, Destroyed)
- **License:** CC BY-NC-SA 3.0
- **Usage:** Training Agent 2 (Building Damage)

---

## Setup

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended, 6GB+ VRAM)
- Anthropic API key (for Coordinator Agent)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/disaster-multiagent-ai.git
cd disaster-multiagent-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### Data Setup

1. Download FloodNet dataset and place in `data/raw/floodnet/` (with `images/` and `masks/` subdirectories)
2. Download xBD dataset and place in `data/raw/xbd/train/` (with `images/`, `labels/`, `targets/` subdirectories)
3. Run preprocessing:

```bash
# Dry-run first to verify counts
python scripts/prepare_data.py --dry-run

# Full preprocessing
python scripts/prepare_data.py
```

---

## Running the Pipeline

```bash
# Step 1: Train Flood Segmentation Agent (smoke test)
python agents/flood_segmentation/train.py --epochs 5

# Step 2: Train Building Damage Agent (smoke test)
python agents/building_damage/train.py --epochs 5

# Step 3: Run Coordinator on a sample zone
python scripts/run_coordinator.py --zone zone_001 --pre <pre_image> --post <post_image>

# Step 4: Interactive Q&A over reports
python scripts/query_cli.py

# Step 5: Full demo pipeline
python scripts/run_full_demo.py
```

---

## Results

<!-- TODO: Fill in actual metrics after training -->

| Model | Metric | Value |
|-------|--------|-------|
| Flood Segmentation | mIoU | — |
| Building Damage | Localization F1 | — |
| Building Damage | Damage F1 | — |
| Building Damage | Combined (xView2) | — |

See [DEMO_SUMMARY.md](outputs/demo/DEMO_SUMMARY.md) for visual results and sample reports.

---

## Limitations & Future Work

- **Geospatial Agent:** Currently rule-based (pixel-ratio thresholds). Future work: OSMnx/NetworkX graph-based routing for connectivity analysis between rescue staging points.
- **Information Retrieval:** External knowledge agent (weather feeds, population data) is out of scope for this prototype.
- **Video / Sensor Fusion:** Only static imagery is processed; temporal video analysis and multi-sensor fusion are future extensions.
- **Single-Region Validation:** Models are validated on FloodNet/xBD splits only; generalization to new geographies is unverified.
- **Real-time Processing:** Pipeline is batch-mode; streaming inference for live disaster feeds requires architectural changes.

---

## License

- **Code:** MIT License (see [LICENSE](LICENSE))
- **xBD Dataset:** CC BY-NC-SA 3.0 — [xView2 Challenge](https://xview2.org/)
- **FloodNet Dataset:** See [FloodNet repository](https://github.com/BinaLab/FloodNet-Supervised_v1.0) for license terms

---

## Citation

If you use this work, please cite:

```bibtex
@misc{disaster-multiagent-ai,
  title={Multi-Agent Disaster Scene Understanding},
  year={2026},
  url={https://github.com/<your-username>/disaster-multiagent-ai}
}
```
