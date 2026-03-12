# TPFM U-Net: Fluid Flow Prediction

[![Paper](https://img.shields.io/badge/Paper-DOI:10.14311/TPFM.2026.019-blue.svg)](https://doi.org/10.14311/TPFM.2026.019)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

This repository implements a U-Net architecture designed for predicting fluid flow fields (velocity and pressure) based on input topologies. It is the official implementation accompanying the research paper: **"Using U-Net to Estimate Fluid Flow in Mechanical Metamaterials"** ([DOI:10.14311/TPFM.2026.019](https://doi.org/10.14311/TPFM.2026.019)).

---

## Overview

The TPFM U-Net model takes a 2D topology (binary mask where 0 = wall, 1 = fluid) and predicts:
- **Velocity components** ($U_x, U_y, U_z$)
- **Pressure field** ($p$)

By leveraging deep learning, we achieve near-instant flow approximations that previously required expensive Computational Fluid Dynamics (CFD) simulations.

## Directory Structure

```text
.
├── configs/            # Hydra configuration files for models and datasets
├── data/               # Processed .npz datasets
├── notebooks/          # Research, visualization and testing notebooks
├── src/                # Core implementation (U-Net, Datamodule, Training)
├── model_checkpoints/  # Saved model weights (.ckpt)
└── requirements.txt    # Project dependencies
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/tpfm_unet.git
   cd tpfm_unet
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv ./venv
   source ./venv/bin/activate
   pip install -r requirements.txt
   ```

## Data Preparation

Because of the large size of the CFD datasets, we provide just a small sample in the `data/` directory.

The training script and dataset structures work with `.npz` files. If you have raw CFD data (in `.dat` format with a `coordinates.csv`), use the helper script:

```bash
python src/dataset_preparation.py
```

*Note: You may need to edit the file paths in `src/dataset_preparation.py` to match your data directory.*

## Training

We use [Hydra](https://hydra.cc/) for configuration management and [PyTorch Lightning](https://www.pytorchlightning.ai/) for training.

### Run Default Experiment
```bash
python src/train.py
```

### Run Specific Configuration
Configurations are stored in `configs/`. You can override them via CLI:
```bash
python src/train.py --config-name mixer_128_upscaled_x2
```

or specify individual parameters:
```bash
python src/train.py model.base_filters=64 model.depth=4
```

### Experiment Tracking
Experiments are automatically logged to **MLflow**. To view the UI:
```bash
mlflow ui -p 1234
```
Access the dashboard at `http://localhost:1234`.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{tpfm2026unet,
  title={Using U-Net to Estimate Fluid Flow in Mechanical Metamaterials},
  author={Ledl, M. and Kubíčková, L. and Isoz, M.},
  journal={Topical Problems of Fluid Mechanics},
  year={2026},
  volume={2026},
  number={019},
  doi={10.14311/TPFM.2026.019}
}
```

---
*Developed as part of the TPFM 2026 conference contributions.*