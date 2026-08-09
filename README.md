# 🛡️ StegaPoison: Stealthy Untargeted Poisoning Attack in Federated Recommendation Systems

This repository contains the complete implementation of **StegaPoison** (an untargeted stealthy poisoning attack against federated recommendation systems) evaluated against standard federated recommendation defense strategies.

---

## 📌 Overview

Federated Recommendation Systems (FRS) enable collaborative model training across decentralized user clients while keeping personal interaction data local. However, FRS remains vulnerable to malicious clients injecting poisoned updates into item embeddings.

- **StegaPoison Attack**: Injects watermarked, stealthy updates using Mirror Shift, Low-Variance Dimension Embedding Perturbation (LVDEP), Velocity-Based Sampling with Momentum, and Statistical Invisibility constraints to maximize attack efficacy across training rounds without decaying.

---

## 📂 Repository Structure

```
.
├── Data/                       # Datasets & data preparation scripts
│   ├── prepare.sh              # Shell script to download and process datasets
│   ├── ml-1m/                  # MovieLens-1M dataset files
│   ├── gowalla_10core.tsv      # Preprocessed Gowalla dataset
│   ├── item_list.txt           # Item mapping
│   └── user_list.txt           # User mapping
├── code/                       # Source code directory
│   ├── train.py                # Single experiment training script
│   ├── train_all.py            # Batch script to run full training suite
│   ├── test.py                 # Single experiment evaluation script
│   ├── eval_all.py             # Batch evaluation script for all trained checkpoints
│   ├── orchestra.py            # Main federated orchestration loop
│   ├── client.py               # Benign client and local training logic
│   ├── model.py                # Model architectures (MF, SASRec)
│   ├── dataset.py              # PyTorch Dataset utilities
│   ├── agg.py                  # Aggregation & Defense algorithms (FedAdam, Krum, TrimmedMean, etc.)
│   ├── utils.py                # Helper utilities and vector operations
│   ├── verify_stegapoison_math.py # Unit test for StegaPoison attack formulation
│   └── attacker/               # Attacker implementations
│       ├── stegapoison.py      # Fixed, un-decaying StegaPoison implementation
│       └── kmeans.py           # Clustering helpers for attack initialization
├── test_fixed_version.sh       # Automated verification and quick-test script
└── README.md                   # Project documentation
```

---

## ⚙️ Requirements & Installation

### 1. Environment Setup

Python 3.8+ and PyTorch 1.12+ (or PyTorch 2.0+) are required.

```bash
# Clone the repository
git clone https://github.com/srobitan/StegaPoison.git
cd StegaPoison

# Install dependencies
pip install torch numpy scipy tqdm
```

### 2. Prepare Datasets

Run the automated data preparation script to download MovieLens-1M and raw Gowalla check-in data:

```bash
cd Data
bash prepare.sh
cd ..
```

---

## 🚀 How to Run Full Training

### Option A: Run a Single Training Experiment

To train a specific combination of dataset, model architecture, and defense aggregator (e.g., Matrix Factorization on MovieLens-1M with FedAdam):

```bash
cd code

python3 train.py \
    --EXP_NAME train6000_ml_MF_stegapoison_FedAdam \
    --MODEL_TYPE MF \
    --DATA ml \
    --SEED 0 \
    --AGG_TYPE FedAdam \
    --ATTACKER_RATIO 0.05 \
    --ATTACKER_STRAT StegaPoison \
    --MAX_ROUND 6000 \
    --SAVE_ROUND 200 \
    --LOG_ROUND 100 \
    --LR 2e-3 \
    --SCALE 1.0
```

#### Key Arguments:
- `--MODEL_TYPE`: Backbone model (`MF` or `SASRec`).
- `--DATA`: Dataset (`ml` for MovieLens-1M, `gowalla` for Gowalla 10-core).
- `--AGG_TYPE`: Aggregation / Defense strategy (`FedAdam`, `TrimmedMean`, `Krum`, `MultiKrum`, `NormBound`, `FLWBC`, `MultiKrumUNION`, `NormBoundUNION`).
- `--ATTACKER_RATIO`: Ratio of malicious clients (default: `0.05` for 5%).
- `--ATTACKER_STRAT`: Attack strategy (`StegaPoison`).
- `--MAX_ROUND`: Total federated communication rounds (e.g., `6000`).
- `--SAVE_ROUND`: Interval rounds at which model checkpoints are saved.

---

## 📋 How to Test & Evaluate Models

### Option A: Evaluate a Single Experiment

After training completes, evaluate top-K recommendation metrics (HR@5, nDCG@5, HR@10, nDCG@10, HR@20, nDCG@20) across all saved checkpoints:

```bash
cd code

python3 test.py \
    --EXP_NAME train6000_ml_MF_stegapoison_FedAdam \
    --MODEL_TYPE MF \
    --DATA ml \
    --SEED 0 \
    --MAX_ROUND 6000 \
    --SAVE_ROUND 200
```

### Option B: Evaluate All Trained Checkpoints

To evaluate every trained model run in `model_all/` and generate a summary report:

```bash
cd code

python3 eval_all.py --MAX_ROUND 6000
```

---

## 🧪 Mathematical Unit Testing & Soundness Verification

You can verify the mathematical correctness of the StegaPoison attack formulation using the included unit test:

```bash
# Verify StegaPoison attack formulation (Watermarking, LVDEP, Mirror Shift, Momentum Velocity)
python3 code/verify_stegapoison_math.py
```

---

## ⚡ Quick Test Script

To run a fast 200-round validation test and verify that attack decay does not occur:

```bash
./test_fixed_version.sh
```

---

## 🛡️ Supported Defense Aggregators

| Aggregator Name | Description |
| :--- | :--- |
| `FedAdam` | Standard Server Adam Optimization (Un-defended baseline) |
| `TrimmedMean` | Trims top & bottom updates per dimension |
| `Krum` | Distance-based robust aggregation |
| `MultiKrum` | Multi-selection variant of Krum |
| `MultiKrumUNION` | Union selection variant of MultiKrum |
| `NormBound` | Gradient norm clipping defense |
| `NormBoundUNION` | Union variant of NormBound defense |
| `FLWBC` | Federated learning with bounding constraints |

---

## 📄 License

This project is licensed under the MIT License.
