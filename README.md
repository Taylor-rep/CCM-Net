# CCMNet: Implementation Guide

---

## 🛠️ 1. Environment Setup

We recommend using **Conda** to manage dependencies.

### Step 1: Create and Activate Environment
```bash
conda create -n ccmnet python=3.8 -y
conda activate ccmnet
```

### Step 2: Install PyTorch (CUDA 11.8)
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Mamba Components
```bash
pip install packaging
pip install causal-conv1d==1.1.0
pip install mamba-ssm==1.1.1
```

### Step 4: Install Other Dependencies
```bash
pip install timm==0.9.12 einops tensorboardX opencv-python matplotlib pillow
```
---
## 📂 2. Data Preparation

### Step 1: Download Datasets
Download the preprocessed ISIC2017, ISIC2018, and PH2 datasets from the link below and place them in the `./data/` directory.
- [Dataset Download Link](https://drive.google.com/file/d/1J6c2dDqX8qka1q4EtmTBA0w3Kez7-M6T/view?usp=sharing)

### Step 2: Organize Directory Structure
Ensure your data is structured as follows:
```text
./data/
├── isic2018/
│   ├── train/
│   │   ├── images/
│   │   ├── masks/
│   │   └── points_boundary2/
│   └── val/
│       ├── images/
│       └── masks/
├── isic2017/
│   ├── train/ ...
│   └── val/ ...
└── PH2/
    ├── train/ ...
    └── val/ ...
```

### Step 3: Generate Boundary Maps (Optional)
If `points_boundary2` is missing, run the following to generate them:
1. Update paths in `boundary.py`.
2. Run: `python boundary.py`

---

## 🚀 3. How to Run

### Training
1. Configure your settings in [config_setting.py].
2. Run training:
```bash
python train.py
```

