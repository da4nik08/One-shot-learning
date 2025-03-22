# One-Shot Learning with Deep Learning

## Introduction
This project implements **One-Shot Learning** using deep learning techniques. It leverages **ResNet18 with Spatial Group-wise Enhancements (SGE)** for feature extraction and employs metric learning strategies such as **Triplet Loss**. The training process includes both classification-based pretraining and metric-based fine-tuning.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Features](#features)
- [Configuration](#configuration)
- [Training Process](#training-process)
- [Inference](#inference)
- [Results](#results)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

## Installation
To set up the environment, run:

```bash
pip install -r requirements.txt
```

## Usage
Run the training pipeline:

```bash
python train.py --configs config_pretrain.yaml config_model.yaml
```

Run the creation database pipeline:

```bash
python siamese_features_db.py --configs models_weights.yaml config_train.yaml
```

## Datasets
[Military Decision-Making Dataset](https://www.kaggle.com/datasets/nzigulic/military-equipment)
The dataset comprises 11,800 images and labels tailored for the YOLO detection algorithm, categorizing objects as follows:
1) Tank (TANK)
2) Infantry fighting vehicle (IFV)
3) Armored personnel carrier (APC)
4) Engineering vehicle (EV)
5) Assault helicopter (AH)
6) Transport helicopter (TH)
7) Assault airplane (AAP)
8) Transport airplane (TA)
9) Anti-aircraft vehicle (AA)
10) Towed artillery (TART)
11) Self-propelled artillery (SPART)
This dataset is clear (has good quality images from different angles, the equipment is clearly visible and poorly disguised), so it will be used for pretraining model

## Features
- One-Shot Learning with Triplet Loss
- ResNet18 with SGE or CBAM attention module for feature extraction
- Data augmentation by torchvision
- Triplets are created using hard batching
- Custom dataset handling with CustomDataset and TripletDataset
- Logging and Metrics tracking (TensorBoard supported)
- YAML-based configurations for flexibility
- EDA was introduced as jupyter-notebooks

## Configuration
Modify YAML files in the *configs/* directory to adjust:

- Model hyperparameters
- Training settings
- Preprocessing settings

## Training Process

1. Pretraining: Classification-based learning on the big dataset.
2. Fine-Tuning: Training real dataset the Siamese Network with Triplet Loss.

## Inference
To obtain results run inference pipeline:

```bash
python inference.py --configs models_weights.yaml config_inference.yaml config_model.yaml config_train.yaml
```

## Metrics
- Training and Validation Loss
- Accuracy over Validation Set
- Recall and Precision over Validation Set
- Similarity Score Distributions

## Results
Training loss and accuracy graphs will be added here:

## Dependencies
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Pandas
- scikit-learn
- tqdm
- PIL

## Troubleshooting
1. CUDA issues? Ensure PyTorch is installed with GPU support.
2. Dataset errors? Check dataset paths in configs/config_train.yaml.