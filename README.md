# One-Shot Learning with Deep Learning

## Introduction
This project implements **One-Shot Learning** using deep learning techniques. It leverages **ResNet18 with Spatial Group-wise Enhancements (SGE)** for feature extraction and employs metric learning strategies such as **Triplet Loss**. The training process includes both classification-based pretraining and metric-based fine-tuning.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
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

