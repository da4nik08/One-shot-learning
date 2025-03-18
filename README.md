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
- [Evaluation](#evaluation)
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

Run the inference pipeline:

```bash
python inference.py --configs models_weights.yaml config_inference.yaml config_model.yaml config_train.yaml
```