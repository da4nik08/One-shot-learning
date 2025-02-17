import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
from tqdm import tqdm, tqdm_notebook
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from src.classification_model_pretrain.train_classification import train
from preprocessing import preprocessing
from preprocessing import calculate_cw
from src.classification_model_pretrain.custom_dataset import CustomDataset
from models import ResNet18WithSGE
from utilities import load_config


def main():
    CONFIGS_PATH = "configs/"
    config = load_config(CONFIGS_PATH, "config_pretrain.yaml")

    train_images, train_labels = preprocessing(config['dataset']['image_folder_train'], 
                                               config['dataset']['labels_folder_train'], config['dataset']['RESCALE_SIZE'])
    val_images, val_labels = preprocessing(config['dataset']['image_folder_val'], 
                                           config['dataset']['labels_folder_val'], config['dataset']['RESCALE_SIZE'])
    class_weights = calculate_cw(train_labels)
    
    transform_v1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomRotation(degrees=25),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.25),
        transforms.Normalize(config['dataset']['MEAN'], config['dataset']['STD'])
    ])

    train_dataset = CustomDataset(train_images, train_labels, 'train', transform_v1)
    train_dataloader = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'], 
                                  num_workers=config['dataloader']['num_workers'], shuffle=True)
    val_dataset = CustomDataset(val_images, val_labels, 'val', transform_v1)
    val_dataloader = DataLoader(val_dataset, batch_size=config['dataloader']['batch_size'],
                                num_workers=num_workers=config['dataloader']['num_workers'])

    model = ResNet18WithSGE(num_classes=config['model']['num_classes'], groups=config['model']['groups'], 
                            pretrained=config['model']['pretrained'])
    
    pretrained_params = []
    sge_params = []
    if config['model']['pretrained']==True:
        for name, param in model.named_parameters():
            if 'module' in name or 'fc' in name:
                sge_params.append(param)
            else:
                pretrained_params.append(param)
        optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': float(config['AdamW']['learning_rate_small']), 
                        'weight_decay': float(config['AdamW']['weight_decay_small'])},       # Smaller LR for pretrained layers
            {'params': sge_params, 'lr': float(config['AdamW']['learning_rate_large']), 
                        'weight_decay': float(config['AdamW']['weight_decay_large'])}        # Larger LR for CBAM layers
        ])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=float(config['AdamW']['learning_rate_large']), 
                                                    weight_decay=float(config['AdamW']['weight_decay_large']))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    model = model.to(device)

    train(model, criterion, optimizer, train_dataloader, val_dataloader, config['dataloader']['batch_size'], 
          config['train']['svs_path'], config['train']['log_path'], 
          save_treshold=config['train']['save_treshold'], 
          epochs=config['train']['epochs'], model_name=config['train']['model_name'])