import numpy as np
import pandas as pd
import os
import torch
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm, tqdm_notebook
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from src.classification_model_pretrain.train_classification import train
from preprocessing import preprocessing, calculate_cw, preprocessing_real_ds, get_captured, train_test_split, SiameseTransform
from src.siamese_model_train.metrics_writer import MetricsWriter
from src.siamese_model_train.smetrics import Metrics
from src.siamese_model_train.triplet_dataset import TripletDataset
from src.siamese_model_train.triplet_loss import TripletLoss
from src.classification_model_pretrain.custom_dataset import CustomDataset
from models import ResNet18WithSGE
from models import ResNet18WithSGEFeatureExtractor
from utilities import load_config, clear_cache


def pretrain(CONFIGS_PATH):
    config = load_config(CONFIGS_PATH, "config_pretrain.yaml")
    model_config = load_config(CONFIGS_PATH, "config_model.yaml")

    train_images, train_labels = preprocessing(config['dataset']['image_folder_train'], 
                                               config['dataset']['labels_folder_train'], 
                                               config['dataset']['RESCALE_SIZE'])
    val_images, val_labels = preprocessing(config['dataset']['image_folder_val'], 
                                           config['dataset']['labels_folder_val'], config['dataset']['RESCALE_SIZE'])
    class_weights = calculate_cw(train_labels)
    
    transform_train = SiameseTransform(config['dataset']['MEAN'], config['dataset']['STD'], mode='train', 
                                 DATA_MODES=config['dataset']['DATA_MODES'])
    transform_val = SiameseTransform(config['dataset']['MEAN'], config['dataset']['STD'], mode='val', 
                                 DATA_MODES=config['dataset']['DATA_MODES'])

    train_dataset = CustomDataset(train_images, train_labels, 'train', transform_train, config['dataset']['DATA_MODES'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'], 
                                  num_workers=config['dataloader']['num_workers'], shuffle=True)
    val_dataset = CustomDataset(val_images, val_labels, 'val', transform_val, config['dataset']['DATA_MODES'])
    val_dataloader = DataLoader(val_dataset, batch_size=config['dataloader']['batch_size'],
                                num_workers=num_workers=config['dataloader']['num_workers'])

    model = ResNet18WithSGE(num_classes=model_config['model']['num_classes'], groups=model_config['model']['groups'], 
                            pretrained=model_config['model']['pretrained'])
    
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


def train_siamese(CONFIGS_PATH, checkpoint_path):
    siamese_config = load_config(CONFIGS_PATH, "config_train.yaml")
    model_config = load_config(CONFIGS_PATH, "config_model.yaml")
    
    capt_df = get_captured(siamese_config['dataset']['path_img_metadata_ru'], siamese_config['dataset']['target_name'])
    train_data, val_data = train_test_split(capt_df, siamese_config['dataset']['target_name'])
    
    train_images, train_labels, _ = preprocessing_real_ds(train_data)
    val_images, val_labels, _ = preprocessing_real_ds(val_data)

    le = LabelEncoder()
    le.fit(list(set(train_labels)))
    
    enc_tlabels = le.transform(train_labels)
    enc_vlabels = le.transform(val_labels)

    transform_train = SiameseTransform(siamese_config['dataset']['MEAN'], siamese_config['dataset']['STD'], mode='train', 
                                 DATA_MODES=siamese_config['dataset']['DATA_MODES'])
    transform_val = SiameseTransform(siamese_config['dataset']['MEAN'], siamese_config['dataset']['STD'], mode='val', 
                                 DATA_MODES=siamese_config['dataset']['DATA_MODES'])
    
    train_dataset = TripletDataset(train_images, enc_tlabels, 'train', transform_train, 
                                   siamese_config['dataset']['DATA_MODES'])
    train_dataloader = DataLoader(train_dataset, batch_size=siamese_config['dataloader']['batch_size'], 
                                  num_workers=siamese_config['dataloader']['num_workers'], shuffle=True)
    val_dataset = TripletDataset(val_images, enc_vlabels, 'val', transform_val, 
                                 siamese_config['dataset']['DATA_MODES'])
    val_dataloader = DataLoader(val_dataset, batch_size=siamese_config['dataloader']['batch_size'], 
                                num_workers=siamese_config['dataloader']['num_workers'])

    model = ResNet18WithSGEFeatureExtractor(num_classes=model_config['model']['num_classes'], 
                                            groups=model_config['model']['groups'], 
                                            pretrained=model_config['model']['pretrained'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    mw = MetricsWriter(model, model_name=siamese_config['train']['model_name'], 
                       save_treshold=siamese_config['train']['save_treshold'])
    loss_func = nn.TripletMarginWithDistanceLoss(margin=siamese_config['train']['margin'])
    optimizer = optim.AdamW(model.parameters(), lr=siamese_config['AdamW']['learning_rate'], 
                            weight_decay=siamese_config['AdamW']['weight_decay'])
    
    train(model, mw, loss_func, optimizer, train_dataloader, val_dataloader, 
          siamese_config['dataloader']['batch_size'], epochs=siamese_config['train']['epochs'])
    

def train():
    CONFIGS_PATH = "configs/"
    config = load_config(CONFIGS_PATH, "models_weights.yaml")
    pretrain(CONFIGS_PATH)
    best_pretrain_model = os.path.join(config['paths']['pretrain_path'], config['weights']['best_pretrained_model'])
    train_siamese(CONFIGS_PATH, best_pretrain_model)
    clear_cache()
    