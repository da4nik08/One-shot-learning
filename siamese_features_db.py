import numpy as np
import pandas as pd
import os
import torch
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from src.classification_model_pretrain.custom_dataset import CustomDataset
from models import ResNet18WithSGEFeatureExtractor
from utilities import load_config, clear_cache, save_pkl


def save_siamese_features_db():
    CONFIGS_PATH = "configs/"
    config = load_config(CONFIGS_PATH, "models_weights.yaml")
    
    best_pretrain_model = os.path.join(config['paths']['pretrain_path'], config['weights']['best_pretrained_model'])
    siamese_config = load_config(CONFIGS_PATH, "config_train.yaml")
    
    capt_df = get_captured(siamese_config['dataset']['path_img_metadata_ru'], siamese_config['dataset']['target_name'])
    images, labels, _ = preprocessing_real_ds(capt_df)

    le = LabelEncoder()
    le.fit(list(set(labels)))
    enc_labels = le.transform(labels)
    encoder_path = os.path.join(config['paths']['encoder_path'], config['encoder']['enc_name'])
    save_pkl(encoder_path, le)
    
    transform = SiameseTransform(siamese_config['dataset']['MEAN'], siamese_config['dataset']['STD'], mode='val', 
                                 DATA_MODES=siamese_config['dataset']['DATA_MODES'])
    
    dataset = CustomDataset(images, enc_labels, 'val', transform, siamese_config['dataset']['DATA_MODES'])
    dataloader = DataLoader(dataset, batch_size=siamese_config['dataloader']['batch_size'], 
                                num_workers=siamese_config['dataloader']['num_workers'])

    model = ResNet18WithSGEFeatureExtractor(num_classes=model_config['model']['num_classes'], 
                                            groups=model_config['model']['groups'], 
                                            pretrained=model_config['model']['pretrained'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 

    best_model = os.path.join(config['paths']['train_path'], config['weights']['best_model'])
    checkpoint = torch.load(best_model, map_location=device)
    model.load_state_dict(checkpoint)
    
    generate_siamese_embeddings_db(model, dataloader, config)       # saves this db