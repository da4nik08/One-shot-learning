import numpy as np
import os
import torch
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from src.classification_model_pretrain.custom_dataset import CustomDataset
from src.inference_model.batch_loop import batch_inference
from models import ResNet18WithSGEFeatureExtractor
from preprocessing import SiameseTransform, preprocessing_inference
from utilities import load_config, clear_cache, save_pkl, get_pkl


def inference():
    CONFIGS_PATH = "configs/"
    config = load_config(CONFIGS_PATH, "models_weights.yaml")
    config_inf = load_config(CONFIGS_PATH, "config_inference.yaml")
    config_train = load_config(CONFIGS_PATH, "config_train.yaml")
    model_config = load_config(CONFIGS_PATH, "config_model.yaml")
    config_inf = load_config(CONFIGS_PATH, "config_inference.yaml")
    
    best_pretrain_model = os.path.join(config['paths']['pretrain_path'], config['weights']['best_pretrained_model'])
    siamese_config = load_config(CONFIGS_PATH, "config_train.yaml")

    encoder_path = os.path.join(config['paths']['encoder_path'], config['encoder']['enc_name'])
    embeddings_path = os.path.join(config['paths']['embeddings_path'], config['img_db']['labels_name'])
    labels_path = os.path.join(config['paths']['embeddings_path'], config['img_db']['embeddings_name'])
    le = get_pkl(encoder_path)
    embeddings_db = get_pkl(embeddings_path)
    labels_db = get_pkl(labels_path)

    list_images = preprocessing_inference(config_inf['input_img_directory'], config_train['dataset'][' RESCALE_SIZE'])
    transform = SiameseTransform(config_train['dataset']['MEAN'], config_train['dataset']['STD'], mode='test', 
                                 DATA_MODES=config_train['dataset']['DATA_MODES'])
    dataset = CustomDataset(list_images, [], 'test', transform, config_train['dataset']['DATA_MODES'])
    dataloader = DataLoader(dataset, batch_size=config_inf['dataloader']['batch_size'], 
                            num_workers=config_inf['dataloader']['num_workers'])
    model = ResNet18WithSGEFeatureExtractor(num_classes=model_config['model']['num_classes'], 
                                            groups=model_config['model']['groups'], 
                                            pretrained=model_config['model']['pretrained'])
    
    embeddings = batch_inference(model, dataloader)
    labels = siamese_classification(embeddings, embeddings_db, labels_db)
    output = le.inverse_transform(labels)
    save_pkl(os.path.join(config_inf['paths']['input_img_directory'], "result"), output)