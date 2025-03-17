import numpy as np
import os
import torch
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from src.classification_model_pretrain.custom_dataset import CustomDataset
from src.inference_model.batch_loop import batch_inference
from models import ResNet18WithSGEFeatureExtractor
from preprocessing import SiameseTransform, preprocessing_inference
from utilities import clear_cache, save_pkl, get_pkl, parse_configs 


def inference():
    CONFIGS_PATH = "configs/"
    configs = parse_configs(CONFIGS_PATH)

    config = configs["models_weights"]
    config_inf = configs["config_inference"]
    config_train = configs["config_train"]
    model_config = configs["config_model"]
    
    best_pretrain_model = os.path.join(config['paths']['pretrain_path'], config['weights']['best_pretrained_model'])

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