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
from preprocessing import preprocessing, calculate_cw, preprocessing_real_ds, get_captured, train_test_split
from src.classification_model_pretrain.custom_dataset import CustomDataset
from models import ResNet18WithSGE
from models import ResNet18WithSGEFeatureExtractor
from utilities import load_config, clear_cache, save_pkl


def generate_siamese_embeddings_db(model, dataloader, config):
    model.eval()

    embeddings = []
    true_labels = []
    with torch.inference_mode():
        for i, vdata in enumerate(val_loader):
            vfeatures, vlabels = vdata
            vfeatures, vlabels = vfeatures.to(device), vlabels.to(device)
            
            y_pred = model(vfeatures)

            embeddings.append(y_pred.cpu())  
            true_labels.append(vlabels.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
    
    embeddings_path = os.path.join(config['paths']['embeddings_path'], config['img_db']['labels_name'])
    labels_path = os.path.join(config['paths']['embeddings_path'], config['img_db']['embeddings_name'])
    save_pkl(embeddings_path, embeddings)
    save_pkl(labels_path, true_labels)
    


    