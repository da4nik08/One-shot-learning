from tqdm import tqdm, tqdm_notebook
from PIL import Image
import numpy as np
import pandas as pd
import os


def custom_transform(img, rescale_size):
    image = img.resize((rescale_size, rescale_size), resample=Image.BILINEAR)
    return image


def preprocessing_inference(data_path, rescale_size=320):
    image_list = []

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        
        try:
            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                transformed_img = custom_transform(img, rescale_size)
                image_list.append(transformed_img)
                
        except Exception as e:
            print(f"Skipping {file_name}: {e}")
            
    return image_list