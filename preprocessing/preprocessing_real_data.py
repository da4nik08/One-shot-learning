from tqdm import tqdm, tqdm_notebook
from PIL import Image
import numpy as np
import pandas as pd
import os


def custom_transform(img, rescale_size):
    image = img.resize((rescale_size, rescale_size), resample=Image.BILINEAR)
    return image


def preprocessing_real_ds(dataframe):
    index_list = []
    image_list = []
    label_list = []
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        if row['folder'] == 'img_russia':
            img_path = os.path.join(folder_base, folder_img_ru[0])
        else:
            img_path = os.path.join(folder_base, folder_img_ru[1])
    
        img_path = os.path.join(img_path, row['equipment'], row['file'])
    
        # Check if the corresponding label file exists
        if not os.path.exists(img_path):
            print(f"Label file missing for {img_path}, skipping.")
            continue
        
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            resized_img = custom_transform(img, RESCALE_SIZE)
            image_list.append(resized_img)
            label_list.append(row[target_name])
            index_list.append(index)
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return image_list, label_list, index_list