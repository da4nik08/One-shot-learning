import numpy as np
import os
from PIL import Image
from tqdm import tqdm, tqdm_notebook


def custom_transform(img, rescale_size):
    image = img.resize((rescale_size, rescale_size), resample=Image.BILINEAR)
    return image


def crop_and_resize(img, bbox, rescale_size):
    # bbox = (center_x, center_y, width, height) in YOLO format
    img_width, img_height = img.size
    center_x, center_y, bbox_width, bbox_height = bbox

    # Convert YOLO bbox format (relative) to absolute pixel coordinates
    center_x = int(center_x * img_width)
    center_y = int(center_y * img_height)
    bbox_width = int(bbox_width * img_width)
    bbox_height = int(bbox_height * img_height)

    # Compute the bounding box corners
    left = max(0, center_x - bbox_width // 2)
    top = max(0, center_y - bbox_height // 2)
    right = min(img_width, center_x + bbox_width // 2)
    bottom = min(img_height, center_y + bbox_height // 2)

    # Crop and resize the image
    cropped_img = img.crop((left, top, right, bottom))
    resized_img = custom_transform(cropped_img, rescale_size)
    return resized_img


def preprocessing(image_folder, labels_folder, rescale_size):
    # Initialize lists
    train_images = []
    train_labels = []
    
    # Process each image and its corresponding label file
    for filename in tqdm(os.listdir(image_folder)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # File paths
            file_path = os.path.join(image_folder, filename)
            label_file = os.path.join(labels_folder, filename.split('.')[0] + ".txt")
            
            try:
                # Open the image
                img = Image.open(file_path)
    
                # Check if the corresponding label file exists
                if not os.path.exists(label_file):
                    print(f"Label file missing for {filename}, skipping.")
                    continue
                
                # Read the label file
                with open(label_file, 'r') as file:
                    lines = file.readlines()
    
                # If no bounding boxes are found, skip the image
                if len(lines) == 0:
                    print(f"No bounding boxes found in {label_file}, skipping.")
                    continue
    
                # Process each bounding box
                for line in lines:
                    # Parse the YOLO format: class_id center_x center_y width height
                    parts = line.strip().split()
                    class_label = int(parts[0])
                    bbox = list(map(float, parts[1:]))
    
                    # If the image has multiple objects, crop it
                    if len(lines) > 1:
                        cropped_img = crop_and_resize(img, bbox, rescale_size)
                        train_images.append(cropped_img)
                        train_labels.append(class_label)
                    else:
                        # If only one object, resize the whole image
                        resized_img = custom_transform(img, rescale_size)
                        train_images.append(resized_img)
                        train_labels.append(class_label)
    
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return train_images, train_labels
    

    