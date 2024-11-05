from segmentation_models import Xnet

# prepare data
# x, y = ... # range in [0,1], the network expects input channels of 3

import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# Define dataset directories
base_dir = "/volumes/gilbert/dataset_rgb/sorted/"
# subfolders = ["salak-1/", "salak-1-2/", "salak-1-3/", "salak-1-4/", "salak-2/", "salak-3/"]
subfolders = ["salak-1-3/"]

# Collect all image and mask paths
image_paths = []
mask_paths = []
for folder in subfolders:
    folder_path = os.path.join(base_dir, folder, "cropped/")
    mask_folder_path = os.path.join(base_dir, folder, "mask/masks/cropped/")

    # Get list of images and masks in the folder
    images = [f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]
    masks = [f for f in os.listdir(mask_folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]

    # Create a dictionary to easily find masks by base filename
    mask_dict = {os.path.splitext(mask)[0]: os.path.join(mask_folder_path, mask) for mask in masks}

    # Loop through images and find matching masks
    for img_name in images:
        img_base = os.path.splitext(img_name)[0]  # Get the base name without extension
        image_path = os.path.join(folder_path, img_name)
        
        # Check if there’s a corresponding mask with the same base name
        if img_base in mask_dict:
            mask_path = mask_dict[img_base]
            image_paths.append(image_path)
            mask_paths.append(mask_path)

# Split dataset into training and validation sets (e.g., 80% train, 20% validation)
train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

# Function to load images and masks into numpy arrays
def load_data(image_paths, mask_paths, img_size=(256, 256)):
    x_data = []
    y_data = []
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        # Load and resize image
        image = Image.open(img_path).convert("RGB")
        # image = image.resize(img_size)
        x_data.append(np.array(image) / 255.0)  # Normalize image to [0, 1]

        # Load and resize mask
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
        # mask = mask.resize(img_size)
        mask = np.array(mask) / 255.0  # Normalize mask to [0, 1]
        mask = (mask > 0.5).astype(np.float32)  # Binarize mask to 0, 1
        y_data.append(mask)

    x_data = np.array(x_data).astype(np.float32)  # Shape: (num_samples, H, W, 3)
    y_data = np.array(y_data).astype(np.float32)  # Shape: (num_samples, H, W)
    y_data = np.expand_dims(y_data, axis=-1)      # Shape to (num_samples, H, W, 1)
    return x_data, y_data

# Load training and validation data
x_train, y_train = load_data(train_image_paths, train_mask_paths)
x_val, y_val = load_data(train_image_paths, train_mask_paths)

print(len(x_train), len(y_train))
print(x_train[1], y_train[1])
print(len(x_val), len(y_val))
print(x_val[1], y_val[1])

# Data is now ready to be used with your UNet++ model

# resnext50, densenet201, densenet201
# prepare model
model = Xnet(backbone_name='resnext50', encoder_weights='imagenet', decoder_block_type='transpose', input_shape=(256,256,3)) # build UNet++

model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

# # train model
model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, epochs=20)