import os
import numpy as np
from tensorflow import load_img, img_to_array

def load_data(image_dir, distortion_mask_dir, image_size=(256, 256)):
    """
    Loads traffic images and corresponding distortion masks
    Assumes the distortion masks are saved as .npy files (not .png)
    """
    images = []
    distortion_masks = []
    
    # iterate over the images
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        img = load_img(image_path, target_size=image_size)
        img = img_to_array(img)
        images.append(img)

        # Load the corresponding distortion mask 
        mask_filename = filename.split('.')[0] + '_mask.npy'  
        mask_path = os.path.join(distortion_mask_dir, mask_filename)
        
        if os.path.exists(mask_path):
            mask_img = np.load(mask_path)  
            # Ensure the mask has the correct shape (height, width, 1)
            if mask_img.shape[:2] != image_size:
                mask_img = np.resize(mask_img, (*image_size, 1))  # Resize the mask if necessary
            distortion_masks.append(mask_img)
        else:
            print(f"Mask for {filename} not found! Using default mask.")
            # If no matching distortion mask, use a default (all zeros mask)
            distortion_masks.append(np.zeros((image_size[0], image_size[1], 1))) 

    # Convert to numpy arrays to match input
    return np.array(images), np.array(distortion_masks)
