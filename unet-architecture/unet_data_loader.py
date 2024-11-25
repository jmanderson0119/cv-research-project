import os
import numpy as np
from tensorflow import load_img, img_to_array

def load_data(radar_image_dir, distortion_mask_dir, image_size=(256, 256)):
    """
    Loads radar images and corresponding distortion masks.
    Ensures images have a matching mask or assigns a default one if not found.
    Assumes the distortion masks have the same filename as the radar images.
    
    """
    radar_images = []
    distortion_masks = []
    
    # iterate over the images
    for filename in os.listdir(radar_image_dir):
        radar_image_path = os.path.join(radar_image_dir, filename)
        radar_img = load_img(radar_image_path, target_size=image_size)
        radar_img = img_to_array(radar_img)
        radar_images.append(radar_img)

        # Load the corresponding distortion mask
        mask_filename = filename.split('.')[0] + '_mask.png' 
        mask_path = os.path.join(distortion_mask_dir, mask_filename)
        
        if os.path.exists(mask_path):
            mask_img = load_img(mask_path, target_size=image_size, color_mode='grayscale')
            mask_img = img_to_array(mask_img)
            distortion_masks.append(mask_img)
        else:
            print(f"Mask for {filename} not found! Using default mask.")
            # If no matching distortion mask, use a default
            distortion_masks.append(np.zeros((image_size[0], image_size[1], 1))) 
    
    # Convert to numpy arrays to match input
    return np.array(radar_images), np.array(distortion_masks)
