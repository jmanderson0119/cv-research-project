import numpy as np
import tensorflow as tf

def process_input(rgb_images, distortion_model):
    """
    Processes 4-channel input (RGB + Mask) for U-Net
    """
    # normalize
    rgb_images = rgb_images / 255.0 if rgb_images.max() > 1 else rgb_images

    # use distortion detection model to generate masks
    distortion_mask = distortion_model.predict(rgb_images)

    # resize mask (match input dimensions)
    batch_size, height, width, _ = rgb_images.shape
    distortion_mask_resized = tf.image.resize(distortion_mask, size=(height, width))

    # concatenate RGB and mask for input
    return np.concatenate([rgb_images, distortion_mask_resized], axis=-1)
