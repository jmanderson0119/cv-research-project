import numpy as np
import tensorflow as tf

def process_input(rgb_images, distortion_model):
    """
    Processes 4-channel input (RGB + Mask) for U-Net
    """
    # normalize the RGB images
    rgb_images = rgb_images / 255.0 if rgb_images.max() > 1 else rgb_images

    # Use the distortion detection model to generate masks
    distortion_mask = distortion_model.predict(rgb_images)

    # make sure the distortion mask has the shape
    if distortion_mask.shape[-1] != 1:
        distortion_mask = np.expand_dims(distortion_mask, axis=-1)

    # Resize the distortion mask to match the input dimensions
    batch_size, height, width, _ = rgb_images.shape
    distortion_mask_resized = tf.image.resize(distortion_mask, size=(height, width))

    # make sure the resized mask has the correct shape
    distortion_mask_resized = tf.cast(distortion_mask_resized, dtype=tf.float32)

    # Concatenate RGB images and the resized mask for input to unet
    return np.concatenate([rgb_images, distortion_mask_resized], axis=-1)
