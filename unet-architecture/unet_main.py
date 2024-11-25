import numpy as np
import tensorflow as tf
from tensorflow import load_model
from unet_data_loader import load_data 
from unet_data_generator import DataGenerator
from modified_unet import build_modified_unet
from process_input import process_input
from unet_train import train_unet

def main():
    radar_image_dir = 'radar_images'  
    distortion_mask_dir = 'distortion_masks'

    # Load radar images and distortion masks
    rgb_images, distortion_masks = load_data(radar_image_dir, distortion_mask_dir)

    # Load distortion model
    distortion_model = load_model('distortion-architecture\dist_net.py')  

    # Define training parameters
    batch_size = 8
    epochs = 10

    # Create the unet model
    unet_model = build_modified_unet(input_shape=(256, 256, 4))
    unet_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # Create the data generator
    train_generator = DataGenerator(rgb_images, distortion_masks, batch_size, distortion_model)

    # Train the unet model
    train_unet(unet_model, train_generator, num_epochs=epochs)

    # Save the model
    unet_model.save('unet_model')

    print("Model training complete")

if __name__ == "__main__":
    main()
