import numpy as np
from tensorflow import load_model
from unet_data_loader import load_data
from unet_data_generator import DataGenerator
from modified_unet import build_modified_unet
from unet_train import train_unet, evaluate_model

def main():
    radar_image_dir = 'radar_images'
    distortion_mask_dir = 'distortion_masks'

    # Load data
    train_images, train_masks = load_data(radar_image_dir, distortion_mask_dir)
    val_images, val_masks = load_data('val_radar_images', 'val_distortion_masks')  # Validation data

    # Load distortion model
    distortion_model = load_model('path/to/distortion_model.h5')

    # Define batch size and epochs
    batch_size = 8
    epochs = 10

    # Create U-Net model
    unet_model = build_modified_unet(input_shape=(256, 256, 4))
    unet_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # Create data generators
    train_generator = DataGenerator(train_images, train_masks, batch_size, distortion_model)
    val_generator = DataGenerator(val_images, val_masks, batch_size, distortion_model)

    # Train model
    train_unet(unet_model, train_generator, num_epochs=epochs)

    # Evaluate model on validation set
    evaluate_model(unet_model, val_generator)

    # Save the trained model
    unet_model.save('unet_model')
    print("Model training and evaluation complete. Model saved.")

if __name__ == "__main__":
    main()
