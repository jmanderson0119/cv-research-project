import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from unet_data_loader import load_data
from unet_data_generator import DataGenerator
from modified_unet import UNet
from unet_train import train_unet, evaluate_model, visualize_results, visualize_ssim_psnr

def load_distortion_model(model_path):
    """
    Loads the distortion model, which is assumed to be a saved numpy array or a PyTorch model.
    """
    # Placeholder: Replace this with your actual distortion model loading logic
    return np.load(model_path)  # Adjust if your distortion model is a trained PyTorch model

def main():
    # Define directories
    image_dir = 'IranianTrafficSignDetection/train'
    mask_dir = 'distortion_masks'
    val_image_dir = 'IranianTrafficSignDetection/val'
    val_mask_dir = 'val_distortion_masks'

    # Load training and validation data
    train_images, train_masks = load_data(image_dir, mask_dir)
    val_images, val_masks = load_data(val_image_dir, val_mask_dir)

    # Load the distortion model
    distortion_model_path = 'path/to/distortion_model.npy'
    distortion_model = load_distortion_model(distortion_model_path)

    # Define parameters
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-4

    # Create U-Net model
    unet_model = UNet(in_channels=4, out_channels=3)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    # Create data generators
    train_dataset = DataGenerator(train_images, train_masks, distortion_model)
    val_dataset = DataGenerator(val_images, val_masks, distortion_model)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    train_unet(unet_model, train_loader, val_loader, num_epochs=num_epochs, optimizer=optimizer, loss_fn=loss_fn)

    # Evaluate the model
    avg_ssim, avg_psnr = evaluate_model(unet_model, val_loader)
    print(f"Average Validation SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")

    # Visualize results on a sample batch
    sample_batch = next(iter(val_loader))  # Get a single batch from the validation loader
    sample_inputs, sample_labels = sample_batch
    sample_predictions = unet_model(sample_inputs).detach().numpy()  # Perform inference

    visualize_results(sample_inputs.numpy(), sample_predictions, sample_labels.numpy(), num_samples=5)

    # Visualize SSIM and PSNR distributions
    ssim_scores = []
    psnr_scores = []
    for inputs, labels in val_loader:
        predictions = unet_model(inputs).detach()
        ssim_scores.append(calculate_ssim(labels, predictions).numpy())
        psnr_scores.append(calculate_psnr(labels, predictions).numpy())

    visualize_ssim_psnr(ssim_scores, psnr_scores)

    # Save the trained model
    model_save_path = 'unet_model.pth'
    torch.save(unet_model.state_dict(), model_save_path)
    print(f"Model training complete. Model saved to {model_save_path}.")

if __name__ == "__main__":
    main()
