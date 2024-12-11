import torch
from unet_train import train_unet
from modified_unet import UNet
import os
import kornia

def main():
    # Configuration settings
    distorted_image_dir = "./traffic_dataset/train/distorted/"  
    mask_dir = "./traffic_dataset/test_distortion_masks/"  
    target_dim = (256, 256)  
    batch_size = 16  
    num_epochs = 25  
    learning_rate = 0.001  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "./traffic_dataset/unet"  

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Print configuration
    print("Configuration:")
    print(f"Distorted Image Directory: {distorted_image_dir}")
    print(f"Mask Directory: {mask_dir}")
    print(f"Target Dimensions: {target_dim}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"Output Directory: {output_dir}")

    # Initialize the model
    model = UNet(in_channels=4, out_channels=3)

    # Train the model
    train_unet(model, 
               distorted_image_dir, 
               mask_dir, 
               target_dim, 
               batch_size, 
               num_epochs, 
               learning_rate, 
               device)

    # Save the trained model
    model_save_path = os.path.join(output_dir, "unet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

if __name__ == "__main__":
    main()
