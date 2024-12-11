import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2

class UNetDataset(Dataset):
    """
    Custom datasat for U-Net training
    Generates batches of data from distorted traffic images and distortion masks
    """
    def __init__(self, distorted_image_dir, mask_dir, target_dim, transform=None):
        self.distorted_image_dir = distorted_image_dir
        self.mask_dir = mask_dir
        self.target_dim = target_dim
        self.transform = transform
        self.distorted_images_paths = sorted([os.path.join(distorted_image_dir, fname) for fname in os.listdir(distorted_image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

        # Ensure matching number of images and masks
        assert len (self.distorted_images_paths) == len(self.mask_paths), "Mismatch between image and mask counts"

    def __len__(self):
        return len(self.distorted_images_paths)

    def __getitem__(self, index):
        # Load image and mask 
        image = Image.open(self.distorted_images_paths[index]).convert("RGB")
        mask = np.load(self.mask_paths[index])

        # Resize and convert to tensors
        image = np.array(image.resize(self.target_dim, Image.ANTIALIAS))
        mask = cv2.resize(mask, self.target_dim)

        # Convert to tensors
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension for mask

        # Apply transformations 
        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
      
        return image_tensor, mask_tensor
    
def load_data(distorted_image_dir, mask_dir, target_dim, batch_size, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for unet training/inference
        
    param distorted_image_dir: path to the distorted images
    param distortion_mask_dir: path the distortion masks 
    param target_dim: target dimensions for resizing
    param shuffle: whether to shuffle the data (bool)
    param num_workers: number of subprocesses to use for data loading 

    returns: DataLoader: PyTorch DataLoader
    """

    dataset = UNetDataset(distorted_image_dir, mask_dir, target_dim)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)