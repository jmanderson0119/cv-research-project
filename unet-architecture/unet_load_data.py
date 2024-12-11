from torch.utils.data import DataLoader
from unet_data_generator import UNetDataset

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

    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader