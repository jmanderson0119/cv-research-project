from torch.utils.data import Dataset

class DistortionDataset(Dataset):
    def __init__(self, clean_images, distorted_images):
        """
        inits the dataset
        
        :param clean_images: List of preprocessed clean image tensors
        :param distorted_images: List of preprocessed distorted image tensors
        """
        # stores the clean and distorted image tensors
        self.clean_images = clean_images
        self.distorted_images = distorted_images
    
    def __len__(self):
        """
        returns the length of the dataset.
        """
        return len(self.clean_images)
    
    def __getitem__(self, i):
        """
        retrieves a clean and a distorted image pair
        
        :param index: The index to retrieve the clean and distorted images.
        :return: tuple of (clean_image, distorted_image)
        """        
        return self.clean_images[i], self.distorted_images[i]
