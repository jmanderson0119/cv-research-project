from torch.utils.data import Dataset

class DistortionDataset(Dataset):
    def __init__(self, images):
        """
        Initializes the dataset
        
        :param images: List of preprocessed image tensors.
        """
        # stores the preprocessed image tensors
        self.images = images
    
    # defines the number of images in the dataset
    def __len__(self): return len(self.images)
    
    # retrieves and returns an image as the input and target
    def __getitem__(self, i): return self.images[i], self.images[i]
