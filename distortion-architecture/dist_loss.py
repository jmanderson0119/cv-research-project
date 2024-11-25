import torch.nn as nn
import torch.nn.functional as F
from dist_utils import compute_distortion_map

class DistortionLoss(nn.Module):
    def __init__(self, distortion_grid_dim):
        super(DistortionLoss, self).__init__()
        self.distortion_grid_dim = distortion_grid_dim
    
    def forward(self, clean_image, distorted_image, predicted_map):
        """
        calculate the loss based on the difference between clean and distorted images + the predicted distortion map

        :param clean_image: Tensor of shape (batch_size, 3, 256, 256)
        :param distorted_image: Tensor of shape (batch_size, 3, 256, 256)
        :param  predicted_map: Tensor of shape (batch_size, 1, 32, 32), predicted distortion map
        
        :returns: total loss
        """
        # computes the ground-truth distortion map reference
        ground_truth_map = compute_distortion_map(clean_image, distorted_image)
        
        # computes mse loss between the predicted and ground truth map
        loss = F.mse_loss(predicted_map, ground_truth_map)
        
        return loss
