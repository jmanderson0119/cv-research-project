import torch.nn as nn
import torch.nn.functional as F

class DistortionLoss(nn.Module):
    def __init__(self):
        super(DistortionLoss, self).__init__()
    
    def forward(self, predicted_maps, ground_truth_maps):
        """
        calculate the loss based on the difference between clean and distorted images + the predicted distortion map

        :param predicted_maps: predicted distortion map tensor batch
        :param ground_truth_maps: precomputed distortion maps for the batch
        
        :returns: average loss across the batch
        """
        ground_truth_maps = ground_truth_maps.to(predicted_maps.device)

        loss = F.mse_loss(predicted_maps, ground_truth_maps.unsqueeze(1))
        
        return loss
    