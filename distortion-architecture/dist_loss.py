import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_config import dist_net_settings

class AnomalyDetectionLoss(nn.Module):
    def __init__(self, grid_dim):
        """
        inits the loss function
        
        :param grid_dim: the size of the grid regions that will be used to compute loss;
        the image is divided into smaller grids, and the loss is calculated for each grid square
        """
        super(AnomalyDetectionLoss, self).__init__()
        self.grid_dim = grid_dim

    def forward(self, output, input_image):
        """
        computes the loss by comparing the output shape and the input shape on a grid basis
        
        :param output: The output tensor from the model.
        :param input_image: The input tensor.
        
        :return: total loss; sum of MSE losses for each grid region
        """
        loss = 0.0
        grayscale_input = torch.mean(input_image, dim=1, keepdim=True)
        
        thresh = dist_net_settings["distortion_score_thresh"]
        binary_output = (output > thresh).float()

        # iterates over input shape in steps of `grid_dim` along the height and width
        for i in range(0, grayscale_input.size(2), self.grid_dim):
            for j in range(0, grayscale_input.size(3), self.grid_dim):
                # specifies appropriate region of input and output shape + accumulates loss
                region_target = grayscale_input[:, :, i:i+self.grid_dim, j:j+self.grid_dim]
                region_output = binary_output[:, :, i:i+self.grid_dim, j:i+self.grid_dim]
                loss += F.mse_loss(region_output, region_target)

        return loss
