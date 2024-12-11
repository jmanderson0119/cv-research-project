import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

class UNetLoss(nn.Module):
    def __init__(self, max_val=1.0):
        """
        UNet loss function with learnable weights for MSE and SSIM loss components.
        
        Args:
            max_val (float): The maximum value for the image (typically 1.0 for normalized images).
        """
        super(UNetLoss, self).__init__()
        
        # Learnable weights for the loss components (MSE and SSIM)
        self.mse_weight = nn.Parameter(torch.tensor(1.0))  # Default weight for MSE loss
        self.ssim_weight = nn.Parameter(torch.tensor(1.0))  # Default weight for SSIM loss
        
        # Maximum value for the image, often 1.0 for normalized images
        self.max_val = max_val

    def forward(self, output, target):
        """
        Forward pass to compute the total loss.
        
        Args:
            output (torch.Tensor): The output from the UNet model (predicted image).
            target (torch.Tensor): The ground truth (target) image.
        
        Returns:
            torch.Tensor: The total computed loss.
        """
       # Ensure tensors are PyTorch tensors
        assert isinstance(output, torch.Tensor), f"Output must be a torch.Tensor, got {type(output)}"
        assert isinstance(target, torch.Tensor), f"Target must be a torch.Tensor, got {type(target)}"

        # Ensure tensors are on the same device
        target = target.to(output.device)

        assert output.shape == target.shape, f"Shape mismatch: {output.shape} vs {target.shape}"
    
        self.mse_weight = self.mse_weight.to(output.device)
        self.ssim_weight = self.ssim_weight.to(output.device)

        # Compute MSE and SSIM losses
        mse_loss = F.mse_loss(output, target)
        ssim_loss = 1 - self.ssim(output, target)

        # Combine losses
        total_loss = self.mse_weight * mse_loss + self.ssim_weight * ssim_loss

        # Ensure the total_loss is a scalar tensor
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss)
        assert total_loss.numel() == 1, "Total loss must be a scalar tensor"

        return total_loss
    
    def ssim(self, output, target, window_size=11, size_average=True):
        """
        Computes the Structural Similarity Index (SSIM) between two images.
        
        Args:
            output (torch.Tensor): The predicted image
            target (torch.Tensor): The ground truth image
            window_size (int): The size of the Gaussian window
            channel (int): The number of channels in the image (e.g., 3 for RGB).
            size_average (bool): Whether to average the SSIM over all pixels.
        
        Returns:
            torch.Tensor: The computed SSIM value.
        """
        # Use Kornia's SSIM implementation for stability and efficiency
        # Ensure inputs are PyTorch tensors
        assert isinstance(output, torch.Tensor), f"Output must be a torch.Tensor, got {type(output)}"
        assert isinstance(target, torch.Tensor), f"Target must be a torch.Tensor, got {type(target)}"

        # Ensure tensors have 4 dimensions
        if output.ndim == 3:
            output = output.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)

        assert output.ndim == 4, f"Output must have 4 dimensions, got {output.ndim}"
        assert target.ndim == 4, f"Target must have 4 dimensions, got {target.ndim}"

        # Ensure tensors are on the same device
        target = target.to(output.device)

        ssim_value = kornia.losses.ssim(output, target, window_size=window_size, max_val=self.max_val)
        
        print(f"Type of ssim_value: {type(ssim_value)}")
        print(f"Shape of ssim_value (if tensor): {getattr(ssim_value, 'shape', None)}")

        if size_average:
            if isinstance(ssim_value, torch.Tensor):
                ssim_value = ssim_value.mean()
            else:
                ssim_value = float(ssim_value)  # Ensure scalar for non-tensor values

        return ssim_value
