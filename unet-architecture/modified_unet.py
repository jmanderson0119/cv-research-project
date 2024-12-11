import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(UNet, self).__init__()

        # Contracting path (Encoder)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Expanding path (Decoder)
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)

        # Final output layer (3 channels for RGB)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

        # Batch Normalization (BN)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)

        # Attention Mechanism (Optional, can be used for finer control over attention)
        self.attn1 = AttentionGate(64)
        self.attn2 = AttentionGate(128)
        self.attn3 = AttentionGate(256)
        self.attn4 = AttentionGate(512)

        # Shape Validation (Ensure the output matches the expected shape)
        self.target_shape = None

    def conv_block(self, in_channels, out_channels):
        """
        Convolutional block with Batch Normalization and ReLU activation.
        """
        bn_layer = nn.BatchNorm2d(out_channels)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            bn_layer,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            bn_layer,
        )


    def upconv_block(self, in_channels, out_channels):
        """
        Transposed Convolution (UpSampling) block
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            self.bn1 if out_channels == 64 else self.bn2,
        )

    def forward(self, x, mask):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder path with attention
        dec4 = self.decoder4(bottleneck)
        dec4 = self.attn4(dec4, enc4)  # Apply attention here
        dec3 = self.decoder3(dec4)
        dec3 = self.attn3(dec3, enc3)  # Apply attention here
        dec2 = self.decoder2(dec3)
        dec2 = self.attn2(dec2, enc2)  # Apply attention here
        dec1 = self.decoder1(dec2)
        dec1 = self.attn1(dec1, enc1)  # Apply attention here

        # Output layer
        out = self.output(dec1)
        out = self.check_shape(out)
        
        return out

    def check_shape(self, x):
        """
        Optionally, use this function to validate if the input size is correct.
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected a PyTorch tensor, but got {type(x)}")

        if x.dim() < 3:  # Ensure x has at least 3 dimensions (batch_size, channels, height, width)
            raise ValueError(f"Tensor has too few dimensions. Expected 4D tensor, got {x.dim()}D")

        if self.target_shape is None:
            self.target_shape = x.shape[2:]  # Get the expected target shape (height, width)
        
        # Validate that the shape matches the target shape
        if x.shape[2:] != self.target_shape:
            raise ValueError(f"Shape mismatch! Expected {self.target_shape}, but got {x.shape[2:]}")

        return x


# Attention Gate Module (used for attention mechanism in the U-Net)
class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()
        self.in_channels = in_channels

        self.attn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.attn_sigmoid = nn.Sigmoid()

    def forward(self, x, skip):
        """
        Apply attention mechanism
        """
        attn_map = self.attn_conv(skip)  # Apply convolution to skip connections
        attn_map = self.attn_sigmoid(attn_map)  # Sigmoid activation for attention map
        return x * attn_map  # Element-wise multiplication for attention