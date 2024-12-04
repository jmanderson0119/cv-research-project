import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=64):
        super(UNet, self).__init__()
        
        # initial convolutional blocks (encoder)
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, features),  # (64, 128, 256, 512, etc.)
            self.conv_block(features, features * 2),
            self.conv_block(features * 2, features * 4),
            self.conv_block(features * 4, features * 8)
        )
        
        self.middle = self.conv_block(features * 8, features * 16)
        
        # Decoder blocks
        self.decoder = nn.Sequential(
            self.upconv_block(features * 16, features * 8),
            self.upconv_block(features * 8, features * 4),
            self.upconv_block(features * 4, features * 2),
            self.upconv_block(features * 2, features)
        )
        
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Added batch normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Added batch normalization
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),  # Added batch normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Added batch normalization
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask):
        # Ensure mask and input image are the same shape
        assert x.shape[2:] == mask.shape[2:], "Input and mask must have the same spatial dimensions."

        # Expand mask to be a single channel
        mask = mask.unsqueeze(1)  # make the mask 1 channel 
    
        # Concatenate the image and mask to form 4 channels
        x = torch.cat((x, mask), dim=1)  # now x has 4 channels 

        # Encoder
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)
        
        # Middle block
        middle = self.middle(enc4)
        
        # Decoder
        dec1 = self.decoder[0](middle)
        dec2 = self.decoder[1](dec1)
        dec3 = self.decoder[2](dec2)
        dec4 = self.decoder[3](dec3)
        
        # Final convolution
        out = self.final_conv(dec4)

        # Combine the original image and the mask
        restored_image = self.restore_image(x[:, :3, :, :], out, mask)  # Use only the original image for restoration

        return restored_image

    def restore_image(self, original_image, restored_image, mask):
        """
        Restores the image based on the distortion mask.
        Uses the mask to replace distorted regions in the original image with the restored ones.
        
        :param original_image: The original image.
        :param restored_image: The output from the U-Net model (restored area). 
        :param mask: The distortion mask (binary mask, 1 for distortion, 0 for no distortion).
        :return: The final restored image.
        """
        # The mask is a binary mask, where 1 means the area is distorted
        mask = mask.expand_as(original_image)  # Expand the mask to match the input image dimensions

        # Replace the distorted regions with the output from the U-Net model
        restored_image = mask * restored_image + (1 - mask) * original_image

        return restored_image