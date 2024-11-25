import torch
import torch.nn as nn
from dist_config import *

class DistortionNet(nn.Module):
    def __init__(self):
        super(DistortionNet, self).__init__()

        # extract settings from config
        self.conv1_in_channels = conv1_settings["in_channels"]
        self.conv2_in_channels = conv2_settings["in_channels"]
        self.conv3_in_channels = conv3_settings["in_channels"]

        self.conv1_out_channels = conv1_settings["out_channels"]
        self.conv2_out_channels = conv2_settings["out_channels"]
        self.conv3_out_channels = conv3_settings["out_channels"]

        self.conv1_kernel = conv1_settings["kernel_size"]
        self.conv2_kernel = conv2_settings["kernel_size"]
        self.conv3_kernel = conv3_settings["kernel_size"]
        self.max_pool_kernel = max_pool_settings["kernel_size"]

        self.conv1_stride = conv1_settings["stride"]
        self.conv2_stride = conv2_settings["stride"]
        self.conv3_stride = conv3_settings["stride"]
        self.max_pool_stride = max_pool_settings["stride"]

        self.conv1_padding = conv1_settings["padding"]
        self.conv2_padding = conv2_settings["padding"]
        self.conv3_padding = conv3_settings["padding"]
        
        # conv layers
        self.conv1 = nn.Conv2d(
            in_channels=self.conv1_in_channels,
            out_channels=self.conv1_out_channels,
            kernel_size=self.conv1_kernel,
            stride=self.conv1_stride,
            padding=self.conv1_padding
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.conv2_in_channels,
            out_channels=self.conv2_out_channels,
            kernel_size=self.conv2_kernel,
            stride=self.conv2_stride,
            padding=self.conv2_padding
        )

        self.conv3 = nn.Conv2d(
            in_channels=self.conv3_in_channels,
            out_channels=self.conv3_out_channels,
            kernel_size=self.conv3_kernel,
            stride=self.conv3_stride,
            padding=self.conv3_padding
        )

        # pooling layers
        self.max_pool = nn.MaxPool2d(kernel_size=self.max_pool_kernel, stride=self.max_pool_stride)
        self.global_avg_pool = nn.AdaptiveAveragePool2d(1)

        # fully connected layer
        self.fc1 = nn.Linear(self.conv3_out_channels, image_settings["target_dim"] ** 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input shape check
        if x.shape[1] != self.conv1_in_channels: raise ValueError(f"expected input channels: {self.conv1_in_channels}, but got: {x.shape[1]}")

        # pass through conv1 + max pooling
        x = self.max_pool(torch.relu(self.conv1(x)))

        # validate shape after first conv layer
        expected_conv1_shape = (x.size(0), self.conv1_out_channels,
                               (x.size(2) - self.conv1_kernel + 2 * self.conv1_padding) // self.conv1_stride + 1,
                               (x.size(3) - self.conv1_kernel + 2 * self.conv1_padding) // self.conv1_stride + 1)
        if x.shape != expected_conv1_shape: raise ValueError(f"Shape mismatch after conv1. Expected: {expected_conv1_shape}, received: {x.shape}")

        # pass through conv2 + max pooling
        x = self.max_pool(torch.relu(self.conv2(x)))

        # validate shape after second conv layer
        expected_conv2_shape = (x.size(0), self.conv2_out_channels,
                               (x.size(2) - self.conv2_kernel + 2 * self.conv2_padding) // self.conv2_stride + 1,
                               (x.size(3) - self.conv2_kernel + 2 * self.conv2_padding) // self.conv2_stride + 1)
        if x.shape != expected_conv2_shape: raise ValueError(f"Shape mismatch after conv2. Expected: {expected_conv2_shape}, received: {x.shape}")

        # pass through conv3 + global average pooling
        x = self.global_avg_pool(torch.relu(self.conv3(x)))

        # validate shape after third conv layer
        expected_conv3_shape = (x.size(0), self.conv3_out_channels, 1, 1)
        if x.shape != expected_conv3_shape: raise ValueError(f"Shape mismatch after conv3. Expected: {expected_conv3_shape}, received: {x.shape}")

        # flatten and pass to fully connected layer
        x = x.view(x.size(0), -1)

        # validate shape after flattening
        expected_flatten_shape = (x.size(0), self.conv3_out_channels)
        if x.shape != expected_flatten_shape: raise ValueError(f"Shape mismatch after flattening. Expected: {expected_flatten_shape}, received: {x.shape}")

        # fully connected layer + sigmoid activation
        x = self.fc1(x)
        x = self.sigmoid(x)

        # validate final output shape before reshaping
        expected_fc_shape = (x.size(0), image_settings["target_dim"] ** 2)
        if x.shape != expected_fc_shape: raise ValueError(f"Shape mismatch after fully connected layer. Expected: {expected_fc_shape}, received: {x.shape}")

        # produce final scoring and reshape to anomaly map
        x = x.view(-1, 1, image_settings["target_dim"], image_settings["target_dim"])

        # validate final output shape
        expected_final_shape = (x.size(0), 1, image_settings["target_dim"], image_settings["target_dim"])
        if x.shape != expected_final_shape: raise ValueError(f"Shape mismatch after final output. Expected: {expected_final_shape}, received: {x.shape}")

        return x
