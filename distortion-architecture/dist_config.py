# example set
image_settings = {
    "clean_train_path": "./traffic_dataset/train/clean/",
    "distorted_train_path": "./traffic_dataset/train/distorted/",    
    "clean_test_path": "./traffic_dataset/test/clean/",
    "distorted_test_path": "./traffic_dataset/test/distorted/",
    "target_dim": 512
}

# training
train_settings = {
    "num_epochs": 15,
    "batch_size": 24,
}

optimizer_settings = {
    "lr": 1e-3,
}

dist_net_settings = {
    "distortion_grid_dim": 8,
    "distortion_score_thresh": 0.5
}

# convolutional layers -- should leave alone
conv1_settings = {  
    "in_channels": 3,
    "out_channels": 64,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1
}

conv2_settings = {
    "in_channels": 64,
    "out_channels": 128,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1
}

conv3_settings = {
    "in_channels": 128,
    "out_channels": 256,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1
}

conv4_settings = {
    "in_channels": 256,
    "out_channels": 512,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1
}

conv5_settings = {
    "in_channels": 512,
    "out_channels": 1024,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1
}

# max pooling
max_pool_settings = {
    "kernel_size": 3,
    "stride": 2
}

activation_settings = {
    "negative_slope_layer_1": 0.01,
    "negative_slope_layer_2": 0.01,
    "negative_slope_layer_3": 0.001,    
    "negative_slope_layer_4": 0.001,   
    "negative_slope_layer_5": 0.001    
}

# final model
distortion_detection_model = {
    "model_destination": "./distortion_detection_model.pth",
    "train_output_destination": "./training_distortion_maps",
    "test_output_destination": "./test_distortion_maps/",
    "binary_mask_destination": "./test_distortion_masks/"
}
