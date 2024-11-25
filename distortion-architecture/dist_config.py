# example set
image_settings = {
    "clean_train_path": "./traffic_dataset/train/clean/",
    "distorted_train_path": "./traffic_dataset/train/distorted/",
    "test_path": "./raidar/test/",
    "target_dim": 256,
}

# training
train_settings = {
    "num_epochs": 5,
    "batch_size": 16
}

optimizer_settings = {
    "lr": 1e-3,
    "max_iter": 20,
    "history_size": 10
}

dist_net_settings = {
    "distortion_grid_dim": 8,
    "distortion_score_thresh": 0.6
}

# convolutional layers
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

# max pooling
max_pool_settings = {
    "kernel_size": 3,
    "stride": 2
}

# final model
model = {
    "destination": "./distortion_detection_model.pth" 
}
