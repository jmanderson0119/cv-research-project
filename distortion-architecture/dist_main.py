import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dist_net import DistortionNet
from dist_loss import DistortionLoss
from dist_detection_dataset import DistortionDataset
from dist_utils import *
from dist_config import *
from dist_train import train_model

def dist_main():
    # sets the training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training device -- " + str(device))
    
    ## training
    # Load and preprocess clean and distorted images from the training directories
    print("loading clean training set...")
    clean_images = load_images_from_directory(image_settings["clean_train_path"], image_settings["target_dim"])
    print("clean training set loaded.\nloading distorted training set...")
    distorted_images = load_images_from_directory(image_settings["distorted_train_path"], image_settings["target_dim"])
    print("distorted training set loaded.")

    # ensure both sets of images have the same length
    assert len(clean_images) == len(distorted_images), "Mismatch in number of clean and distorted images."

    # initialize the dataset with clean and distorted images
    dataset = DistortionDataset(clean_images, distorted_images)

    # initialize the data loader
    train_loader = DataLoader(dataset, train_settings["batch_size"], shuffle=True)

    # load gt distortion maps
    ground_truth_maps_output_dir = os.path.join(distortion_detection_model["train_output_destination"], "ground_truth_maps")
    ground_truth_distortion_maps = load_ground_truth_distortion_maps(ground_truth_maps_output_dir)

    # initialize the model
    model = DistortionNet().to(device)

    # initialize the loss function
    criterion = DistortionLoss()

    # initialize the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_settings["lr"])

    # number of epochs
    num_epochs = train_settings["num_epochs"]

    # train the model
    print("training distortion network...")
    train_model(model, train_loader, num_epochs, criterion, optimizer, device, ground_truth_distortion_maps)

    # save the model
    torch.save(model.state_dict(), distortion_detection_model["model_destination"])
    print(f"Distortion Detection model is saved to " + os.path.join(os.getcwd(), distortion_detection_model["model_destination"]))

if __name__ == "__main__": dist_main()
