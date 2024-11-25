import torch
from torch.utils.data import DataLoader
from dist_net import DistortionNet
from dist_loss import DistortionLoss
from dist_detection_dataset import DistortionDataset
from dist_utils import load_images_from_directory
from dist_train import train
from dist_config import image_settings, train_settings, optimizer_settings, dist_net_settings

def dist_main():
    
    # Load and preprocess clean and distorted images from the directories
    clean_images = load_images_from_directory(image_settings["clean_train_path"], image_settings["target_dim"])
    distorted_images = load_images_from_directory(image_settings["distorted_train_path"], image_settings["target_dim"])

    # sets the training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ensure both sets of images have the same length
    assert len(clean_images) == len(distorted_images), "Mismatch in number of clean and distorted images."

    # initialize the dataset with clean and distorted images
    dataset = DistortionDataset(clean_images, distorted_images)

    # initialize the data loader
    train_loader = DataLoader(dataset, train_settings["batch_size"], shuffle=False)

    # initialize the model
    model = DistortionNet().to(device)

    # initialize the loss function
    criterion = DistortionLoss(dist_net_settings["distortion_grid_dim"])

    # initialize the LBFGS optimizer
    optimizer = torch.optim.LBFGS(model.parameters(), lr=optimizer_settings["lr"], max_iter=optimizer_settings["max_iter"])

    # number of epochs
    num_epochs = train_settings["num_epochs"]

    # train the model
    train(model, train_loader, criterion, optimizer, device, num_epochs)

    # save the model
    torch.save(model.state_dict(), model["destination"])
    print(f"Model is saved to {model['destination']}")

if __name__ == "__main__": dist_main()
