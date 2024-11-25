import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dist_net import DistortionNet
from dist_loss import AnomalyDetectionLoss
from dist_detection_dataset import DistortionDataset
from dist_train import train
from dist_utils import preprocess_images
from dist_config import *

def dist_main():    
    # preprocesses images, normalizing them to the target height and width and preserving their aspect ratios
    preprocessed_images = preprocess_images(
        input_directory=image_settings["train_path"],
        target_dim=image_settings["target_dim"]
    )

    # sets the training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # init distortion net
    model = DistortionNet().to(device)
    
    # init data loader
    train_loader = DataLoader(DistortionDataset(preprocessed_images), train_settings["batch_size"], shuffle=True)

    # init loss function
    criterion = AnomalyDetectionLoss(dist_net_settings["distortion_grid_dim"])

    # init the lbfgs optimizer
    optimizer = optim.LBFGS(model.parameters(), optimizer_settings["lr"], optimizer_settings["max_iter"], optimizer_settings["history_size"])

    # set number of epochs for training loop
    num_epochs = train_settings["num_epochs"]

    # training arc
    train(model, train_loader, criterion, optimizer, device, num_epochs)

    # save model
    torch.save(model.state_dict(), model["destination"])
    print("model is saved to" + model["destination"])

if __name__ == "__main__": dist_main()
