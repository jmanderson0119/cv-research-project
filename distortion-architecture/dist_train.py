import torch
from dist_loss import DistortionLoss

def train_model(model, train_loader, num_epochs, distortion_grid_dim, device):
    """
    runs the training loop

    :param model: the distortion net model
    :param train_loader: data loader for the image pairs
    :param num_epochs: training epochs
    :param distortion_grid_dim: grid dimension for distortion map
    :param device: device used for training
    """
    # init loss
    distortion_loss = DistortionLoss(distortion_grid_dim).to(device)
    
    # define LBFGS optimizer
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-3, max_iter=20)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for clean_image, distorted_image in train_loader:
            clean_image, distorted_image = clean_image.to(device), distorted_image.to(device)
            
            # closure function
            def closure():
                optimizer.zero_grad()

                # get predicted map
                predicted_map = model(distorted_image)

                # computes loss
                loss = distortion_loss(clean_image, distorted_image, predicted_map)

                # backprop
                loss.backward()

                return loss

            # perform one step of optimization using lbfgs
            loss = optimizer.step(closure)

            total_loss += loss.item()

        print(f"epoch {epoch+1}/{num_epochs} - loss: {total_loss/len(train_loader):.4f}")
        
    print("training done")
