import torch
import matplotlib.pyplot as plt
from modified_unet import UNet
from unet_loss import UNetLoss
from unet_data_generator import UNetDataset
from unet_load_data import load_data
import numpy as np

def train_unet(model, distorted_image_dir, mask_dir, target_dim, batch_size, num_epochs, learning_rate, device):
    print("Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model, loss function, and optimizer
    model = UNet(in_channels=4, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = UNetLoss(max_val=1.0).to(device)

    # Load data using your DataLoader
    dataloader = load_data(distorted_image_dir, mask_dir, target_dim, batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Loop over the batches of data
        for i, (inputs, targets) in enumerate(dataloader):

            # Ensure that inputs and targets are PyTorch tensors with correct dtype
            inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True).to(device)
            targets = torch.tensor(targets, dtype=torch.float32, requires_grad=False).to(device)

            # Forward pass
            outputs = model(inputs[:, :3, :, :], inputs[:, 3:, :, :])

            # Compute the loss
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track running loss
            running_loss += loss.item()

            # Visualize every few iterations
            if i % 100 == 99:
                print(f"Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(dataloader)}], Loss: {running_loss / 100:.4f}")
                visualize_results(inputs, targets, outputs, criterion)

                # Reset running loss
                running_loss = 0.0

    print("Training complete")

def visualize_results(inputs, targets, outputs, criterion):
    # Convert tensors to numpy arrays for visualization
    inputs = inputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()

    # Plot input, prediction, and ground truth side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.transpose(inputs[0, :3, :, :], (1, 2, 0)))  # Input image (first 3 channels)
    axes[0].set_title('Input Image')
    axes[1].imshow(np.transpose(outputs[0], (1, 2, 0)))  # Prediction
    axes[1].set_title('Prediction')
    axes[2].imshow(np.transpose(targets[0], (1, 2, 0)))  # Ground Truth
    axes[2].set_title('Ground Truth')
    plt.show()

    # Compute SSIM and MSE
    ssim_val = np.mean([criterion.ssim(torch.tensor(output).unsqueeze(0), torch.tensor(target)).cpu().numpy() for output, target in zip(outputs, targets)])
    mse_val = np.mean([criterion.mse_loss(torch.tensor(output).unsqueeze(0), torch.tensor(target)).cpu().numpy() for output, target in zip(outputs, targets)])
    
    print(f"SSIM: {ssim_val:.4f}, MSE: {mse_val:.4f}")

    # Optionally visualize the distribution of SSIM and MSE
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(ssim_val, bins=20)
    axes[0].set_title('SSIM Distribution')
    axes[1].hist(mse_val, bins=20)
    axes[1].set_title('MSE Distribution')
    plt.show()
    