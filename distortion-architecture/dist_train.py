import torch
from dist_config import train_settings


def train_model(model, train_loader, num_epochs, criterion, optimizer, device, ground_truth_distortion_maps):
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, threshold=0.01, 
                                                           threshold_mode="abs", cooldown=2, min_lr=1e-7)

    total_batches = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1} - Current Learning Rate: {current_lr}")
        for i, (clean_image, distorted_image) in enumerate(train_loader):
            clean_image, distorted_image = clean_image.to(device), distorted_image.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # predicted map
            predicted_maps = model(distorted_image)

            # load ground truth distortion maps for the current batch
            start_idx = i * train_settings["batch_size"]
            end_idx = (i + 1) * train_settings["batch_size"]
            ground_truth_maps = ground_truth_distortion_maps[start_idx:end_idx]
            
            # loss
            loss = criterion(predicted_maps, ground_truth_maps)

            # backpropagation
            loss.backward()
      
            # optimization step
            optimizer.step()

            total_loss += loss.item()
            
            # progress for each batch within the epoch
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i+1}/{total_batches}], Loss: {loss.item():.6f}")

        # average loss for the epoch
        avg_loss = total_loss / total_batches
        
        # step the scheduler using the average loss
        scheduler.step(avg_loss)

        # after each epoch, print the average loss
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
    print("Training done")