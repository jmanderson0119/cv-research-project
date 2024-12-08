import torch
from dist_config import train_settings


def train_model(model, train_loader, num_epochs, criterion, optimizer, device, ground_truth_distortion_maps):
        # Add this function at the top of your training script
    def debug_gradients(model, optimizer):
        print("\n--- Gradient Debugging ---")
        
        # Check requires_grad and parameter details
        print("\nParameter Requires Grad Status:")
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}, shape = {param.shape}")
        
        # Capture initial parameter values
        print("\nBefore Optimization:")
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.data.clone()
            print(f"{name}: mean = {param.data.mean().item():.6f}, norm = {param.data.norm().item():.6f}")
        
        # Perform optimization step
        optimizer.step()
        
        # Check parameter changes
        print("\nAfter Optimization:")
        for name, param in model.named_parameters():
            print(f"{name}: mean = {param.data.mean().item():.6f}, norm = {param.data.norm().item():.6f}")
            
            # Check if parameter actually changed
            if torch.equal(initial_params[name], param.data):
                print(f"WARNING: {name} DID NOT CHANGE")
            
            # Check gradient information if available
            if param.grad is not None:
                print(f"{name} gradient - mean: {param.grad.mean().item():.6f}, norm: {param.grad.norm().item():.6f}")
            else:
                print(f"WARNING: {name} has NO GRADIENT")


    # Create ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, threshold=0.01, 
                                                           threshold_mode="abs", cooldown=2, min_lr=1e-7)

    total_batches = len(train_loader)  # Total number of batches in the current epoch

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1} - Current Learning Rate: {current_lr}")
        for i, (clean_image, distorted_image) in enumerate(train_loader):
            clean_image, distorted_image = clean_image.to(device), distorted_image.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Get predicted map
            predicted_maps = model(distorted_image)

            # Compute the ground truth distortion maps for the current batch using precomputed maps
            start_idx = i * train_settings["batch_size"]
            end_idx = (i + 1) * train_settings["batch_size"]
            ground_truth_maps = ground_truth_distortion_maps[start_idx:end_idx]
            
            # Compute loss
            loss = criterion(predicted_maps, ground_truth_maps)

            # Backpropagation
            loss.backward()
      
            # Perform optimization step
            optimizer.step()

            total_loss += loss.item()
            
            # Print progress for each batch within the epoch
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i+1}/{total_batches}], Loss: {loss.item():.6f}")

        # Calculate average loss for the epoch
        avg_loss = total_loss / total_batches
        
        # step the scheduler using the average loss
        scheduler.step(avg_loss)

        # After each epoch, print the average loss
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
    print("Training done")