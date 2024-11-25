from dist_utils import lbfgs_closure

# trains the distortion net
def train(model, train_loader, criterion, optimizer, device, num_epochs):
    # set to training mode
    model.train()

    # train over epochs
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # train over batches
        for data in enumerate(train_loader):
            # move batch to the device
            data = data.to(device)

            # lbfgs_closure computes the loss and gradient (see dist_utils.py)
            loss = optimizer.step(lambda: lbfgs_closure(optimizer, model, data, data, criterion))

            # accumulates loss over the epoch
            epoch_loss += loss.item()


        # prints average loss over the epoch
        print(f"epoch {epoch+1}/{num_epochs} - loss: {epoch_loss/len(train_loader):.4f}")

    print("training done")
