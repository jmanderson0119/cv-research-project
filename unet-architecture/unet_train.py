import numpy as np
import tensorflow as tf
from tensorflow import Adam
from tensorflow import MeanSquaredError

def train_unet(model, train_generator, num_epochs=10):
    """
    Trains the U-Net model using a custom training loop.
    """
    # Adam optimizer + MSE loss
    optimizer = Adam()
    loss_fn = MeanSquaredError()

    # training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # iterate over batches -- using the data generator
        for batch_input, batch_labels in train_generator:
            # Concatenate the image and mask into a 4-channel input for the model
            batch_input = np.concatenate([batch_input, batch_labels], axis=-1)  # Concatenate along the channel axis
            
            with tf.GradientTape() as tape:
                # forward pass
                predictions = model(batch_input, training=True)
                
                # calculate loss
                loss = loss_fn(batch_labels, predictions)
            
            # calculate gradients + update model weights
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # total loss for the epoch
            epoch_loss += loss.numpy()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_generator)}")

    print("Training completed.")
