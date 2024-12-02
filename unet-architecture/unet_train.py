import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def calculate_ssim(y_true, y_pred):
    """
    Computes the Structural Similarity Index (SSIM).
    """
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0) 
    return tf.reduce_mean(ssim)


def calculate_psnr(y_true, y_pred):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR).
    """
    psnr = tf.image.psnr(y_true, y_pred, max_val=1.0) 
    return tf.reduce_mean(psnr)


def train_unet(model, train_generator, num_epochs=10):
    """
    Trains the U-Net model with SSIM and PSNR evaluation.
    """
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_ssim = 0.0
        epoch_psnr = 0.0
        num_batches = 0

        for batch_input, batch_labels in train_generator:
            with tf.GradientTape() as tape:
                predictions = model(batch_input, training=True)
                loss = loss_fn(batch_labels, predictions)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Accumulate metrics
            epoch_loss += loss.numpy()
            epoch_ssim += calculate_ssim(batch_labels, predictions).numpy()
            epoch_psnr += calculate_psnr(batch_labels, predictions).numpy()
            num_batches += 1

        # Print metrics for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / num_batches:.4f}, "
              f"SSIM: {epoch_ssim / num_batches:.4f}, PSNR: {epoch_psnr / num_batches:.4f}")

    print("Training completed.")


def evaluate_model(model, validation_generator):
    """
    Evaluates the U-Net model on validation data using SSIM and PSNR.
    """
    total_ssim = 0.0
    total_psnr = 0.0
    num_batches = 0

    for batch_input, batch_labels in validation_generator:
        predictions = model.predict(batch_input)
        total_ssim += calculate_ssim(batch_labels, predictions).numpy()
        total_psnr += calculate_psnr(batch_labels, predictions).numpy()
        num_batches += 1

    avg_ssim = total_ssim / num_batches
    avg_psnr = total_psnr / num_batches

    print(f"Validation Results - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")
    return avg_ssim, avg_psnr

def visualize_results(inputs, predictions, labels, num_samples=5):
    """
    Visualizes input, prediction, and ground truth side-by-side.
    """
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(inputs[i])
        plt.title("Input")
        plt.axis("off")

        plt.subplot(3, num_samples, num_samples + i + 1)
        plt.imshow(predictions[i])
        plt.title("Prediction")
        plt.axis("off")

        plt.subplot(3, num_samples, 2 * num_samples + i + 1)
        plt.imshow(labels[i])
        plt.title("Ground Truth")
        plt.axis("off")

    plt.show()
