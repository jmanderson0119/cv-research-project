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

def train_unet(model, train_generator, val_generator, num_epochs=10, patience=5):
    """
    Trains the U-Net model with SSIM and PSNR evaluation and early stopping.
    """
    optimizer = tf.Adam()
    loss_fn = tf.MeanSquaredError()

    # Metrics
    ssim_metric = tf.Mean(name='ssim')
    psnr_metric = tf.Mean(name='psnr')

    # Callbacks
    early_stopping = tf.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    reduce_lr = tf.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    checkpoint = tf.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)

    callbacks = [early_stopping, reduce_lr, checkpoint]

    # Training the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=num_epochs,
        callbacks=callbacks
    )

    print("Training completed.")
    return model

def evaluate_model(model, validation_generator):
    """
    Evaluates the U-Net model on validation data using SSIM and PSNR.
    """
    total_ssim = 0.0
    total_psnr = 0.0
    num_batches = 0

    for batch_input, batch_labels in validation_generator:
        predictions = model(batch_input, training=False)  # Inference mode
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
        # Input image
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(inputs[i])
        plt.title("Input Image")
        plt.axis("off")

        # Prediction
        plt.subplot(3, num_samples, num_samples + i + 1)
        plt.imshow(predictions[i])
        plt.title("Predicted Output")
        plt.axis("off")

        # Ground Truth
        plt.subplot(3, num_samples, 2 * num_samples + i + 1)
        plt.imshow(labels[i])
        plt.title("Ground Truth")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_ssim_psnr(ssim_scores, psnr_scores):
    """
    Visualize the distribution of SSIM and PSNR across the dataset.
    """
    plt.figure(figsize=(15, 6))

    # Plot SSIM
    plt.subplot(1, 2, 1)
    plt.hist(ssim_scores, bins=20, color='b', alpha=0.7)
    plt.title('SSIM Distribution')
    plt.xlabel('SSIM')
    plt.ylabel('Frequency')

    # Plot PSNR
    plt.subplot(1, 2, 2)
    plt.hist(psnr_scores, bins=20, color='g', alpha=0.7)
    plt.title('PSNR Distribution')
    plt.xlabel('PSNR')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
