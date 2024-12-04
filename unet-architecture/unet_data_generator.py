import numpy as np
from tensorflow import Sequence
from process_input import process_input

class DataGenerator(Sequence):
    """
    Custom data generator for U-Net
    Generates batches of data from traffic images and distortion masks
    """
    def __init__(self, traffic_images, distortion_masks, batch_size, distortion_model, shuffle=True):
        self.traffic_images = traffic_images
        self.distortion_masks = distortion_masks
        self.batch_size = batch_size
        self.distortion_model = distortion_model
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.traffic_images))

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(len(self.traffic_images) / self.batch_size))

    def on_epoch_end(self):
        # shuffle data after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # generate one batch of data
        
        # get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        traffic_images = self.traffic_images[batch_indexes]
        batch_distortion_masks = self.distortion_masks[batch_indexes]

        # process inputs
        batch_input = process_input(traffic_images, self.distortion_model)
        
        # Concatenate traffic images with distortion masks
        batch_input = np.concatenate([batch_input, batch_distortion_masks], axis=-1)  # concatenate along the last axis

        return batch_input, batch_distortion_masks
