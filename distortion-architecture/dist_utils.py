import os
from torchvision import transforms
from PIL import Image


def resize_with_padding(image, target_dim):
    """
    resizes an image while preserving aspect ratio and adds padding to satisfy the target dimensions
    
    :param image: image to resize.
    :param target_dim: Target dimensions for image.
    :return: PIL Image of dimensions target_height x target_width.
    """
    img_width, img_height = image.size
    scale_factor = min(target_dim / img_height, target_dim / img_width)

    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)

    resized_image = image.resize((new_width, new_height))

    pad_left = (target_dim - new_width) // 2
    pad_right = target_dim - new_width - pad_left
    pad_top = (target_dim - new_height) // 2
    pad_bottom = target_dim - new_height - pad_top

    padded_image = transforms.functional.pad(resized_image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

    return padded_image


def preprocess_images(input_directory, target_dim):
    """
    preprocesses images from the input directory + resizes them + adds padding + converts them to tensors
    
    :param input_directory: Path to the images.
    :param target_dim: Target dimensions for the images (square).
    :return: List of clean images as tensors.
    """

    # I only saw pngs and jpgs in the dataset but I included jpegs as well just in case
    image_paths = [os.path.join(input_directory, filename) for filename in os.listdir(input_directory) if filename.endswith(('jpg', 'png', 'jpeg'))]
    
    preprocess_transform = transforms.ToTensor()

    preprocessed_images = []

    for image_path in image_paths:
        image = Image.open(image_path)
        padded_image = resize_with_padding(image, target_dim)
        preprocessed_images.append(preprocess_transform(padded_image))

    return preprocessed_images


def lbfgs_closure(optimizer, model, data, target, criterion):
    """
    closure function for lbfgs optimization; computes loss + gradients

    :param optimizer: The optimizer to perform step with.
    :param model: The model to make predictions with.
    :param data: The input data for the model.
    :param target: The target values (ground truth) for comparison.
    :param criterion: The loss function used to calculate the loss.
    :return: The computed loss.
    """
    optimizer.zero_grad()

    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    return loss
