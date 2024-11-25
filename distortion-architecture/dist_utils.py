import os
import torch
import cv2
import numpy as np
import random
import os
import shutil
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from dist_config import image_settings


def load_images_from_directory(directory, target_dim):
    """
    Load and preprocess images from a directory.
    
    :param directory: Path to the image directory.
    :param target_dim: Target dimensions for the images (square).
    :return: List of preprocessed images as tensors.
    """

    # I only saw jpgs in the new dataset but I included the other two common formats just in case
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(('jpg', 'png', 'jpeg'))]
    
    preprocess_transform = transforms.ToTensor()

    preprocessed_images = []

    for image_path in image_paths:
        image = Image.open(image_path)
        padded_image = resize_with_padding(image, target_dim)
        preprocessed_images.append(preprocess_transform(padded_image))

    return preprocessed_images


def resize_with_padding(image, target_dim):
    """
    resizes an image while preserving aspect ratio and adds padding to satisfy the target dimensions

    :param image: image to resize
    :param target_dim: target dimension for image
    :return: PIL Image of dimensions target_dim x target_dim
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


def compute_distortion_map(self, clean_image, distorted_image):
    """
    Compute the ground truth distortion map by comparing the clean and distorted images

    :param clean_image: undistorted image as a tensor
    "param distorted image: distorted image as a tensor
    
    :return: the ground-truth distortion map as a tensor
    """
    batch_size, _, height, width = clean_image.shape

    # init distortion map
    distortion_map = torch.zeros(batch_size, 1, height // self.distortion_grid_dim, width // self.distortion_grid_dim, device=clean_image.device)

    # iterate over grid regions
    for i in range(0, height, self.distortion_grid_dim):
        for j in range(0, width, self.distortion_grid_dim):
            # extract apprprite region for each image
            clean_block = clean_image[:, :, i:i+self.distortion_grid_dim, j:j+self.distortion_grid_dim]
            distorted_block = distorted_image[:, :, i:i+self.distortion_grid_dim, j:j+self.distortion_grid_dim]

            # computes distortion score
            block_distortion = torch.mean((clean_block - distorted_block) ** 2, dim=(1, 2, 3), keepdim=True)  # Shape (batch_size, 1, 1, 1)
            
            # assign score to appropriate location
            x_idx = i // self.distortion_grid_dim
            y_idx = j // self.distortion_grid_dim
            distortion_map[:, 0, x_idx, y_idx] = block_distortion.squeeze()

    return distortion_map


def elastic_deformation(image, alpha=5.0, sigma=1.0, region_center=None, region_radius=None):
    shape = image.shape[:2]
    random_state = np.random.RandomState(None)

    displacement_field = [random_state.randn(*shape) * alpha for _ in range(2)]

    displacement_field[0] = ndimage.gaussian_filter(displacement_field[0], sigma=sigma, mode="constant", cval=0)
    displacement_field[1] = ndimage.gaussian_filter(displacement_field[1], sigma=sigma, mode="constant", cval=0)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    distorted_x = x + displacement_field[0]
    distorted_y = y + displacement_field[1]

    distorted_image = np.copy(image)

    if region_center is not None and region_radius is not None:
        cx, cy = region_center
        mask = (x - cx)**2 + (y - cy)**2 <= region_radius**2

        for i in range(image.shape[2]):
            distorted_image[..., i][mask] = ndimage.map_coordinates(image[..., i], 
                                                                   [distorted_y[mask], distorted_x[mask]], 
                                                                   order=1, mode='reflect')

    distorted_image = np.clip(distorted_image, 0, 255).astype(np.uint8)

    return distorted_image


def apply_gaussian_blur(image, sigma=2): return cv2.GaussianBlur(image, (0, 0), sigma)


def process_images(input_dir, output_dir, alpha=5.0, sigma_blur=2.0, max_regions=4, min_regions=2, region_radius=50):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])

    for filename in input_files:
        input_path = os.path.join(input_dir, filename)

        image = cv2.imread(input_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape

        regions = []
        for _ in range(max_regions):
            cx = random.randint(region_radius, width - region_radius)
            cy = random.randint(region_radius, height - region_radius)
            radius = random.randint(region_radius, min(width, height) // 4)
            regions.append((cx, cy, radius))

        image_deformed = np.copy(image)
        num_regions = random.randint(min_regions, max_regions)
        selected_regions = random.sample(regions, num_regions)

        for region in selected_regions:
            cx, cy, radius = region
            image_deformed = elastic_deformation(image_deformed, alpha, sigma=1.0, region_center=(cx, cy), region_radius=radius)

        image_blurred = apply_gaussian_blur(image_deformed, sigma=sigma_blur)

        image_final = np.clip(image_blurred, 0, 255).astype(np.uint8)

        image_final = cv2.cvtColor(image_final, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(output_dir, f"modified_{filename}")
        cv2.imwrite(output_path, image_final)
        print(f"Processed and saved: {output_path}")


def sort_images_in_directory(input_directory):
    files = os.listdir(input_directory)
    image_files = [f for f in files if f.startswith('modified_') and f.endswith('.jpg')]
    sorted_files = sorted(image_files, key=lambda f: int(f.split('_')[1].split('.')[0]))
    temp_directory = os.path.join(input_directory, 'temp')
    os.makedirs(temp_directory, exist_ok=True)
    
    try:
        for file in sorted_files:
            old_path = os.path.join(input_directory, file)
            shutil.move(old_path, os.path.join(temp_directory, file))
        
        for index, file in enumerate(sorted_files, start=513):
            new_path = os.path.join(input_directory, f"modified_{index}.jpg")
            shutil.move(os.path.join(temp_directory, file), new_path)
        
    finally: shutil.rmtree(temp_directory)


process_images(image_settings["clean_train_path"], image_settings["distorted_train_path"])
sort_images_in_directory(image_settings["distorted_train_path"])
