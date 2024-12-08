import os
import torch
import cv2
import numpy as np
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from dist_config import *


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


def compute_distortion_map(clean_image_batch, distorted_image_batch, distortion_grid_dim):
    """
    Compute distortion maps for each image pair in the batch.
    
    :param clean_image_batch: Batch tensor of clean images
    :param distorted_image_batch: Batch tensor of distorted images
    :param distortion_grid_dim: Grid dimension for distortion calculation
    :return: Batch tensor of distortion maps
    """
    # ensure the batch sizes are the same
    assert clean_image_batch.shape[0] == distorted_image_batch.shape[0], \
        f"mismatch: clean set has {clean_image_batch.shape[0]} images, distorted batch set has {distorted_image_batch.shape[0]} images"

    # lists to store individual image tensors
    distortion_maps = []
    for i in range(clean_image_batch.shape[0]):
        distortion_map = _compute_single_distortion_map(clean_image_batch[i], distorted_image_batch[i], distortion_grid_dim)
        distortion_maps.append(distortion_map.unsqueeze(0))
        print(f"generated gt-distortion map {i+1}")
        
    return torch.cat(distortion_maps, dim=0)



def _compute_single_distortion_map(clean_image, distorted_image, distortion_grid_dim):
    """
    Compute distortion map for a single image pair.
    
    :param clean_image: clean image tensor
    :param distorted_image: Corresponding distorted image tensor
    :param distortion_grid_dim: Grid dimension for distortion calculation
    :return: Distortion map tensor
    """
    _, height, width = clean_image.shape

    # init distortion map
    distortion_map = torch.zeros(height // distortion_grid_dim, width // distortion_grid_dim, device=clean_image.device)

    # Iterate over grid regions
    for i in range(0, height, distortion_grid_dim):
        for j in range(0, width, distortion_grid_dim):
            # extract appropriate region for each image
            clean_block = clean_image[:, i:i+distortion_grid_dim, j:j+distortion_grid_dim]
            distorted_block = distorted_image[:, i:i+distortion_grid_dim, j:j+distortion_grid_dim]

            # compute distortion
            block_distortion = torch.mean((clean_block - distorted_block) ** 2, dim=(0, 1, 2), keepdim=True)[0]

            # assign score to appropriate location
            x_idx = i // distortion_grid_dim
            y_idx = j // distortion_grid_dim
            distortion_map[x_idx, y_idx] = block_distortion

    # percentile normalization
    lower_percentile = torch.quantile(distortion_map, 0.1)
    upper_percentile = torch.quantile(distortion_map, 0.9)
    distortion_map = torch.clamp((distortion_map - lower_percentile) / (upper_percentile - lower_percentile), 0, 1)

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


def apply_random_gaussian_blur(image, num_blur_spots=35, min_sigma=1.0, max_sigma=2.4, min_radius=10, max_radius=200):
    # create a copy of the image to modify
    blurred_image = image.copy()
    height, width = image.shape[:2]
    
    for _ in range(num_blur_spots):
        # random center coordinates
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        
        # random blur parameters
        sigma = np.random.uniform(min_sigma, max_sigma)
        radius = np.random.randint(min_radius, max_radius)
        
        # calculate kernel size
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # create a mask for this blur spot
        mask = np.zeros_like(image[:,:,0])
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        mask = mask / 255.0
        
        # blur the entire image
        blurred_region = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        # blend the blurred region with the original image using the mask
        for c in range(image.shape[2]):
            blurred_image[:,:,c] = (
                blurred_image[:,:,c] * (1 - mask) + 
                blurred_region[:,:,c] * mask
            )
    
    return blurred_image.astype(np.uint8)


def process_images(input_dir, output_dir, alpha=5.0, max_regions=4, min_regions=2, region_radius=50):
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

        image_blurred = apply_random_gaussian_blur(image_deformed)

        image_final = np.clip(image_blurred, 0, 255).astype(np.uint8)

        image_final = cv2.cvtColor(image_final, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image_final)
        print(f"Processed and saved: {output_path}")
        

def upsample_distortion_maps(distortion_maps_dir, original_height=720, original_width=1280):
    distortion_grid_dim = dist_net_settings["distortion_grid_dim"]
    distortion_score_thresh = float(dist_net_settings["distortion_score_thresh"])
    
    # calculate region sizes
    region_height = original_height // distortion_grid_dim
    region_width = original_width // distortion_grid_dim
    
    binary_masks = []
    
    for map_file in os.listdir(distortion_maps_dir):
        if not map_file.endswith('.npy'): continue
        
        # load the distortion map
        distortion_map = np.load(os.path.join(distortion_maps_dir, map_file))


        # create a full-size binary mask
        binary_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        
        # iterate through each grid cell
        for i in range(distortion_map.shape[0]):
            for j in range(distortion_map.shape[1]):
                # get the distortion score for this grid cell
                score = distortion_map[i, j]

                # determine the region in the original image
                start_y = i * int(original_height / distortion_map.shape[0])
                start_x = j * int(original_width / distortion_map.shape[1])
                
                # fill the entire region with 1 or 0 based on the threshold
                if int(score*1000) >= int(distortion_score_thresh*1000):
                    binary_mask[start_y:start_y+region_height, start_x:start_x+region_width] = 1
                else:
                    binary_mask[start_y:start_y+region_height, start_x:start_x+region_width] = 0 

        binary_masks.append(binary_mask)
    
    return binary_masks


def save_binary_masks(binary_masks, output_dir):
    """
    save binary masks to the output directory
    
    :param binary_masks: list of binary masks
    :param output_dir: directory to save masks
    """    
    for i, mask in enumerate(binary_masks):
        output_path = os.path.join(output_dir, f"binary_mask_{i+1}.npy")
        np.save(output_path, mask)
        print(f"Saved binary mask: {output_path}")


# saves raw distortion maps as binary masks
def process_distortion_maps(input_maps_dir, output_masks_dir):
    """
    upsamples and saves binary masks
    
    :param input_maps_dir: directory with input distortion maps
    :param output_masks_dir: directory to save output binary masks
    """
    # upsample distortion maps to binary masks
    binary_masks = upsample_distortion_maps(input_maps_dir)
    
    # save binary masks
    save_binary_masks(binary_masks, output_masks_dir)


def weighted_accuracy(ground_truth_mask, predicted_mask):
    """
    compute weighted accuracy for binary masks.
    
    :param ground_truth_mask: Ground truth binary mask
    :param predicted_mask: Predicted binary mask
    :return: Weighted accuracy score
    """
    # flatten masks
    gt_flat = ground_truth_mask.flatten()
    pred_flat = predicted_mask.flatten()
    
    # compute tn tp fn fp
    tn = np.sum((gt_flat == 0) & (pred_flat == 0))
    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
        
    # weighted accuracy where negative is weight and positive
    weighted_acc = (0.5 * ((tp)/(tp + fn))) + (0.5 * ((tn)/(tn + fp)))
    
    return weighted_acc


def accuracy(ground_truth_mask, predicted_mask):
    """
    compute accuracy for binary masks.
    
    :param ground_truth_mask: Ground truth binary mask
    :param predicted_mask: Predicted binary mask
    :return: accuracy score
    """
    # flatten masks
    gt_flat = ground_truth_mask.flatten()
    pred_flat = predicted_mask.flatten()
    
    # compute tn tp fn fp
    tn = np.sum((gt_flat == 0) & (pred_flat == 0))
    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
        
    # accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    return acc


def load_ground_truth_masks(ground_truth_masks_dir):
    """
    load ground truth binary masks from a directory
    
    :param ground_truth_masks_dir: Path to directory containing ground truth masks
    :return: List of ground truth binary masks
    """
    ground_truth_masks = []
    
    # check directory exists
    if not os.path.exists(ground_truth_masks_dir): raise ValueError(f"Directory {ground_truth_masks_dir} does not exist.")
    
    # iterate through mask files
    for mask_file in sorted(os.listdir(ground_truth_masks_dir)):
        if mask_file.endswith('.npy'):
            mask_path = os.path.join(ground_truth_masks_dir, mask_file)
            mask = np.load(mask_path)
            ground_truth_masks.append(mask)
    
    return ground_truth_masks


def load_predicted_masks(predicted_masks_dir, threshold=0.6):
    """
    load predicted binary masks from a directory.
    
    :param predicted_masks_dir: Path to directory containing predicted masks
    :param threshold: Threshold for converting scores to binary masks
    :return: List of predicted binary masks
    """
    predicted_masks = []
    
    # check directory exists
    if not os.path.exists(predicted_masks_dir): raise ValueError(f"Directory {predicted_masks_dir} does not exist.")
    
    # iterate through mask files
    for mask_file in sorted(os.listdir(predicted_masks_dir)):
        if mask_file.endswith('.npy'):
            mask_path = os.path.join(predicted_masks_dir, mask_file)
            mask = np.load(mask_path)
            predicted_masks.append((mask > threshold).astype(np.uint8))
    
    return predicted_masks

def load_predicted_scores(predicted_maps_dir):
    """
    load predicted distortion map scores from a directory.
    
    :param predicted_maps_dir: Path to directory containing predicted distortion maps
    :return: List of predicted score maps
    """
    predicted_scores = []
    
    # check directory exists
    if not os.path.exists(predicted_maps_dir): raise ValueError(f"Directory {predicted_maps_dir} does not exist.")
    
    # iterate through map files
    for map_file in sorted(os.listdir(predicted_maps_dir)):
        if map_file.endswith('.npy'):
            map_path = os.path.join(predicted_maps_dir, map_file)
            score_map = np.load(map_path)
            predicted_scores.append(score_map)
    
    return predicted_scores


def load_ground_truth_distortion_maps(maps_directory):
    """
    Load pre-computed ground truth distortion maps from a directory.
    
    :param maps_directory: Path to the directory containing ground truth distortion maps
    :return: Tensor of ground truth distortion maps
    """
    # Check if directory exists
    if not os.path.exists(maps_directory):
        raise ValueError(f"Directory {maps_directory} does not exist.")
    
    # List and sort .npy files
    map_files = sorted([f for f in os.listdir(maps_directory) if f.endswith('.npy')])
    
    # Check if any .npy files exist
    if not map_files:
        raise ValueError(f"No .npy files found in {maps_directory}")
    
    # Load distortion maps
    ground_truth_distortion_maps = []
    for map_file in map_files:
        file_path = os.path.join(maps_directory, map_file)
        
        # Load numpy array and convert to tensor, add singleton dimension
        distortion_map = torch.from_numpy(np.load(file_path)).unsqueeze(0)
        ground_truth_distortion_maps.append(distortion_map)
    
    # Convert to a single tensor
    ground_truth_distortion_maps = torch.cat(ground_truth_distortion_maps, dim=0)
    
    print(f"Loaded {ground_truth_distortion_maps.shape[0]} pre-computed ground truth distortion maps")
    
    return ground_truth_distortion_maps


def display_map(directory, i):
    """
    loads the ith .npy file from the specified directory and creates a colormap
    
    :params directory: path to a directory containing maps or masks
    :params i: indexer

    :return: randomly selected array as a colormap
    """
    # get list of .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    # check if .npy files exist
    if not npy_files: raise ValueError(f"no .npy files found in {directory}")
    if i < 0 or i >= len(npy_files): raise ValueError(f"invalid index. got: {i}, expected: value in range [{0}, {len(npy_files)-1}]")
    
    # random file
    map_file = npy_files[i]
    file_path = os.path.join(directory, map_file)
    
    # load the 2D array
    data = np.load(file_path)
    
    # squeeze array data to 2d
    if data.ndim > 2: data = data.squeeze()
        
    # create the figure and plot the heatmap
    plt.figure(figsize=(10, 8))
    
    # Use a fixed alpha if needed
    plt.imshow(data, cmap='Greens', alpha=0.7, interpolation='nearest')
    
    plt.colorbar(label='Distortion')
    plt.title(f'{map_file}')
    plt.tight_layout()
    plt.show()
    
    return data