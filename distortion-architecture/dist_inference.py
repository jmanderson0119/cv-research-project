import os
import shutil
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from dist_net import DistortionNet
from dist_utils import resize_with_padding, process_distortion_maps
from dist_config import image_settings

def generate_binary_masks(image_dir, model_path, output_maps_dir, output_masks_dir):
    """
    generate binary distortion masks for an image set
    
    :param image_dir: Path containing input images
    :param model_path: Path to the trained model
    :return: List of binary distortion masks
    """
    
    # determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build directory paths
    os.makedirs(output_maps_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # load the model
    model = DistortionNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # preprocessing transform
    preprocess_transform = transforms.Compose([transforms.ToTensor()])
    
    # process images
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # full path to image
            image_path = os.path.join(image_dir, filename)
            
            # open and preprocess image
            image = Image.open(image_path)
            padded_image = resize_with_padding(image, image_settings["target_dim"])
            tensor_image = preprocess_transform(padded_image).unsqueeze(0).to(device)
            
            # disable gradient calculation + get predicted map
            with torch.no_grad(): distortion_map = model(tensor_image).cpu().numpy()
            
            # save distortion map
            output_map_path = os.path.join(output_maps_dir, f"distortion_map_{i}.npy")
            np.save(output_map_path, distortion_map)

    # generates binary masks and saves to the specified directory
    process_distortion_maps(output_maps_dir, output_masks_dir)
