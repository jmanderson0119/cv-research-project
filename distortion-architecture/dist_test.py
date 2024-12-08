import os
import torch
import numpy as np
from dist_net import DistortionNet
from dist_utils import *
from dist_config import *

def dist_main():
    # sets the training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device set to {device}")

    # load model
    model = DistortionNet().to(device)
    model.load_state_dict(torch.load(distortion_detection_model["model_destination"], weights_only=True))
    print("model loaded")

    ## testing 
    distorted_test_images = load_images_from_directory(image_settings["distorted_test_path"], image_settings["target_dim"])
    print("test images loaded")

    # set model to evaluation mode
    model.eval()

    # save ground truth distortion maps
    ground_truth_maps_output_dir = os.path.join(distortion_detection_model["test_output_destination"], "ground_truth_maps")

    # get random index for data visualization of predicted and ground truth maps and masks
    random_index = random.randint(0, 1787)
    # display a random gt map for visualization
    display_map(ground_truth_maps_output_dir, random_index)


    predicted_maps_output_dir = os.path.join(distortion_detection_model["test_output_destination"], "predicted_maps")
    os.makedirs(predicted_maps_output_dir, exist_ok=True)
    # disable gradient calculation for inference
    with torch.no_grad():
        predicted_maps = []
        for i, distorted_image in enumerate(distorted_test_images):
            # prepare the image
            distorted_image = distorted_image.unsqueeze(0).to(device)
            
            # get predicted distortion map
            distortion_map = model(distorted_image)
            
            # save the raw distortion map
            output_path = os.path.join(predicted_maps_output_dir, f"predicted_distortion_map_{i+1}.npy")
            np.save(output_path, distortion_map.squeeze().cpu().numpy())
            print(f"Saved the predicted distortion map at {output_path}")
            
            predicted_maps.append(distortion_map.squeeze().cpu().numpy())

    # display a random predicted map for visualization
    display_map(predicted_maps_output_dir, random_index)

    print("Testing done. Distortion maps saved in test_distortion_maps/")

    # process the saved predicted distortion maps as binary masks
    predicted_masks_dir = os.path.join(distortion_detection_model["binary_mask_destination"], "predicted_masks")
    os.makedirs(predicted_masks_dir, exist_ok=True)
    
    # save predicted binary masks 
    predicted_binary_masks = upsample_distortion_maps(predicted_maps_output_dir)
    save_binary_masks(predicted_binary_masks, predicted_masks_dir)

    print(f"predicted binary masks are saved in {predicted_masks_dir}")
    
    # display a random predicted mask for visualization
    display_map(predicted_masks_dir, random_index) 

    # process and save ground truth distortion maps as binary masks
    ground_truth_masks_dir = os.path.join(distortion_detection_model["binary_mask_destination"], "ground_truth_masks")
    os.makedirs(ground_truth_masks_dir, exist_ok=True)
    
    # save ground truth binary masks
    ground_truth_binary_masks = upsample_distortion_maps(ground_truth_maps_output_dir)
    save_binary_masks(ground_truth_binary_masks, ground_truth_masks_dir)

    print(f"ground-truth binary masks are saved in {ground_truth_masks_dir}")

    # display a random gt mask for visualization
    display_map(ground_truth_masks_dir, random_index)

    print(f"binary masks processed and saved in {distortion_detection_model['binary_mask_destination']}")

    ## Model Evaluation
    # load ground truth and predicted masks
    ground_truth_masks = load_ground_truth_masks(ground_truth_masks_dir)
    predicted_masks = load_predicted_masks(predicted_masks_dir)

    print("\n******* Distortion Detection Model *******")
    
    # weighted Accuracy
    weighted_accuracies = []
    for gt_mask, pred_mask in zip(ground_truth_masks, predicted_masks):
        wa = weighted_accuracy(gt_mask, pred_mask)
        weighted_accuracies.append(wa)
    
    print(f"Weighted Accuracy:")
    print(f"  Mean: {np.mean(weighted_accuracies):.4f}")
    print(f"  Std: {np.std(weighted_accuracies):.4f}\n\n")

    # accuracy
    accuracies = []
    for gt_mask, pred_mask in zip(ground_truth_masks, predicted_masks):
        a = accuracy(gt_mask, pred_mask)
        accuracies.append(a)
    
    print(f"Accuracy:")
    print(f"  Mean: {np.mean(accuracies):.4f}")
    print(f"  Std: {np.std(accuracies):.4f}\n\n")


if __name__ == "__main__": dist_main()