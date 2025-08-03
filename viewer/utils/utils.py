import sys
import torch
import os
import joblib
import logging
from nn.dist_net.models import ScalarNet, ImageNet, CombinedNet

def load_combined_model(model_path, device, feature_scaler_path, target_scaler_path):
    """
    Loads the trained CombinedNet model along with feature and target scalers.
    Returns the model, feature_scaler, and target_scaler.
    """
    # Load scalers
    try:
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        print("Scalers loaded successfully.")
    except FileNotFoundError:
        print(f"Scaler files not found. Please ensure scalers are saved during training.")
        sys.exit(1)

    # Initialize networks
    scalar_net = ScalarNet()
    image_net = ImageNet(input_size=(50, 50))  # Updated input size to 50x50

    model = CombinedNet(scalar_net, image_net)

    # Check if model_path exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("Loaded model_state_dict from checkpoint.")
        else:
            model.load_state_dict(checkpoint)
            logging.info("Loaded state_dict directly from checkpoint.")
    else:
        model.load_state_dict(checkpoint)
        logging.info("Loaded state_dict directly from checkpoint.")

    model.to(device)
    model.eval()
    return model, feature_scaler, target_scaler

def is_valid_bbox(x1, y1, x2, y2, width, height):
    """
    Checks if the bounding box coordinates are valid within the image dimensions.
    """
    return x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height

def crop_to_square(image):
    """
    Crops the input image to a square without distorting it.
    - If the image is taller than it is wide, crops the top.
    - If the image is wider than it is tall, crops the sides to retain the center.
    - If the image is already square, returns it unchanged.

    Parameters:
    - image (numpy.ndarray): The cropped bounding box image.

    Returns:
    - numpy.ndarray: The square-cropped image.
    """
    height, width = image.shape[:2]
    if width > height:
        # Image is wider than it is tall: crop the sides, keep the center
        start_x = (width - height) // 2
        end_x = start_x + height
        cropped = image[:, start_x:end_x]
    elif height > width:
        # Image is taller than it is wide: crop the top, keep the bottom
        start_y = height - width
        end_y = start_y + width
        cropped = image[start_y:end_y, :]
    else:
        # Image is already square
        cropped = image
    return cropped
