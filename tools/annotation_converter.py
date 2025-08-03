import json
import os
import shutil
import cv2
import numpy as np
import re

def add_gaussian_noise(image, var=10):
    """
    Adds Gaussian noise to a color image.

    Parameters:
    - image (numpy.ndarray): Color image in BGR format.
    - var (float): Variance of the Gaussian noise.

    Returns:
    - numpy.ndarray: Noisy image.
    """
    if var <= 0:
        raise ValueError("Variance must be positive.")

    sigma = var ** 0.5
    gauss = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """Helper function to generate a sort key for natural sorting."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def convert_annotations_to_yolo(input_folders, output_folder, max_distance, image_width, image_height,
                                enable_noise=False, noise_variance=10):
    """
    Converts JSON annotations from multiple input folders to YOLO format and optionally performs data augmentation by adding noise.

    Parameters:
    - input_folders (list of str): List of paths to input folders containing JSON annotation files.
    - output_folder (str): Path to the output folder where YOLO-formatted images and labels will be saved.
    - max_distance (float): Maximum distance threshold to filter annotations.
    - image_width (int): Width of the images for YOLO normalization.
    - image_height (int): Height of the images for YOLO normalization.
    - enable_noise (bool): If True, adds Gaussian noise to images for data augmentation.
    - noise_variance (float): Variance of the Gaussian noise to be added when noise augmentation is enabled.
    """
    # Prepare the output directories
    images_folder = os.path.join(output_folder, "img_40")
    labels_folder = os.path.join(output_folder, "ann_40")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # Initialize a global file counter for unique naming
    file_counter = 0

    # Iterate over each input folder
    for input_folder in input_folders:
        if not os.path.isdir(input_folder):
            print(f"Warning: Input folder '{input_folder}' does not exist or is not a directory. Skipping.")
            continue

        # Get all JSON files in the current input folder and sort them naturally
        json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
        json_files_sorted = sorted(json_files, key=natural_sort_key)

        for json_file in json_files_sorted:
            # Construct the full path to the JSON file
            json_path = os.path.join(input_folder, json_file)

            # Load the JSON data
            try:
                with open(json_path, 'r') as file:
                    data = json.load(file)
            except Exception as e:
                print(f"Error reading JSON file '{json_path}': {e}. Skipping file.")
                continue

            # Extract roll and pitch from metrics
            roll = None
            pitch = None
            for metric in data.get('metrics', []):
                metric_id = metric.get('id', '')
                if metric_id.startswith('CameraPitchMetric1'):
                    pitch = metric.get('value')
                elif metric_id.startswith('CameraRollMetric1'):
                    roll = metric.get('value')

            # Ensure roll and pitch are available
            if roll is None or pitch is None:
                print(f"Warning: Roll or pitch not found in metrics for '{json_file}'. Skipping file.")
                continue

            # Extract image data
            captures = data.get('captures', [])

            # Sort captures based on image filename numerically
            def extract_step_number(capture):
                filename = capture.get('filename', '')
                match = re.search(r'step(\d+)\.camera\.png', filename)
                return int(match.group(1)) if match else -1

            captures_sorted = sorted(captures, key=extract_step_number)

            for capture in captures_sorted:
                image_file = capture.get('filename')
                if not image_file:
                    print(f"Warning: No image filename found in capture for '{json_file}'. Skipping capture.")
                    continue

                # Path to the source image
                source_image_path = os.path.join(input_folder, image_file)
                if not os.path.isfile(source_image_path):
                    print(f"Warning: Image file '{source_image_path}' does not exist. Skipping image.")
                    continue

                # Define unique target image and label names
                target_image_name = f"{file_counter:06d}.png"
                target_label_name = f"{file_counter:06d}.txt"

                target_image_path = os.path.join(images_folder, target_image_name)
                target_label_path = os.path.join(labels_folder, target_label_name)

                # Copy the image to the output images directory
                try:
                    shutil.copy(source_image_path, target_image_path)
                except Exception as e:
                    print(f"Error copying image '{source_image_path}' to '{target_image_path}': {e}. Skipping image.")
                    continue

                # Prepare to collect distance and angle data from ConePolarPositions
                distance_dict = {}
                angle_dict = {}
                for annotation in capture.get('annotations', []):
                    if annotation.get('@type') == 'ConePolarPositions':
                        instance_ids = annotation.get('instanceIds', [])
                        radii = annotation.get('radii', [])
                        angles = annotation.get('angles', [])
                        for idx, instance_id in enumerate(instance_ids):
                            # Safely get radii and angles
                            try:
                                distance = float(radii[idx])
                                angle = float(angles[idx])
                                distance_dict[instance_id] = distance
                                angle_dict[instance_id] = angle
                            except (IndexError, ValueError, TypeError) as e:
                                print(
                                    f"Error processing ConePolarPositions in '{json_file}': {e}. Skipping instance '{instance_id}'.")
                                continue

                # Prepare the YOLO annotation data
                yolo_data = []
                for annotation in capture.get('annotations', []):
                    if annotation.get('@type') == 'type.unity.com/unity.solo.BoundingBox2DAnnotation':
                        for bbox in annotation.get('values', []):
                            instance_id = bbox.get('instanceId')
                            if instance_id is None:
                                print(f"Warning: No instanceId found in bounding box in '{json_file}'. Skipping bbox.")
                                continue

                            distance = distance_dict.get(instance_id, float('inf'))
                            angle = angle_dict.get(instance_id, float('nan'))

                            # Adjust angle
                            try:
                                angle = (angle + 180) % 360
                            except TypeError:
                                print(
                                    f"Warning: Invalid angle '{angle}' for instanceId '{instance_id}' in '{json_file}'. Skipping bbox.")
                                continue

                            # Only add the bounding box if within the specified max distance
                            if distance > max_distance:
                                continue

                            label_id = bbox.get('labelId', 'unknown')

                            if label_id == "unknown":
                                raise Exception("unknown labelId")

                            origin = bbox.get('origin', [0, 0])
                            dimension = bbox.get('dimension', [0, 0])

                            if len(origin) < 2 or len(dimension) < 2:
                                print(
                                    f"Warning: Incomplete origin or dimension for instanceId '{instance_id}' in '{json_file}'. Skipping bbox.")
                                continue

                            x_min, y_min = origin[:2]
                            bbox_width, bbox_height = dimension[:2]

                            # Validate numerical values
                            try:
                                x_min = float(x_min)
                                y_min = float(y_min)
                                bbox_width = float(bbox_width)
                                bbox_height = float(bbox_height)
                            except ValueError:
                                print(
                                    f"Warning: Non-numeric bbox values for instanceId '{instance_id}' in '{json_file}'. Skipping bbox.")
                                continue

                            # Convert to YOLO format (normalized)
                            x_center = (x_min + bbox_width / 2) / image_width
                            y_center = (y_min + bbox_height / 2) / image_height
                            width = bbox_width / image_width
                            height = bbox_height / image_height

                            # Ensure normalized values are between 0 and 1
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                                print(
                                    f"Warning: Normalized bbox values out of range for instanceId '{instance_id}' in '{json_file}'. Skipping bbox.")
                                continue

                            # Create the annotation string including roll and pitch
                            yolo_annotation = (
                                f"{label_id} "
                                f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} "
                                f"{((roll + 180) % 360):.6f} {pitch:.6f} {distance:.6f} {angle:.6f}"
                            )
                            yolo_data.append(yolo_annotation)

                # Save the YOLO annotations if there are any
                if yolo_data:
                    try:
                        with open(target_label_path, 'w') as out_file:
                            out_file.write("\n".join(yolo_data))
                    except Exception as e:
                        print(f"Error writing label file '{target_label_path}': {e}. Skipping label.")
                        # Optionally, remove the copied image if label saving fails
                        try:
                            os.remove(target_image_path)
                        except:
                            pass
                        continue

                # ----------------------------
                # Data Augmentation: Add Noise (Optional)
                # ----------------------------
                if enable_noise:
                    try:
                        # Read the copied image
                        image = cv2.imread(target_image_path)
                        if image is None:
                            print(f"Error reading image '{target_image_path}'. Skipping augmentation.")
                            continue

                        # Add Gaussian noise
                        noisy_image = add_gaussian_noise(image, var=noise_variance)

                        # Define the noisy image filename
                        noisy_image_name = f"{file_counter:06d}_noise.png"  # Use a suffix to indicate augmentation
                        noisy_image_path = os.path.join(images_folder, noisy_image_name)

                        # Save the noisy image
                        success_noisy = cv2.imwrite(noisy_image_path, noisy_image)
                        if not success_noisy:
                            print(
                                f"Failed to save noisy image: {noisy_image_path}. Skipping augmentation for this image.")
                            continue

                        # Define the noisy label filename (same as original label)
                        noisy_label_name = f"{file_counter:06d}_noise.txt"
                        noisy_label_path = os.path.join(labels_folder, noisy_label_name)

                        # Copy the label file to the noisy label file
                        shutil.copy(target_label_path, noisy_label_path)

                        # Increment the file counter for the augmented image
                        # Note: If you prefer sequential numbering, adjust accordingly
                        # Here, we use the same file_counter since the augmented files have a suffix
                        # If unique numbering without suffix is preferred, uncomment the next line
                        # file_counter += 1

                    except Exception as e:
                        print(f"Error during augmentation for image '{target_image_path}': {e}. Skipping augmentation.")
                        continue

                # Increment the file counter for the original image
                file_counter += 1

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    # Specify multiple input folders
    input_folders = [
        r'E:\pa_sim\data\solo_5\sequence.0'
        # Add more input folder paths as needed
    ]

    # Specify output folder
    output_folder = r'E:/PA/data/test_validate_sort'

    # Set parameters
    max_distance = 28  # Set your desired max distance in units
    image_width, image_height = 1280, 720

    # Set data augmentation parameters
    enable_noise = False  # Set to True to enable noise augmentation
    noise_variance = 20  # Variance for Gaussian noise

    # Run the conversion with data augmentation
    convert_annotations_to_yolo(
        input_folders,
        output_folder,
        max_distance,
        image_width,
        image_height,
        enable_noise=enable_noise,
        noise_variance=noise_variance
    )
