import os
import cv2
import numpy as np
import random
from collections import defaultdict

# ----------------------------
# Configuration Section
# ----------------------------

# Paths
images_path = r'E:\PA\data\cone_dataset\img_40'
labels_path = r'E:\PA\data\cone_dataset\ann_40'
output_base_path = r'E:\PA\data\cone_dataset\cone_dataset_balanced_24_4'  # Base path for output directories

# Split ratio
VALIDATION_RATIO = 0.2  # for validation (20% of data)

# Maximum distance
max_distance = 28

# Bin size for even dataset
bin_size = 4  # Meters

# Classes to skip
SKIP_CLASSES = []

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Desired size after resizing
RESIZE_DIM = (50, 50)  # You can adjust this as needed

# Define output directories
train_img_box_path = os.path.join(output_base_path, 'train_40', 'img_box')
train_ann_box_path = os.path.join(output_base_path, 'train_40', 'ann_box')
val_img_box_path = os.path.join(output_base_path, 'val_40', 'img_box')
val_ann_box_path = os.path.join(output_base_path, 'val_40', 'ann_box')

# Create directories if they don't exist
os.makedirs(train_img_box_path, exist_ok=True)
os.makedirs(train_ann_box_path, exist_ok=True)
os.makedirs(val_img_box_path, exist_ok=True)
os.makedirs(val_ann_box_path, exist_ok=True)

# ----------------------------
# Utility Functions
# ----------------------------

def is_valid_bbox(x1, y1, x2, y2, width, height):
    """
    Checks if the bounding box coordinates are valid within the image dimensions.
    """
    return x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height

def crop_to_square(image):
    """
    Crops the input image to a square without distorting it.
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

def get_bin(distance):
    if distance >= max_distance:
        return None
    bin_index = int(distance // bin_size)
    bin_start = bin_index * bin_size
    bin_end = bin_start + bin_size
    return (bin_start, bin_end)

# ----------------------------
# Main Processing
# ----------------------------

skip_cont = 0
annotations = []

# Collect all annotations
for label_file in sorted(os.listdir(labels_path)):
    if not label_file.endswith('.txt'):
        continue

    # Read label file
    label_file_path = os.path.join(labels_path, label_file)
    with open(label_file_path, 'r') as file:
        lines = file.readlines()

    # Get corresponding image file
    img_file = label_file.replace('.txt', '.png')
    img_path = os.path.join(images_path, img_file)

    # Load image to get dimensions
    image = cv2.imread(img_path)
    if image is None:
        print(f"Image not found: {img_path}")
        continue

    height, width, _ = image.shape

    # Process each line (bounding box)
    for i, line in enumerate(lines):
        elements = line.strip().split()
        if len(elements) != 9:
            print(f"Invalid label format in {label_file}, line {i}")
            continue

        # Extract data
        try:
            class_id = int(elements[0])
            center_x = float(elements[1]) * width
            center_y = float(elements[2]) * height
            bbox_width = float(elements[3]) * width
            bbox_height = float(elements[4]) * height
            roll = float(elements[5])
            pitch = float(elements[6])
            distance = float(elements[7])
            angle = float(elements[8])
        except ValueError:
            print(f"Invalid numerical values in {label_file}, line {i}")
            continue

        # Skip classes in SKIP_CLASSES
        if class_id in SKIP_CLASSES:
            continue

        # Skip if distance > max_distance
        if distance > max_distance:
            skip_cont += 1
            continue

        # Store annotation data
        annotations.append({
            'label_file': label_file,
            'line': line.strip(),
            'img_file': img_file,
            'img_path': img_path,
            'index': i,
            'class_id': class_id,
            'center_x': center_x,
            'center_y': center_y,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'roll': roll,
            'pitch': pitch,
            'distance': distance,
            'angle': angle,
            'image_width': width,
            'image_height': height,
        })

# Bin annotations by distance
bins_dict = defaultdict(list)
for annotation in annotations:
    bin_range = get_bin(annotation['distance'])
    if bin_range is None:
        continue
    bins_dict[bin_range].append(annotation)

# Find the smallest bin size
bin_sizes = {bin_range: len(annotations_list) for bin_range, annotations_list in bins_dict.items()}
min_bin_size = min(bin_sizes.values())

# Balance the dataset
balanced_annotations = []
for bin_range, annotations_in_bin in bins_dict.items():
    if len(annotations_in_bin) > min_bin_size:
        annotations_in_bin = random.sample(annotations_in_bin, min_bin_size)
    balanced_annotations.extend(annotations_in_bin)

# Group annotations by image path
img_annotations = defaultdict(list)
for annotation in balanced_annotations:
    img_annotations[annotation['img_path']].append(annotation)

# Process images and annotations
for img_path, annotations_list in img_annotations.items():
    image = cv2.imread(img_path)
    if image is None:
        print(f"Image not found: {img_path}")
        continue

    height, width, _ = image.shape

    for annotation in annotations_list:
        # Extract data
        index = annotation['index']
        line = annotation['line']
        class_id = annotation['class_id']
        center_x = annotation['center_x']
        center_y = annotation['center_y']
        bbox_width = annotation['bbox_width']
        bbox_height = annotation['bbox_height']
        label_file = annotation['label_file']

        # Calculate margins and bounding box coordinates
        margin_x = int(bbox_width * 0.2)  # 20% total margin on x-axis
        margin_y = int(bbox_height * 0.2)  # 20% total margin on y-axis

        # Adjust the coordinates to include the margin
        x1 = int(center_x - bbox_width / 2 - margin_x // 2)
        y1 = int(center_y - bbox_height / 2 - margin_y // 2)
        x2 = int(center_x + bbox_width / 2 + margin_x // 2)
        y2 = int(center_y + bbox_height / 2 + margin_y // 2)

        # Ensure coordinates are within image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Validate bounding box coordinates
        if not is_valid_bbox(x1, y1, x2, y2, width, height):
            print(f"Invalid bounding box in {label_file}, line {index}: ({x1}, {y1}), ({x2}, {y2})")
            continue

        # Crop the bounding box from the image with the extra margin
        cropped_img = image[y1:y2, x1:x2]

        # Check if the cropped image is valid
        if cropped_img.size == 0:
            print(f"Empty crop in {label_file}, line {index}")
            continue

        # Crop to square without distorting
        square_cropped_img = crop_to_square(cropped_img)

        # Convert to grayscale
        gray_img = cv2.cvtColor(square_cropped_img, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale image
        resized_gray = cv2.resize(gray_img, RESIZE_DIM, interpolation=cv2.INTER_AREA)

        # Decide whether to assign to train or val
        split = 'train' if random.random() > VALIDATION_RATIO else 'val'

        if split == 'train':
            img_output_path = train_img_box_path
            label_output_path = train_ann_box_path
        else:
            img_output_path = val_img_box_path
            label_output_path = val_ann_box_path

        # Save processed image
        cropped_img_name = f"{os.path.splitext(label_file)[0]}_{index:03d}.png"
        cropped_img_path = os.path.join(img_output_path, cropped_img_name)
        success = cv2.imwrite(cropped_img_path, resized_gray)
        if not success:
            print(f"Failed to save image: {cropped_img_path}")
            continue

        # Save corresponding label line as-is
        cropped_label_name = f"{os.path.splitext(label_file)[0]}_{index:03d}.txt"
        cropped_label_path = os.path.join(label_output_path, cropped_label_name)
        with open(cropped_label_path, 'w') as cropped_label_file:
            cropped_label_file.write(annotation['line'] + '\n')

print(f"Skipped due to distance: {skip_cont}")
print("Processing completed.")
