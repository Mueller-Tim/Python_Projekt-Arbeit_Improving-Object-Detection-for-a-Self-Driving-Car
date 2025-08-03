# s25_pa_perception

This Code part of the ZHAW Project Work (PA) **Improving Object Detection for a Self-Driving
Car** by Nils Keller and Tim Müller

# DistNet

The training of DistNet is handled by the script located at `nn/dist_net/train.py`.

## Training Parameters

- **num_epochs_scalar**: Number of epochs for training the scalar network (default: 20)
- **num_epochs_combined_frozen**: Number of epochs for training combined network with frozen scalar network (default: 10)
- **num_epochs_combined_unfrozen**: Number of epochs for training combined network with unfrozen networks (default: 40)
- **batch_size**: Batch size for training (default: 512)
- **lr**: Learning rate (default: 0.0005)
- **early_stop_scalarNet**: Early stopping patience for scalar network (default: 5)
- **early_stop_combinedNet_frozen**: Early stopping patience for combined network with frozen scalar net (default: 5)
- **early_stop_combinedNet**: Early stopping patience for combined network (default: 10)

### Distance-Based Sample Weighting
The training implements a distance-based sample weighting strategy to handle class imbalance:

Distance thresholds: [0-8m], [8-16m], [>16m] \
Corresponding weights: 8.0, 4.0, 1.0

This ensures that closer objects (which are more critical) receive higher weights during training

# Tools

The 'tools' folder contains useful tools for creating a DistNet dataset and evaluating a DistNet model.

## annotation_converter

Converts the synthetic data from the Unity Simulator to a YOLO style annotation

**To create a DistNet dataset from Unity, use this tool first.**

### Input Folders
- **Type**: List of strings
- **Description**: Paths to input folders containing JSON annotations and images
- **Example**:
```python
input_folders = [r'E:\pa_sim\data\solo_5\sequence.0']
```

### Output Folder
- **Type**: String  
- **Description**: Path to the folder where YOLO-formatted images and labels will be saved
- **Example**:
```python
output_folder = r'E:/PA/data/test_validate_sort'
```

### Maximum Distance
- **Type**: Float
- **Description**: Maximum distance threshold to filter annotations. Objects farther than this distance will be excluded
- **Default**: `28`
- **Example**:
```python
max_distance = 28
```

### Enable Noise
- **Type**: Boolean
- **Description**: Enable or disable Gaussian noise augmentation for the images
- **Default**: `False`
- **Example**:
```python
enable_noise = True
```

### Noise Variance
- **Type**: Float
- **Description**: Variance of the Gaussian noise to be added when `enable_noise` is `True`
- **Default**: `20`
- **Example**:
```python
noise_variance = 20
```

## box_cropper

This script processes image datasets and their annotations, balancing them based on object distance and creating train/validation splits. 
It includes functionality for cropping objects, converting to grayscale, and resizing images.

### Configuration Parameters

### Path Settings
```python
# Input Paths
images_path = r'E:\PA\data\cone_dataset\img_40'      # Source images directory
labels_path = r'E:\PA\data\cone_dataset\ann_40'      # Source annotations directory
output_base_path = r'E:\PA\data\cone_dataset\cone_dataset_balanced_24_4'  # Output base directory
```

### Dataset Split Configuration
- **Type**: Float
- **Description**: Ratio for validation split
- **Default**: `0.2` (20% validation, 80% training)
```python
VALIDATION_RATIO = 0.2
```

### Distance Parameters
- **Type**: Integer
- **Description**: Maximum distance threshold for object detection
- **Default**: `28` (meters)
```python
max_distance = 28
```

### Binning Configuration
- **Type**: Integer
- **Description**: Size of distance bins for dataset balancing
- **Default**: `4` (meters)
```python
bin_size = 4
```

### Image Processing
- **Type**: Tuple (Integer, Integer)
- **Description**: Target dimensions for resized images
- **Default**: `(50, 50)`
```python
RESIZE_DIM = (50, 50)
```

### Output Directory Structure
```
output_base_path/
├── train_40/
│   ├── img_box/     # Training images
│   └── ann_box/     # Training annotations
└── val_40/
    ├── img_box/     # Validation images
    └── ann_box/     # Validation annotations
```

## error_calc

This tool is used to calculate the difference between the reference cones and predicted cones for the DistNet Viewer.

to use the tool, change the **frame_path** in the script with an export JSON from the DistNet Viewer.

# DistNet Viewer

DistNet Viewer is an application that runs the entire DistNet pipeline (YOLO + DistNet) including visualization.

## Configuration

All configuration parameters are defined in the `viewer/config.py` file. Ensure that the paths to models, scalers, and other resources are correctly set.

## Key Configuration Parameters

- **YOLO_MODEL_PATH**: Path to the YOLO model file.
- **COMBINED_MODEL_PATH**: Path to the CombinedNet model file.
- **FEATURE_SCALER_PATH**: Path to the feature scaler pickle file.
- **TARGET_SCALER_PATH**: Path to the target scaler pickle file.
- **VIDEO_SOURCE**: Video source index (e.g., `0` for the default webcam).
- **YOLO_CLASS_MAPPING**: Mapping of YOLO class IDs to CombinedNet class IDs, names, and colors.
- **REFERENCE_CONES**: Predefined reference cones with coordinates and attributes.
- **SHOW_REFERENCE_CONES_BY_DEFAULT**: Boolean to show reference cones on startup.

## Running the Application

Run the viewer/main.py in a IDE

Only tested with Pycharm

## Prerequisites

Python 3.9 or higher

pip (Python package installer)

Nvidia cuda is recommended for better performance

## Installation

```
pip install -r viewer/requirements.txt
```


