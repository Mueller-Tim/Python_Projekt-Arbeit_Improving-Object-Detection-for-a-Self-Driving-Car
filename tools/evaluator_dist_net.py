import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import joblib
import sys


# Define a function to binarize the tensor
def binarize_tensor(x):
    return (x > 0).float()


# Define the dataset class
class ConeDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, feature_scaler=None):
        """
        Initializes the dataset by loading image and label file paths.

        Parameters:
        - image_dir (str): Path to the directory containing images.
        - label_dir (str): Path to the directory containing label files.
        - transform (callable, optional): Optional transform to be applied on an image.
        - feature_scaler (sklearn.preprocessing.StandardScaler, optional): Scaler to apply to scalar inputs.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.feature_scaler = feature_scaler
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        assert len(self.image_files) == len(self.label_files), "Number of images and labels must be the same."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and preprocess image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            image = (image > 0).float()

        # Load label
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        with open(label_path, 'r') as f:
            label_line = f.readline().strip().split()
            # Extract scalar inputs
            scalar_input = [float(x) for x in label_line[0:7]]
            # Extract labels (distance and angle)
            label = [float(label_line[7]), float(label_line[8])]

        # Apply feature scaler if provided
        if self.feature_scaler:
            scalar_input = self.feature_scaler.transform([scalar_input])[0]

        scalar_input = torch.tensor(scalar_input, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return scalar_input, image, label


# Define the scalar network
class ScalarNet(nn.Module):
    def __init__(self):
        super(ScalarNet, self).__init__()
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Define the image network
class ImageNet(nn.Module):
    def __init__(self, input_size=(50, 50)):
        super(ImageNet, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Grayscale image
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)  # 50x50 -> 25x25
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 25x25 -> 12x12
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 12x12 -> 6x6
        self.bn3 = nn.BatchNorm2d(64)

        self.flatten_size = 64 * 6 * 6  # 64 channels, 6x6 spatial dimensions
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, 9)  # Output reduced to 9 features

        self.dropout = nn.Dropout(0.3)  # Optional: Add dropout for regularization

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 16, 50, 50]
        x = self.pool(x)                        # [B, 16, 25, 25]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 32, 25, 25]
        x = self.pool(x)                        # [B, 32, 12, 12]
        x = self.relu(self.bn3(self.conv3(x)))  # [B, 64, 12, 12]
        x = self.pool(x)                        # [B, 64, 6, 6]
        x = self.view_flatten(x)                # [B, 64*6*6] = [B, 2304]
        x = self.relu(self.fc1(x))              # [B, 64]
        x = self.dropout(x)                     # [B, 64]
        x = self.fc2(x)                         # [B, 9]
        return x

    def view_flatten(self, x):
        return x.view(x.size(0), -1)


# Define the combined network
class CombinedNet(nn.Module):
    def __init__(self, scalar_net, image_net):
        super(CombinedNet, self).__init__()
        self.scalar_net = scalar_net
        self.image_net = image_net
        self.fc_combined = nn.Linear(9 + 2, 32)  # Combined features: 9 (image) + 2 (scalar) = 11
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, 2)  # Output: distance and angle

    def forward(self, scalar_input, image_input):
        scalar_output = self.scalar_net(scalar_input)
        image_output = self.image_net(image_input)
        combined = torch.cat((scalar_output, image_output), dim=1)  # [B, 11]
        x = self.relu(self.fc_combined(combined))                   # [B, 32]
        x = self.fc_out(x)                                          # [B, 2]
        return x


# Function to load the model for inference
def load_model_for_inference(model, checkpoint_path, device='cpu'):
    """
    Loads the model weights from a checkpoint file.

    Parameters:
    - model (nn.Module): The model to load weights into.
    - checkpoint_path (str): Path to the checkpoint file.
    - device (torch.device): Device to map the model weights to.

    Returns:
    - model (nn.Module): The model with loaded weights.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found at '{checkpoint_path}'. Exiting.")
        sys.exit(1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded for inference from {checkpoint_path}")
    return model


# Function to validate and save predictions to CSV
def validate_and_save_to_csv(model, test_loader, output_csv_path, target_scaler, scalar_scaler, device='cpu'):
    """
    Runs inference on the test dataset and saves inverse-transformed scalar inputs,
    ground truth labels, and predictions to a CSV file.

    Parameters:
    - model (nn.Module): The trained model for inference.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - output_csv_path (str): Path to save the CSV file with predictions.
    - target_scaler (sklearn.preprocessing.StandardScaler): Scaler to inverse transform predictions.
    - scalar_scaler (sklearn.preprocessing.StandardScaler): Scaler to inverse transform scalar inputs.
    - device (torch.device or str): Device to perform computations on ('cpu' or 'cuda').
    """
    model.eval()
    combined_data = []

    with torch.no_grad():
        for batch_idx, (scalar_input, image_input, labels) in enumerate(test_loader):
            # Move inputs to the specified device
            scalar_input = scalar_input.to(device)
            image_input = image_input.to(device)

            # Perform inference
            outputs = model(scalar_input, image_input).cpu().numpy()

            # Inverse transform predictions to original scale
            outputs_original = target_scaler.inverse_transform(outputs)

            # Convert scalar inputs to NumPy arrays and move to CPU
            scalar_input_np = scalar_input.cpu().numpy()

            # Inverse transform scalar inputs to original scale
            scalar_input_original = scalar_scaler.inverse_transform(scalar_input_np)

            # Convert labels to NumPy arrays
            labels_np = labels.cpu().numpy()

            # Collect inverse-transformed scalar inputs, ground truths, and predictions
            for scalar_feat, label, output in zip(scalar_input_original, labels_np, outputs_original):
                # Create a dictionary for scalar features with specific names
                scalar_dict = {
                    'class': scalar_feat[0],
                    'x_center': scalar_feat[1],
                    'y_center': scalar_feat[2],
                    'width': scalar_feat[3],
                    'height': scalar_feat[4],
                    'roll': scalar_feat[5],
                    'pitch': scalar_feat[6]
                }

                # Create dictionaries for ground truth and predictions
                ground_truth_dict = {
                    'Actual_Distance': label[0],
                    'Actual_Angle': label[1]
                }
                prediction_dict = {
                    'Predicted_Distance': output[0],
                    'Predicted_Angle': output[1]
                }

                # Combine all dictionaries into one
                combined_dict = {**scalar_dict, **ground_truth_dict, **prediction_dict}
                combined_data.append(combined_dict)

            # Optional: Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches")

    # Create a DataFrame to store the combined data
    results_df = pd.DataFrame(combined_data)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


# Main execution for inference
if __name__ == "__main__":
    # Paths for scalers and checkpoints
    SCALER_SAVE_DIR = r'../nn/dist_net'  # Directory where scalers are saved
    feature_scaler_path = os.path.join(SCALER_SAVE_DIR, 'feature_scaler.pkl')
    target_scaler_path = os.path.join(SCALER_SAVE_DIR, 'target_scaler.pkl')
    checkpoint_path = os.path.join(SCALER_SAVE_DIR, 'dist_net.pth')

    # Load scalers
    try:
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        print("Scalers loaded successfully.")
    except FileNotFoundError:
        print(f"Scaler files not found in '{SCALER_SAVE_DIR}'. Please ensure scalers are saved during training.")
        sys.exit(1)

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the networks
    scalar_net = ScalarNet().to(device)
    image_net = ImageNet(input_size=(50, 50)).to(device)
    combined_net = CombinedNet(scalar_net, image_net).to(device)

    # Load the trained model from checkpoint
    combined_net = load_model_for_inference(combined_net, checkpoint_path, device=device)

    # Define transformations for test dataset
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Lambda(binarize_tensor)
    ])

    # Test dataset directory paths
    test_image_dir = r'E:\PA\data\test_validate_sort\cone_dataset\val_40\img_box'  # Replace with your test images directory
    test_label_dir = r'E:\PA\data\test_validate_sort\cone_dataset\val_40\ann_box'  # Replace with your test labels directory

    # Create test dataset and loader with feature_scaler
    test_dataset = ConeDataset(
        image_dir=test_image_dir,
        label_dir=test_label_dir,
        transform=transform,
        feature_scaler=feature_scaler  # Apply feature scaler to scalar inputs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Run validation and save predictions to CSV
    output_csv_path = '../nn/dist_net/test_predictions.csv'
    validate_and_save_to_csv(
        combined_net,
        test_loader,
        output_csv_path,
        target_scaler,
        feature_scaler,
        device=device
    )
