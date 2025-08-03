import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# ----------------------------
# Utility Functions
# ----------------------------

def set_seed(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binarize_tensor(x):
    """
    Binarizes a tensor: values greater than 0 are set to 1, others to 0.
    """
    return (x > 0).float()


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The current loss value.
        path (str): The file path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device='cpu'):
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        path (str): The file path of the checkpoint.
        device (torch.device): The device to map the checkpoint.

    Returns:
        tuple: (model, optimizer, start_epoch, loss)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file '{path}' not found.")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {path}, starting from epoch {start_epoch}")
    return model, optimizer, start_epoch, loss


# ----------------------------
# Dataset Definition
# ----------------------------

class ConeDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, feature_scaler=None, target_scaler=None,
                 sample_weights=None, indices=None):
        """
        Custom Dataset for loading cone images and labels with optional scaling and sample weights.

        Args:
            image_dir (str): Directory containing image files.
            label_dir (str): Directory containing label files.
            transform (callable, optional): Transform to apply to images.
            feature_scaler (StandardScaler, optional): Scaler for scalar inputs.
            target_scaler (StandardScaler, optional): Scaler for target labels.
            sample_weights (np.ndarray, optional): Array of sample weights.
            indices (list or np.ndarray, optional): List of indices to include in the dataset.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.sample_weights = sample_weights  # Should be a NumPy array or None
        self.all_image_files = sorted(os.listdir(image_dir))
        self.all_label_files = sorted(os.listdir(label_dir))
        assert len(self.all_image_files) == len(self.all_label_files), "Number of images and labels must be the same."

        # If indices are provided, subset the image and label files
        if indices is not None:
            self.image_files = [self.all_image_files[i] for i in indices]
            self.label_files = [self.all_label_files[i] for i in indices]
            if self.sample_weights is not None and sample_weights is not None:
                self.sample_weights = self.sample_weights[indices]
        else:
            self.image_files = self.all_image_files
            self.label_files = self.all_label_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            # Normalize to 0 and 1
            image = (image > 0).float()

        # Load label
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        with open(label_path, 'r') as f:
            label_line = f.readline().strip().split()
            # Ensure there are at least 9 elements (7 for scalar_input and 2 for label)
            if len(label_line) < 9:
                raise ValueError(f"Label file {label_path} has insufficient data.")
            scalar_input = np.array([float(x) for x in label_line[0:7]], dtype=np.float32)
            label = np.array([float(label_line[7]), float(label_line[8])], dtype=np.float32)

        # Apply scalers if provided
        if self.feature_scaler:
            scalar_input = self.feature_scaler.transform([scalar_input])[0]
        if self.target_scaler:
            label = self.target_scaler.transform([label])[0]

        scalar_input = torch.tensor(scalar_input, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # Get weight
        if self.sample_weights is not None:
            weight = self.sample_weights[idx]
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)

        return scalar_input, image, label, weight
