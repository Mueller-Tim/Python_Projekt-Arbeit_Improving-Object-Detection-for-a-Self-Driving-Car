import os
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import joblib
import torch.optim as optim

from models import ScalarNet, ImageNet, CombinedNet
from utils import (set_seed, binarize_tensor, save_checkpoint, load_checkpoint, ConeDataset)


# ----------------------------
# Sample Weight Computation Function
# ----------------------------

def compute_sample_weights(dataset, thresholds, weights, scaler):
    """
    Computes weights for each sample based on multiple distance thresholds.

    Args:
        dataset (ConeDataset): The dataset subset.
        thresholds (list of float): List of distance thresholds in ascending order.
        weights (list of float): List of weights corresponding to each distance range.
                                Length should be len(thresholds) + 1.
        scaler (StandardScaler): The target scaler for inverse transformation.

    Returns:
        np.ndarray: Array of weights corresponding to each sample in the dataset.
    """
    if len(weights) != len(thresholds) + 1:
        raise ValueError("Number of weights must be one more than number of thresholds.")

    # Extract labels for the dataset
    scaled_labels = [dataset[i][2].numpy() for i in range(len(dataset))]
    unscaled_labels = scaler.inverse_transform(scaled_labels)
    distances = unscaled_labels[:, 0]

    # Initialize weights array
    weights_array = np.ones_like(distances) * weights[-1]  # Default weight for distances beyond last threshold

    # Assign weights based on thresholds
    for i, threshold in enumerate(thresholds):
        # Assign weights for distances <= threshold and > previous threshold
        if i == 0:
            lower_bound = -np.inf
        else:
            lower_bound = thresholds[i - 1]
        upper_bound = threshold
        mask = (distances > lower_bound) & (distances <= upper_bound)
        weights_array[mask] = weights[i]

    return weights_array


# ----------------------------
# Training Functions
# ----------------------------

def train_scalar_net(scalar_net, train_loader, val_loader, loss_fn, writer, num_epochs=10, device='cpu',
                     save_dir='checkpoints/scalar_net', early_stopping_patience=5):
    """
    Trains the ScalarNet model.

    Args:
        scalar_net (torch.nn.Module): The scalar network model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        loss_fn (torch.nn.Module): Loss function.
        writer (SummaryWriter): TensorBoard writer.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on.
        save_dir (str): Directory to save checkpoints.
        early_stopping_patience (int): Patience for early stopping.

    Returns:
        tuple: (train_losses, val_losses)
    """
    optimizer = optim.Adam(scalar_net.parameters(), lr=0.0005)
    scalar_net.to(device)
    scalar_net.train()
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    last_model_path = os.path.join(save_dir, 'last_scalar_net.pth')

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        scalar_net.train()
        for batch_idx, (scalar_input, _, labels, weights) in enumerate(train_loader):
            scalar_input = scalar_input.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            outputs = scalar_net(scalar_input)
            loss = loss_fn(outputs, labels)
            # Apply weights: multiply each sample's loss by its weight
            loss = loss * weights.unsqueeze(1)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * scalar_input.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        scalar_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for scalar_input, _, labels, _ in val_loader:
                scalar_input = scalar_input.to(device)
                labels = labels.to(device)

                outputs = scalar_net(scalar_input)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * scalar_input.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scalar_net.train()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # Log the loss to TensorBoard
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)

        # Check for improvement
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            print(f"Validation loss improved to {epoch_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Save the last model
    save_checkpoint(scalar_net, optimizer, epoch, epoch_val_loss, path=last_model_path)
    print(f"Last model saved at epoch {epoch + 1} with Val Loss: {epoch_val_loss:.4f}")

    return train_losses, val_losses


def train_combined_net(combined_net, train_loader, val_loader, loss_fn, writer, num_epochs=10, optimizer=None,
                       device='cpu', start_epoch=0, save_dir='checkpoints/combined_net', early_stopping_patience=5):
    """
    Trains the CombinedNet model.

    Args:
        combined_net (torch.nn.Module): The combined network model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        loss_fn (torch.nn.Module): Loss function.
        writer (SummaryWriter): TensorBoard writer.
        num_epochs (int): Number of training epochs.
        optimizer (torch.optim.Optimizer, optional): Optimizer. If None, a new Adam optimizer is created.
        device (torch.device): Device to train on.
        start_epoch (int): Starting epoch number (useful for resuming training).
        save_dir (str): Directory to save checkpoints.
        early_stopping_patience (int): Patience for early stopping.

    Returns:
        tuple: (train_losses, val_losses)
    """
    if optimizer is None:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, combined_net.parameters()), lr=0.001)
    combined_net.to(device)
    train_losses = []
    val_losses = []
    combined_net.train()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    last_model_path = os.path.join(save_dir, 'last_combined_net.pth')

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        running_loss = 0.0
        combined_net.train()
        for batch_idx, (scalar_input, image_input, labels, weights) in enumerate(train_loader):
            scalar_input = scalar_input.to(device)
            image_input = image_input.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            outputs = combined_net(scalar_input, image_input)
            loss = loss_fn(outputs, labels)
            # Apply weights: multiply each sample's loss by its weight
            loss = loss * weights.unsqueeze(1)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * scalar_input.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        combined_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for scalar_input, image_input, labels, _ in val_loader:
                scalar_input = scalar_input.to(device)
                image_input = image_input.to(device)
                labels = labels.to(device)

                outputs = combined_net(scalar_input, image_input)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * scalar_input.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        combined_net.train()

        print(f"Epoch [{epoch + 1}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # Log the losses to TensorBoard
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)

        # Check for improvement
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            print(f"Validation loss improved to {epoch_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Save the last model
    save_checkpoint(combined_net, optimizer, epoch, epoch_val_loss, path=last_model_path)
    print(f"Last model saved at epoch {epoch + 1} with Val Loss: {epoch_val_loss:.4f}")

    return train_losses, val_losses


# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    set_seed(42)

    # Paths to your images and labels directories
    image_dir_train = r'E:\PA\data\cone_dataset/cone_dataset_balanced_28_4/train_40/img_box'
    label_dir_train = r'E:\PA\data\cone_dataset/cone_dataset_balanced_28_4/train_40/ann_box'
    image_dir_val = r'E:\PA\data\cone_dataset/cone_dataset_balanced_28_4/val_40/img_box'
    label_dir_val = r'E:\PA\data\cone_dataset/cone_dataset_balanced_28_4/val_40/ann_box'

    print(f"Training Images Directory: {image_dir_train}")
    print(f"Number of training images: {len(os.listdir(image_dir_train))}")
    print(f"Training Labels Directory: {label_dir_train}")
    print(f"Number of training labels: {len(os.listdir(label_dir_train))}")

    print(f"Validation Images Directory: {image_dir_val}")
    print(f"Number of validation images: {len(os.listdir(image_dir_val))}")
    print(f"Validation Labels Directory: {label_dir_val}")
    print(f"Number of validation labels: {len(os.listdir(label_dir_val))}")

    # Training parameters
    num_epochs_scalar = 20
    num_epochs_combined_frozen = 10
    num_epochs_combined_unfrozen = 40
    num_additional_epochs = 5

    lr = 0.0005
    early_stop_scalarNet = 5
    early_stop_combinedNet_frozen = 5
    early_stop_combinedNet = 10

    resume_training = False  # Set to True to resume training from saved model

    # Define the distance thresholds in meters and corresponding weights
    # Example: [0-8] meters: weight=8.0, [8-16] meters: weight=4.0, >16 meters: weight=1.0
    DISTANCE_THRESHOLDS = [8.0, 16.0]  # Ascending order
    WEIGHTS = [8.0, 4.0, 1.0]  # Corresponding weights

    # Paths for scalers and checkpoints
    SCALER_SAVE_DIR = r'dist_net'  # Directory where scalers are saved
    os.makedirs(SCALER_SAVE_DIR, exist_ok=True)  # Create the directory if it doesn't exist
    checkpoint_path = os.path.join(SCALER_SAVE_DIR, 'dist_net.pth')

    # Initialize scalers
    feature_scaler_path = os.path.join(SCALER_SAVE_DIR, 'feature_scaler.pkl')
    target_scaler_path = os.path.join(SCALER_SAVE_DIR, 'target_scaler.pkl')

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Lambda(binarize_tensor)  # Use the named function
    ])

    if os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
        # Load existing scalers
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        print("Scalers loaded successfully.")
    else:
        # Fit scalers on the training dataset and save them
        print("Scalers not found. Fitting scalers on the training dataset...")
        temp_train_dataset = ConeDataset(
            image_dir=image_dir_train,
            label_dir=label_dir_train,
            transform=transform,
            feature_scaler=None,
            target_scaler=None,
            sample_weights=None,
            indices=None
        )

        # Extract scalar inputs and labels from the training data
        all_scalar_inputs = [sample[0].numpy() for sample in temp_train_dataset]
        all_labels = [sample[2].numpy() for sample in temp_train_dataset]

        # Initialize and fit scalers
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        feature_scaler.fit(all_scalar_inputs)
        target_scaler.fit(all_labels)

        # Save the fitted scalers
        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(target_scaler, target_scaler_path)
        print(f"Scalers fitted and saved to '{SCALER_SAVE_DIR}'.")

    # Create training and validation datasets without weights initially
    full_train_dataset = ConeDataset(
        image_dir=image_dir_train,
        label_dir=label_dir_train,
        transform=transform,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sample_weights=None,
        indices=None
    )

    val_dataset = ConeDataset(
        image_dir=image_dir_val,
        label_dir=label_dir_val,
        transform=transform,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sample_weights=None,
        indices=None
    )

    # Compute min and max distance on the original (unscaled) training data
    print("Computing min and max distances from the original training data...")
    all_distances = [sample[2][0].item() for sample in full_train_dataset]
    min_distance = min(all_distances)
    max_distance = max(all_distances)
    print(f"Min Distance: {min_distance}, Max Distance: {max_distance}")

    # Create data loaders
    batch_size = 512
    num_workers = min(os.cpu_count(), 8)  # Adjust based on your system

    # -----------------------------------------
    # Create Threshold-Based Balanced Sampler for Training Data
    # -----------------------------------------

    # Get unscaled distances for all samples in the training dataset
    unscaled_labels = [sample[2].numpy() for sample in full_train_dataset]
    unscaled_labels = target_scaler.inverse_transform(unscaled_labels)
    train_distances = unscaled_labels[:, 0]

    # Define distance bins
    bin_width = 2.0
    bins = np.arange(min_distance, max_distance + bin_width, bin_width)

    # Assign each sample to a bin
    bin_indices = np.digitize(train_distances, bins, right=False)

    # Map bin indices to sample indices
    bin_to_indices = defaultdict(list)
    for idx, bin_idx in enumerate(bin_indices):
        bin_to_indices[bin_idx].append(idx)

    # Separate bins into below and above the threshold
    indices_below = [i for i, d in enumerate(train_distances) if d <= DISTANCE_THRESHOLDS[0]]
    indices_mid = [i for i, d in enumerate(train_distances) if DISTANCE_THRESHOLDS[0] < d <= DISTANCE_THRESHOLDS[1]]
    indices_above = [i for i, d in enumerate(train_distances) if d > DISTANCE_THRESHOLDS[1]]

    # Determine the number of samples for each group based on desired proportions
    total_samples = len(full_train_dataset)
    desired_below = int(0.6 * total_samples)
    desired_mid = int(0.3 * total_samples)
    desired_above = total_samples - desired_below - desired_mid

    # Current counts
    current_below = len(indices_below)
    current_mid = len(indices_mid)
    current_above = len(indices_above)

    # Calculate sampling ratios
    ratio_below = desired_below / current_below if current_below > 0 else 1
    ratio_mid = desired_mid / current_mid if current_mid > 0 else 1
    ratio_above = desired_above / current_above if current_above > 0 else 1

    # Oversample or undersample as needed
    if ratio_below > 1:
        # Oversample below threshold
        oversampled_below = np.random.choice(indices_below, size=desired_below, replace=True)
    else:
        # Undersample below threshold
        oversampled_below = np.random.choice(indices_below, size=desired_below, replace=False)

    if ratio_mid > 1:
        # Oversample mid threshold
        oversampled_mid = np.random.choice(indices_mid, size=desired_mid, replace=True)
    else:
        # Undersample mid threshold
        oversampled_mid = np.random.choice(indices_mid, size=desired_mid, replace=False)

    if ratio_above > 1:
        # Oversample above threshold
        oversampled_above = np.random.choice(indices_above, size=desired_above, replace=True)
    else:
        # Undersample above threshold
        oversampled_above = np.random.choice(indices_above, size=desired_above, replace=False)

    # Combine the indices
    balanced_indices = np.concatenate([oversampled_below, oversampled_mid, oversampled_above])

    # Shuffle the balanced indices
    np.random.shuffle(balanced_indices)

    # Create the balanced training dataset with only the balanced indices
    balanced_train_dataset = ConeDataset(
        image_dir=image_dir_train,
        label_dir=label_dir_train,
        transform=transform,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sample_weights=None,  # Initialize without weights
        indices=balanced_indices
    )

    # Compute sample weights based on the multiple thresholds
    sample_weights = compute_sample_weights(
        balanced_train_dataset,
        thresholds=DISTANCE_THRESHOLDS,
        weights=WEIGHTS,
        scaler=target_scaler
    )

    # Assign the computed sample weights to the dataset
    balanced_train_dataset.sample_weights = sample_weights

    # Create a SubsetRandomSampler with the range of balanced dataset indices
    sampler = SubsetRandomSampler(range(len(balanced_train_dataset)))

    # Create DataLoader with the balanced sampler
    train_loader = DataLoader(
        balanced_train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    # For validation, use the standard DataLoader without balancing
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ----------------------------
    # Parameter to decide training mode
    # ----------------------------

    # Define loss function
    loss_fn = torch.nn.MSELoss(reduction='mean')  # We handle reduction manually

    if not resume_training:
        # ----------------------------
        # Start new training with freezing layers
        # ----------------------------

        # Initialize the SummaryWriter
        writer_scalar = SummaryWriter('runs/scalar_net_balanced')

        # Step 1: Initialize the scalar network
        scalar_net = ScalarNet().to(device)

        # Step 3: Train the scalar network with validation
        scalar_save_dir = os.path.join(SCALER_SAVE_DIR, 'scalar_net')
        scalar_train_losses, scalar_val_losses = train_scalar_net(
            scalar_net, train_loader, val_loader, loss_fn, writer_scalar,
            num_epochs=num_epochs_scalar, device=device,
            save_dir=scalar_save_dir, early_stopping_patience=early_stop_scalarNet)

        # Close the writer for scalar net
        writer_scalar.close()

        # Step 4: Freeze the scalar network's parameters
        for param in scalar_net.parameters():
            param.requires_grad = False

        # Step 5: Create the image network and combined network
        image_net = ImageNet(input_size=(50, 50)).to(device)  # Ensure input size matches
        combined_net = CombinedNet(scalar_net, image_net).to(device)

        # Initialize the SummaryWriter for combined net with frozen scalar net
        writer_combined_frozen = SummaryWriter('runs/combined_net_frozen_scalar_balanced')

        # Step 6: Train the combined network with frozen scalar network
        combined_save_dir_frozen = os.path.join(SCALER_SAVE_DIR, 'combined_net_frozen')
        combined_train_losses, combined_val_losses = train_combined_net(
            combined_net, train_loader, val_loader, loss_fn, writer_combined_frozen,
            num_epochs=num_epochs_combined_frozen,
            device=device, save_dir=combined_save_dir_frozen, early_stopping_patience=early_stop_combinedNet_frozen
        )

        # Close the writer for combined net with frozen scalar net
        writer_combined_frozen.close()

        # Step 8: Unfreeze the scalar network's parameters
        for param in scalar_net.parameters():
            param.requires_grad = True

        # Step 9: Create a new optimizer including all parameters
        optimizer_full = optim.Adam(combined_net.parameters(), lr=lr)  # Smaller learning rate

        # Initialize the SummaryWriter for combined net with unfrozen scalar net
        writer_combined_unfrozen = SummaryWriter('runs/combined_net_unfrozen_scalar_balanced')

        # Step 10: Continue training the combined network with unfrozen scalar network
        combined_save_dir_unfrozen = os.path.join(SCALER_SAVE_DIR, 'combined_net_unfrozen')
        combined_train_losses_unfrozen, combined_val_losses_unfrozen = train_combined_net(
            combined_net, train_loader, val_loader, loss_fn, writer_combined_unfrozen,
            num_epochs=num_epochs_combined_unfrozen,
            optimizer=optimizer_full, device=device, save_dir=combined_save_dir_unfrozen, early_stopping_patience=early_stop_combinedNet
        )

        # Close the writer for combined net with unfrozen scalar net
        writer_combined_unfrozen.close()

        # Step 12: Save the trained model and optimizer state as a checkpoint
        total_epochs = num_epochs_scalar + num_epochs_combined_frozen + num_epochs_combined_unfrozen
        final_loss = combined_val_losses_unfrozen[-1]
        save_checkpoint(combined_net, optimizer_full, epoch=total_epochs, loss=final_loss, path=checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    else:
        # ----------------------------
        # Continue training from saved checkpoint
        # ----------------------------

        # Step 1: Initialize the combined network and optimizer
        scalar_net = ScalarNet().to(device)
        image_net = ImageNet(input_size=(50, 50)).to(device)
        combined_net = CombinedNet(scalar_net, image_net).to(device)

        optimizer_full = optim.Adam(combined_net.parameters(), lr=0.000005)  # Adjust learning rate if needed

        # Step 2: Load the model and optimizer state
        checkpoint_path = os.path.join(SCALER_SAVE_DIR, 'dist_net.pth')  # Ensure the correct path
        combined_net, optimizer_full, start_epoch, loss = load_checkpoint(combined_net, optimizer_full, checkpoint_path,
                                                                          device=device)
        print(f"Resuming training from epoch {start_epoch + 1}")

        # Step 3: Initialize the SummaryWriter for continued training
        writer_continued = SummaryWriter('runs/dist_net_continued_training_balanced')

        # Step 5: Continue training
        combined_save_dir_continued = os.path.join(SCALER_SAVE_DIR, 'combined_net_continued')
        combined_train_losses_continued, combined_val_losses_continued = train_combined_net(
            combined_net, train_loader, val_loader, loss_fn,
            writer_continued, num_epochs=num_additional_epochs, optimizer=optimizer_full, device=device,
            save_dir=combined_save_dir_continued, early_stopping_patience=5
        )

        # Close the writer for continued training
        writer_continued.close()

        # Step 7: Save the model after continued training
        total_epochs = start_epoch + num_additional_epochs
        final_loss = combined_val_losses_continued[-1]
        save_checkpoint(combined_net, optimizer_full, epoch=total_epochs, loss=final_loss, path=checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path} after continued training")
