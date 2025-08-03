import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Model Definitions
# ----------------------------

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
