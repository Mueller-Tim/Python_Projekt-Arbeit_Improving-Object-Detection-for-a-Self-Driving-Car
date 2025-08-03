import os

# Paths to required files
YOLO_MODEL_PATH = 'yolo/v11_n.pt'  # Replace with your YOLO model path

# Paths for scalers and checkpoints
SCALER_SAVE_DIR = os.path.join('../nn', 'dist_net')  # Directory where scalers are saved
FEATURE_SCALER_PATH = os.path.join(SCALER_SAVE_DIR, 'feature_scaler.pkl')
TARGET_SCALER_PATH = os.path.join(SCALER_SAVE_DIR, 'target_scaler.pkl')

COMBINED_MODEL_PATH = os.path.join(SCALER_SAVE_DIR, 'dist_net.pth')  # Path to CombinedNet model

# Video source
VIDEO_SOURCE = 2  # Example device index for webcam

# Class Mapping: YOLO Class ID -> CombinedNet Class ID, Name, and Color
# Colors are in RGBA format for pyqtgraph and GUI consistency
YOLO_CLASS_MAPPING = {
    0: {
        'dist_net_class': 0,
        'name': 'blue_cone',
        'color': (0, 0, 255, 255)  # Blue
    },
    1: {
        'dist_net_class': 1,
        'name': 'yellow_cone',
        'color': (253, 218, 13, 255)  # Yellow
    },
    2: {
        'dist_net_class': 2,
        'name': 'orange_cone',
        'color': (255, 140, 0, 255) # Orange
    },
    3: {
        'dist_net_class': 3,
        'name': 'big_cone',
        'color': (255, 0, 0, 255) # Red
    },
    4: {
        'dist_net_class': 4,
        'name': 'down_cone',
        'color': (0, 0, 0, 255)  # Black
    },
}

# Reference Cones Configuration
REFERENCE_CONES = [
    {
        'x': 7.55,  # X-coordinate in meters
        'y': 4,  # Y-coordinate in meters
        'class_id': 1,  # yellow_cone
        'name': 'ref_yellow_cone',
        'color': (253, 218, 13, 255)
    },
    {
        'x': -7.95,  # X-coordinate in meters
        'y': 21,  # Y-coordinate in meters
        'class_id': 4,  # down_cone
        'name': 'ref_down_cone',
        'color': (0, 0, 0, 255)
    },
    {
        'x': 0,  # X-coordinate in meters
        'y': 16,  # Y-coordinate in meters
        'class_id': 2,  # orange_cone
        'name': 'ref_orange_cone',
        'color': (255, 140, 0, 255)
    },
    {
        'x': -7.95,  # X-coordinate in meters
        'y': 16,  # Y-coordinate in meters
        'class_id': 1,  # yellow_cone
        'name': 'ref_yellow_cone',
        'color': (253, 218, 13, 255)
    },
    {
        'x': 3.6,  # X-coordinate in meters
        'y': 22,  # Y-coordinate in meters
        'class_id': 1,  # yellow_cone
        'name': 'ref_yellow_cone',
        'color': (253, 218, 13, 255)
    },
    {
        'x': 3.6,  # X-coordinate in meters
        'y': 11,  # Y-coordinate in meters
        'class_id': 2,  # orange_cone
        'name': 'ref_orange_cone',
        'color': (255, 140, 0, 255)
    },
    {
        'x': 0,  # X-coordinate in meters
        'y': 9,  # Y-coordinate in meters
        'class_id': 0,  # blue_cone
        'name': 'ref_blue_cone',
        'color': (0, 0, 255, 255)
    },
    {
        'x': -3.97,  # X-coordinate in meters
        'y': 12,  # Y-coordinate in meters
        'class_id': 0,  # blue_cone
        'name': 'ref_blue_cone',
        'color': (0, 0, 255, 255)
    },
    {
        'x': 3.6,  # X-coordinate in meters
        'y': 7,  # Y-coordinate in meters
        'class_id': 4,  # down_cone
        'name': 'ref_down_cone',
        'color': (0, 0, 0, 255)
    },
    {
        'x': -7.95,  # X-coordinate in meters
        'y': 24,  # Y-coordinate in meters
        'class_id': 2,  # orange_cone
        'name': 'ref_orange_cone',
        'color': (255, 140, 0, 255)
    },
    {
        'x': 7.55,  # X-coordinate in meters
        'y': 10,  # Y-coordinate in meters
        'class_id': 2,  # orange_cone
        'name': 'ref_orange_cone',
        'color': (255, 140, 0, 255)
    },
    {
        'x': 0,  # X-coordinate in meters
        'y': 27,  # Y-coordinate in meters
        'class_id': 3,  # big_cone
        'name': 'ref_big_cone',
        'color': (255, 0, 0, 255)
    },
    {
        'x': -7.95,  # X-coordinate in meters
        'y': 12,  # Y-coordinate in meters
        'class_id': 2,  # orange_cone
        'name': 'ref_orange_cone',
        'color': (255, 140, 0, 255)
    },
    {
        'x': -3.97,  # X-coordinate in meters
        'y': 15,  # Y-coordinate in meters
        'class_id': 1,  # yellow_cone
        'name': 'ref_yellow_cone',
        'color': (253, 218, 13, 255)
    },
    {
        'x': 7.55,  # X-coordinate in meters
        'y': 13,  # Y-coordinate in meters
        'class_id': 3,  # down_cone
        'name': 'ref_big_cone',
        'color': (255, 0, 0, 255)
    },
    {
        'x': 0,  # X-coordinate in meters
        'y': 6,  # Y-coordinate in meters
        'class_id': 0,  # blue_cone
        'name': 'ref_blue_cone',
        'color': (0, 0, 255, 255)
    },
    {
        'x': -3.97,  # X-coordinate in meters
        'y': 8,  # Y-coordinate in meters
        'class_id': 1,  # yellow_cone
        'name': 'ref_yellow_cone',
        'color': (253, 218, 13, 255)
    },
]

SHOW_REFERENCE_CONES_BY_DEFAULT = True