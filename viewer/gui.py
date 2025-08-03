import queue
import sys
import logging
import cv2
import torch
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from threads import FrameCapture, FrameProcessor
from utils import load_combined_model
from config import (
    YOLO_MODEL_PATH, COMBINED_MODEL_PATH, FEATURE_SCALER_PATH, TARGET_SCALER_PATH,
    VIDEO_SOURCE, YOLO_CLASS_MAPPING, REFERENCE_CONES, SHOW_REFERENCE_CONES_BY_DEFAULT
)
from ultralytics import YOLO

class AspectRatioViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super(AspectRatioViewBox, self).__init__(*args, **kwargs)
        self.setAspectLocked(True)  # Lock the aspect ratio


class VideoProcessingApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_variables()
        self.init_ui()
        self.init_processing()

    def init_variables(self):
        # Variables for controlling display features
        self.show_bounding_boxes = True
        self.show_bounding_box_labels = True
        self.show_prediction_text = True
        self.bounding_box_text_size = 0.5  # Default size
        self.show_lines_between_dots = True

        self.show_reference_cones = SHOW_REFERENCE_CONES_BY_DEFAULT

        # **New Variables for Sizes**
        self.detected_cone_size = 10  # Default size for detected cones
        self.reference_cone_size = 10  # Default size for reference cones

        # Initialize pitch and roll
        self.pitch = 11  # Default value
        self.roll = 180  # Default value

        # **Initialize latest detections for export**
        self.latest_detections_for_export = []

    def init_ui(self):
        self.setWindowTitle("Video Processing Application")
        self.layout = QtWidgets.QHBoxLayout(self)

        # Left side: Video and Metrics (70%)
        self.left_layout = QtWidgets.QVBoxLayout()
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        self.left_layout.setSpacing(10)

        # Video display using QLabel
        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.left_layout.addWidget(self.video_label)

        # Metrics display (under the video)
        self.metrics_label = QtWidgets.QLabel(self)
        self.metrics_label.setText("YOLO FPS: 0.00 | CombinedNet FPS: 0.00")
        self.metrics_label.setAlignment(QtCore.Qt.AlignCenter)
        self.left_layout.addWidget(self.metrics_label)

        # Right side: Plot and Controls (30%)
        right_layout = QtWidgets.QVBoxLayout()

        # Plot setup
        pg.setConfigOption('background', 'w')  # White background
        pg.setConfigOption('foreground', 'k')  # Black foreground

        self.plot_widget = pg.PlotWidget(viewBox=AspectRatioViewBox(), title="Detected Objects (Ground Coordinates)")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Y (m)')
        self.plot_widget.setLabel('bottom', 'X (m)')
        self.scatter = pg.ScatterPlotItem(size=self.detected_cone_size, pen=pg.mkPen(None),
                                          brush=pg.mkBrush(255, 255, 0, 120))  # Detected cones
        self.plot_widget.addItem(self.scatter)

        # Define class colors based on YOLO_CLASS_MAPPING
        self.class_colors = {
            class_id: mapping['color']
            for class_id, mapping in YOLO_CLASS_MAPPING.items()
        }

        # Initialize lines for connections
        self.connection_plot = pg.PlotDataItem(pen=pg.mkPen(color=(0, 0, 0), width=1))  # Black lines
        self.plot_widget.addItem(self.connection_plot)

        # Scatter for Reference Cones
        self.ref_cones_scatter = pg.ScatterPlotItem(size=self.reference_cone_size, pen=pg.mkPen(None),
                                                    brush=pg.mkBrush(255, 255, 255, 120), symbol='x')  # Reference cones
        self.plot_widget.addItem(self.ref_cones_scatter)

        self.plot_widget.setXRange(-15, 15)
        self.plot_widget.setYRange(0, 25)

        # Add PNG Image at (0, 0) with proper centering
        image_item = pg.ImageItem()
        image_data = cv2.imread('resources/car.png', cv2.IMREAD_UNCHANGED)  # Load PNG with transparency
        if image_data is not None:
            image_item.setImage(image_data)
        else:
            logging.warning("car.png not found or failed to load.")

        scale_factor = 0.002
        if image_data is not None:
            height, width = image_data.shape[:2]
            height = height * scale_factor
            width = width * scale_factor

            image_item.setPos(-width / 2, height / 2)
            image_item.setScale(scale_factor)
            image_item.setRotation(-90)
            self.plot_widget.addItem(image_item)

        right_layout.addWidget(self.plot_widget)

        # Controls Section
        controls_group = QtWidgets.QGroupBox("Display Options")
        controls_layout = QtWidgets.QVBoxLayout()

        # Bounding box enable/disable
        self.bbox_checkbox = QtWidgets.QCheckBox("Show Bounding Boxes")
        self.bbox_checkbox.setChecked(self.show_bounding_boxes)
        self.bbox_checkbox.stateChanged.connect(self.on_bbox_checkbox_changed)
        controls_layout.addWidget(self.bbox_checkbox)

        # Bounding box labels (class/confidence)
        self.bbox_labels_checkbox = QtWidgets.QCheckBox("Show Bounding Box Labels")
        self.bbox_labels_checkbox.setChecked(self.show_bounding_box_labels)
        self.bbox_labels_checkbox.stateChanged.connect(self.on_bbox_labels_checkbox_changed)
        controls_layout.addWidget(self.bbox_labels_checkbox)

        # Prediction text (the text of predictions on frame)
        self.prediction_text_checkbox = QtWidgets.QCheckBox("Show Prediction Text")
        self.prediction_text_checkbox.setChecked(self.show_prediction_text)
        self.prediction_text_checkbox.stateChanged.connect(self.on_prediction_text_checkbox_changed)
        controls_layout.addWidget(self.prediction_text_checkbox)

        # Bounding box text size
        text_size_label = QtWidgets.QLabel("Bounding Box Text Size:")
        controls_layout.addWidget(text_size_label)

        # Use QDoubleSpinBox instead of QSpinBox
        self.text_size_spinbox = QtWidgets.QDoubleSpinBox()
        self.text_size_spinbox.setRange(0.1, 2.0)
        self.text_size_spinbox.setSingleStep(0.1)
        self.text_size_spinbox.setValue(self.bounding_box_text_size)
        self.text_size_spinbox.valueChanged.connect(self.on_text_size_changed)
        controls_layout.addWidget(self.text_size_spinbox)

        # Lines between dots on plot
        self.lines_checkbox = QtWidgets.QCheckBox("Show Lines Between Dots")
        self.lines_checkbox.setChecked(self.show_lines_between_dots)
        self.lines_checkbox.stateChanged.connect(self.on_lines_checkbox_changed)
        controls_layout.addWidget(self.lines_checkbox)

        # **Add Size Controls for Plotted Elements**
        size_controls_group = QtWidgets.QGroupBox("Plot Sizes")
        size_controls_layout = QtWidgets.QVBoxLayout()

        # Detected Cones Size
        detected_size_label = QtWidgets.QLabel("Detected Cones Size:")
        size_controls_layout.addWidget(detected_size_label)

        self.detected_size_spinbox = QtWidgets.QSpinBox()
        self.detected_size_spinbox.setRange(5, 20)
        self.detected_size_spinbox.setSingleStep(1)
        self.detected_size_spinbox.setValue(self.detected_cone_size)
        self.detected_size_spinbox.valueChanged.connect(self.on_detected_size_changed)
        size_controls_layout.addWidget(self.detected_size_spinbox)

        # Reference Cones Size
        reference_size_label = QtWidgets.QLabel("Reference Cones Size:")
        size_controls_layout.addWidget(reference_size_label)

        self.reference_size_spinbox = QtWidgets.QSpinBox()
        self.reference_size_spinbox.setRange(5, 20)
        self.reference_size_spinbox.setSingleStep(1)
        self.reference_size_spinbox.setValue(self.reference_cone_size)
        self.reference_size_spinbox.valueChanged.connect(self.on_reference_size_changed)
        size_controls_layout.addWidget(self.reference_size_spinbox)

        size_controls_group.setLayout(size_controls_layout)
        controls_layout.addWidget(size_controls_group)

        # **Add Reference Cones Toggle**
        self.ref_cones_checkbox = QtWidgets.QCheckBox("Show Reference Cones")
        self.ref_cones_checkbox.setChecked(self.show_reference_cones)
        self.ref_cones_checkbox.stateChanged.connect(self.on_ref_cones_checkbox_changed)
        controls_layout.addWidget(self.ref_cones_checkbox)

        # **Add Pitch and Roll Controls**
        pitch_label = QtWidgets.QLabel("Pitch (degrees):")
        controls_layout.addWidget(pitch_label)

        self.pitch_spinbox = QtWidgets.QDoubleSpinBox()
        self.pitch_spinbox.setRange(9.0, 13.0)
        self.pitch_spinbox.setSingleStep(0.1)
        self.pitch_spinbox.setValue(self.pitch)
        self.pitch_spinbox.valueChanged.connect(self.on_pitch_changed)
        controls_layout.addWidget(self.pitch_spinbox)

        roll_label = QtWidgets.QLabel("Roll (degrees):")
        controls_layout.addWidget(roll_label)

        self.roll_spinbox = QtWidgets.QDoubleSpinBox()
        self.roll_spinbox.setRange(175.0, 185.0)
        self.roll_spinbox.setSingleStep(0.1)
        self.roll_spinbox.setValue(self.roll)
        self.roll_spinbox.valueChanged.connect(self.on_roll_changed)
        controls_layout.addWidget(self.roll_spinbox)

        # **Add Export Button**
        export_button = QtWidgets.QPushButton("Export Current Frame Data")
        export_button.clicked.connect(self.export_current_frame_data)
        controls_layout.addWidget(export_button)

        controls_group.setLayout(controls_layout)
        right_layout.addWidget(controls_group)

        # **Add layouts to the main layout with stretch factors**
        self.layout.addLayout(self.left_layout, 7)  # 70%
        self.layout.addLayout(right_layout, 3)  # 30%

        self.setLayout(self.layout)
        self.showMaximized()  # Start maximized for better visibility

    def init_processing(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # ----------------------------
        # Device Configuration
        # ----------------------------
        if torch.backends.mps.is_available():
            device = 'mps'
            logging.info("MPS (Metal Performance Shaders) is available. Using GPU.")
        elif torch.cuda.is_available():
            device = 'cuda'
            logging.info("CUDA is available. Using GPU.")
        else:
            device = 'cpu'
            logging.info("No GPU found. Using CPU.")

        # ----------------------------
        # Load Models and Scalers
        # ----------------------------
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
            logging.info(f"YOLO model '{YOLO_MODEL_PATH}' loaded successfully on {device}.")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error loading YOLO model: {e}")
            sys.exit(1)

        try:
            combined_model, feature_scaler, target_scaler = load_combined_model(
                COMBINED_MODEL_PATH, device, FEATURE_SCALER_PATH, TARGET_SCALER_PATH
            )
            logging.info(f"CombinedNet model '{COMBINED_MODEL_PATH}' loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading CombinedNet model: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error loading CombinedNet model: {e}")
            sys.exit(1)

        # ----------------------------
        # Initialize Queues and Threads
        # ----------------------------
        self.frame_queue = queue.Queue(maxsize=5)  # Buffer up to 5 frames
        self.display_queue = queue.Queue(maxsize=1)  # Latest processed frame
        self.plot_queue = queue.Queue(maxsize=10)  # Latest detections

        try:
            self.frame_capture = FrameCapture(VIDEO_SOURCE, self.frame_queue)
        except ValueError as ve:
            logging.error(ve)
            QtWidgets.QMessageBox.critical(self, "Error", str(ve))
            sys.exit(1)

        # Pass the display parameters to FrameProcessor so it knows what to draw
        self.frame_processor = FrameProcessor(
            self.frame_queue,
            self.display_queue,
            self.plot_queue,
            yolo_model,
            combined_model,
            feature_scaler,
            target_scaler,
            device,
            class_mapping=YOLO_CLASS_MAPPING,
            show_bboxes=self.show_bounding_boxes,
            show_bbox_labels=self.show_bounding_box_labels,
            show_prediction_text=self.show_prediction_text,
            bbox_text_size=self.bounding_box_text_size,
            initial_pitch=self.pitch,  # Pass initial pitch
            initial_roll=self.roll     # Pass initial roll
        )

        self.frame_capture.start()
        self.frame_processor.start()

        logging.info("Starting video processing.")

        # Set up a timer to update the GUI
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(30)  # Update every 30 ms (~33 FPS)

        # **Set initial sizes for scatter plots**
        self.scatter.setSize(self.detected_cone_size)
        self.ref_cones_scatter.setSize(self.reference_cone_size)

    def update_gui(self):
        # Update video frame
        try:
            processed_frame = self.display_queue.get_nowait()
            # Convert the frame to QImage
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            qt_pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(
                self.video_label.width(), self.video_label.height(), QtCore.Qt.KeepAspectRatio)
            self.video_label.setPixmap(qt_pixmap)

            # Update metrics
            yolo_fps = self.frame_processor.yolo_fps
            combined_fps = self.frame_processor.combined_fps

            # Compute averages if we have processed any frames with cones
            fp = self.frame_processor
            if fp.yolo_time_count > 0:
                avg_yolo_time = fp.yolo_time_sum / fp.yolo_time_count

                avg_dist_net_time_all_cones = (
                            fp.dist_net_time_sum_all_cones / fp.dist_net_time_count_frames) if fp.dist_net_time_count_frames > 0 else 0
                avg_prep_time_all_cones = (
                            fp.image_prep_time_sum_all_cones / fp.dist_net_time_count_frames) if fp.dist_net_time_count_frames > 0 else 0
                avg_dist_net_plus_prep_all_cones = (
                            fp.dist_net_plus_prep_time_sum_all_cones / fp.dist_net_time_count_frames) if fp.dist_net_time_count_frames > 0 else 0

                avg_dist_net_time_per_cone = (
                            fp.dist_net_time_sum_per_cone / fp.dist_net_time_count_per_cone) if fp.dist_net_time_count_per_cone > 0 else 0
                avg_prep_time_per_cone = (
                            fp.image_prep_time_sum_per_cone / fp.dist_net_time_count_per_cone) if fp.dist_net_time_count_per_cone > 0 else 0
                avg_dist_net_plus_prep_per_cone = (
                            fp.dist_net_plus_prep_time_sum_per_cone / fp.dist_net_time_count_per_cone) if fp.dist_net_time_count_per_cone > 0 else 0
            else:
                # No frames with cones processed yet, set all to 0
                avg_yolo_time = 0
                avg_dist_net_time_all_cones = 0
                avg_prep_time_all_cones = 0
                avg_dist_net_plus_prep_all_cones = 0
                avg_dist_net_time_per_cone = 0
                avg_prep_time_per_cone = 0
                avg_dist_net_plus_prep_per_cone = 0

            # Update metrics label
            self.metrics_label.setText(
                f"YOLO FPS: {yolo_fps:.2f} | CombinedNet FPS: {combined_fps:.2f}\n"
                f"Avg YOLO Time (per frame w/ cones): {avg_yolo_time:.2f} ms\n"
                f"Avg DistNet Time (all cones/frame): {avg_dist_net_time_all_cones:.2f} ms | (per cone): {avg_dist_net_time_per_cone:.2f} ms\n"
                f"Avg Prep Time (all cones/frame): {avg_prep_time_all_cones:.2f} ms | (per cone): {avg_prep_time_per_cone:.2f} ms\n"
                f"Avg DistNet+Prep (all cones/frame): {avg_dist_net_plus_prep_all_cones:.2f} ms | (per cone): {avg_dist_net_plus_prep_per_cone:.2f} ms"
            )

        except queue.Empty:
            pass  # No new frame to display

        # Update plot
        try:
            while True:
                detections = self.plot_queue.get_nowait()
                if detections:
                    self.scatter.clear()
                    self.connection_plot.clear()

                    # **Reset latest detections for export**
                    self.latest_detections_for_export = []

                    positions = []
                    for det in detections:
                        class_id = det['yolo_class_id']
                        distance = det['distance']
                        angle_deg = det['angle']
                        angle_rad = np.deg2rad(angle_deg)

                        # Calculate ground distance
                        distance_horizontal = np.sqrt(max(distance ** 2 - 1.0 ** 2, 0))

                        # Compute ground coordinates
                        y_ground = -(distance_horizontal * np.cos(angle_rad))
                        x_ground = -(distance_horizontal * np.sin(angle_rad) - 0.06)

                        # **Store processed detection**
                        processed_det = {
                            'yolo_class_id': class_id,
                            'distance': distance,
                            'angle': angle_deg,
                            'x_ground': x_ground,
                            'y_ground': y_ground
                        }
                        self.latest_detections_for_export.append(processed_det)
                        logging.debug(f"Stored detection for export: {processed_det}")

                        positions.append((x_ground, y_ground, class_id))
                        color = YOLO_CLASS_MAPPING.get(class_id, {}).get('color', (255, 255, 255, 120))
                        self.scatter.addPoints([{'pos': (x_ground, y_ground), 'brush': pg.mkBrush(*color)}])

                    # Plot connections between detected cones
                    if self.show_lines_between_dots:
                        connections_x = []
                        connections_y = []
                        connections = set()
                        for i, (x1, y1, class_id1) in enumerate(positions):
                            min_dist = float('inf')
                            closest_j = -1
                            for j, (x2, y2, class_id2) in enumerate(positions):
                                if i == j or class_id1 != class_id2:
                                    continue
                                dist = np.hypot(x2 - x1, y2 - y1)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_j = j
                            if closest_j != -1:
                                sorted_indices = tuple(sorted((i, closest_j)))
                                if sorted_indices not in connections:
                                    connections.add(sorted_indices)
                                    x2, y2, _ = positions[closest_j]
                                    connections_x += [x1, x2, np.nan]
                                    connections_y += [y1, y2, np.nan]

                        if connections_x and connections_y:
                            self.connection_plot.setData(x=connections_x, y=connections_y)

                    # Plot Reference Cones if enabled
                    if self.show_reference_cones:
                        ref_positions = []
                        ref_brushes = []
                        for ref_cone in REFERENCE_CONES:
                            x_ref = ref_cone['x']
                            y_ref = ref_cone['y']
                            color_ref = ref_cone['color']  # RGBA with alpha=120
                            ref_positions.append((x_ref, y_ref))
                            ref_brushes.append(pg.mkBrush(color_ref[0], color_ref[1], color_ref[2], color_ref[3]))

                        # **Corrected setData Call: Use 'pos' as a keyword argument**
                        self.ref_cones_scatter.setData(pos=ref_positions, brush=ref_brushes,
                                                       size=self.reference_cone_size)
                    else:
                        self.ref_cones_scatter.clear()

        except queue.Empty:
            pass  # No new plot data

    def closeEvent(self, event):
        # Stop threads
        self.frame_capture.stop()
        self.frame_processor.stop()
        self.frame_capture.join()
        self.frame_processor.join()
        event.accept()

    # Slots for checkbox and spinbox changes
    def on_bbox_checkbox_changed(self, state):
        self.show_bounding_boxes = (state == QtCore.Qt.Checked)
        self.frame_processor.show_bboxes = self.show_bounding_boxes

    def on_bbox_labels_checkbox_changed(self, state):
        self.show_bounding_box_labels = (state == QtCore.Qt.Checked)
        self.frame_processor.show_bbox_labels = self.show_bounding_box_labels

    def on_prediction_text_checkbox_changed(self, state):
        self.show_prediction_text = (state == QtCore.Qt.Checked)
        self.frame_processor.show_prediction_text = self.show_prediction_text

    def on_text_size_changed(self, value):
        self.bounding_box_text_size = value
        self.frame_processor.bbox_text_size = value

    def on_lines_checkbox_changed(self, state):
        self.show_lines_between_dots = (state == QtCore.Qt.Checked)

    def on_detected_size_changed(self, value):
        self.detected_cone_size = value
        self.scatter.setSize(self.detected_cone_size)

    def on_reference_size_changed(self, value):
        self.reference_cone_size = value
        self.ref_cones_scatter.setSize(self.reference_cone_size)

    def on_ref_cones_checkbox_changed(self, state):
        self.show_reference_cones = (state == QtCore.Qt.Checked)

    # Slots for pitch and roll changes
    def on_pitch_changed(self, value):
        self.pitch = value
        self.frame_processor.set_pitch(self.pitch)

    def on_roll_changed(self, value):
        self.roll = value
        self.frame_processor.set_roll(self.roll)

    def export_current_frame_data(self):
        import os
        import json
        import datetime
        import numpy as np

        # Helper function to convert NumPy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(v) for v in obj)
            return obj

        # **Gather detected cones data**
        detected_cones = []
        try:
            # Access the latest detections
            detections = self.latest_detections_for_export if hasattr(self, 'latest_detections_for_export') else []
            if not detections:
                QtWidgets.QMessageBox.warning(self, "No Detections", "There are no detected cones to export.")
                return
            for det in detections:
                detected_cones.append({
                    'class_id': det['yolo_class_id'],
                    'name': YOLO_CLASS_MAPPING.get(det['yolo_class_id'], {}).get('name', 'unknown'),
                    'x_ground_m': det['x_ground'],
                    'y_ground_m': det['y_ground'],
                    'distance_m': det['distance'],
                    'angle_deg': det['angle']
                })
                logging.debug(f"Exporting detected_cones: {detected_cones}")
        except Exception as e:
            logging.error(f"Error retrieving detected cones: {e}")

        # **Gather reference cones data**
        reference_cones = []
        for ref_cone in REFERENCE_CONES:
            reference_cones.append({
                'name': ref_cone['name'],
                'class_id': ref_cone['class_id'],
                'x_ground_m': ref_cone['x'],
                'y_ground_m': ref_cone['y']
            })

        # **Gather performance stats**
        yolo_fps = self.frame_processor.yolo_fps
        combined_fps = self.frame_processor.combined_fps

        # Compute averages if available
        fp = self.frame_processor
        if fp.yolo_time_count > 0:
            avg_yolo_time = fp.yolo_time_sum / fp.yolo_time_count

            avg_dist_net_time_all_cones = (
                    fp.dist_net_time_sum_all_cones / fp.dist_net_time_count_frames) if fp.dist_net_time_count_frames > 0 else 0
            avg_prep_time_all_cones = (
                    fp.image_prep_time_sum_all_cones / fp.dist_net_time_count_frames) if fp.dist_net_time_count_frames > 0 else 0
            avg_dist_net_plus_prep_all_cones = (
                    fp.dist_net_plus_prep_time_sum_all_cones / fp.dist_net_time_count_frames) if fp.dist_net_time_count_frames > 0 else 0

            avg_dist_net_time_per_cone = (
                    fp.dist_net_time_sum_per_cone / fp.dist_net_time_count_per_cone) if fp.dist_net_time_count_per_cone > 0 else 0
            avg_prep_time_per_cone = (
                    fp.image_prep_time_sum_per_cone / fp.dist_net_time_count_per_cone) if fp.dist_net_time_count_per_cone > 0 else 0
            avg_dist_net_plus_prep_per_cone = (
                    fp.dist_net_plus_prep_time_sum_per_cone / fp.dist_net_time_count_per_cone) if fp.dist_net_time_count_per_cone > 0 else 0
        else:
            # No frames with cones processed yet, set all to 0
            avg_yolo_time = 0
            avg_dist_net_time_all_cones = 0
            avg_prep_time_all_cones = 0
            avg_dist_net_plus_prep_all_cones = 0
            avg_dist_net_time_per_cone = 0
            avg_prep_time_per_cone = 0
            avg_dist_net_plus_prep_per_cone = 0

        performance_stats = {
            'YOLO_FPS': yolo_fps,
            'CombinedNet_FPS': combined_fps,
            'Average_YOLO_Time_ms': avg_yolo_time,
            'Average_DistNet_Time_All_Cones_Per_Frame_ms': avg_dist_net_time_all_cones,
            'Average_Prep_Time_All_Cones_Per_Frame_ms': avg_prep_time_all_cones,
            'Average_DistNet_Prep_Time_All_Cones_Per_Frame_ms': avg_dist_net_plus_prep_all_cones,
            'Average_DistNet_Time_Per_Cone_ms': avg_dist_net_time_per_cone,
            'Average_Prep_Time_Per_Cone_ms': avg_prep_time_per_cone,
            'Average_DistNet_Prep_Time_Per_Cone_ms': avg_dist_net_plus_prep_per_cone
        }

        # **Combine all data**
        export_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'detected_cones': detected_cones,
            'reference_cones': reference_cones,
            'performance_stats': performance_stats
        }

        # Convert numpy types to native Python types before exporting
        export_data_converted = convert_numpy_types(export_data)

        # **Choose file save location**
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        default_filename_json = f"frame_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        default_filename_img = f"annotated_frame_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        default_filename_plot = f"current_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        # Prompt user to select directory to save all files together
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save Exported Data",
            "",
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        if directory:
            try:
                # Save JSON Data
                json_path = os.path.join(directory, default_filename_json)
                with open(json_path, 'w') as f:
                    json.dump(export_data_converted, f, indent=4)
                logging.info(f"Exported JSON data to {json_path}")

                # Save Annotated Image
                annotated_image_path = os.path.join(directory, default_filename_img)
                # Retrieve the latest processed frame
                try:
                    processed_frame = self.display_queue.get_nowait()
                    # Save the image using OpenCV
                    cv2.imwrite(annotated_image_path, processed_frame)
                    logging.info(f"Saved annotated image to {annotated_image_path}")
                except queue.Empty:
                    QtWidgets.QMessageBox.warning(self, "No Annotated Image", "No annotated image available to save.")
                    logging.warning("No annotated image available to save.")

                # Save Current Plot
                plot_image_path = os.path.join(directory, default_filename_plot)
                # Grab the plot widget as an image
                plot_pixmap = self.plot_widget.grab()
                plot_pixmap.save(plot_image_path, 'PNG')
                logging.info(f"Saved plot image to {plot_image_path}")

                # Inform the user
                QtWidgets.QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Data exported to:\n{json_path}\n{annotated_image_path}\n{plot_image_path}"
                )
            except Exception as e:
                logging.error(f"Error exporting data: {e}")
                QtWidgets.QMessageBox.critical(self, "Export Failed", f"Failed to export data: {e}")
