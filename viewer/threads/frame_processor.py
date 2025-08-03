import cv2
import torch
import time
import logging
import threading
import queue
import numpy as np
from torchvision import transforms

from viewer.utils import is_valid_bbox, crop_to_square

class FrameProcessor(threading.Thread):
    def __init__(
        self,
        frame_queue,
        display_queue,
        plot_queue,
        yolo_model,
        combined_model,
        feature_scaler,
        target_scaler,
        device,
        class_mapping,
        show_bboxes=True,
        show_bbox_labels=True,
        show_prediction_text=True,
        bbox_text_size=1,
        initial_pitch=10.959,
        initial_roll=180
    ):
        super(FrameProcessor, self).__init__()
        self.frame_queue = frame_queue
        self.display_queue = display_queue
        self.plot_queue = plot_queue
        self.yolo_model = yolo_model
        self.combined_model = combined_model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.device = device
        self.class_mapping = class_mapping

        # Display control variables
        self.show_bboxes = show_bboxes
        self.show_bbox_labels = show_bbox_labels
        self.show_prediction_text = show_prediction_text
        self.bbox_text_size = bbox_text_size

        # Initialize pitch and roll
        self.pitch = initial_pitch
        self.roll = initial_roll

        # Initialize a lock for thread-safe updates
        self.pitch_roll_lock = threading.Lock()

        self.running = True
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((50, 50)),
            transforms.ToTensor()
        ])

        # FPS tracking
        self.yolo_fps = 0.0
        self.combined_fps = 0.0
        self.yolo_timer = time.time()
        self.combined_timer = time.time()
        self.yolo_counter = 0
        self.combined_counter = 0

        # Performance timing accumulators
        # only accumulate and average over frames that have at least one cone
        self.yolo_time_sum = 0.0
        self.yolo_time_count = 0

        # keep track of totals and counts:
        # For frames:
        self.dist_net_time_sum_all_cones = 0.0      # Sum of DistNet times for all cones in a frame
        self.image_prep_time_sum_all_cones = 0.0    # Sum of image prep times for all cones in a frame
        self.dist_net_plus_prep_time_sum_all_cones = 0.0

        self.dist_net_time_count_frames = 0         # How many frames with cones we've processed
        # For per-cone averages (cumulative):
        self.dist_net_time_sum_per_cone = 0.0
        self.image_prep_time_sum_per_cone = 0.0
        self.dist_net_plus_prep_time_sum_per_cone = 0.0
        self.dist_net_time_count_per_cone = 0       # Total cones processed over runtime

    def run(self):
        while self.running:
            try:
                frame_number, frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue  # No frame available, try again
            processed_frame, detections, timings = self.process_frame(frame, frame_number)
            if processed_frame is not None:
                try:
                    # Clear old frames in display queue
                    if not self.display_queue.empty():
                        try:
                            self.display_queue.get_nowait()  # Remove old frame
                        except queue.Empty:
                            pass
                    self.display_queue.put_nowait(processed_frame)
                except queue.Full:
                    logging.warning("Display queue is full. Dropping frame.")

                # Send detections to plot queue
                if detections:
                    self.plot_queue.put_nowait(detections)

                # Update timing statistics if we have cones in this frame
                if detections:
                    yolo_time_ms = timings['yolo_time_ms']
                    dist_net_times_ms = timings['dist_net_times_ms']
                    prep_times_ms = timings['prep_times_ms']
                    dist_net_plus_prep_ms = timings['dist_net_plus_prep_ms']

                    # Update YOLO time
                    self.yolo_time_sum += yolo_time_ms
                    self.yolo_time_count += 1

                    # Update DistNet times for this frame (all cones)
                    frame_dist_net_sum = sum(dist_net_times_ms)
                    frame_prep_sum = sum(prep_times_ms)
                    frame_dist_net_plus_prep_sum = sum(dist_net_plus_prep_ms)

                    # Update frame-based averages (for all cones combined)
                    self.dist_net_time_sum_all_cones += frame_dist_net_sum
                    self.image_prep_time_sum_all_cones += frame_prep_sum
                    self.dist_net_plus_prep_time_sum_all_cones += frame_dist_net_plus_prep_sum
                    self.dist_net_time_count_frames += 1

                    # Update per-cone averages
                    self.dist_net_time_sum_per_cone += frame_dist_net_sum
                    self.image_prep_time_sum_per_cone += frame_prep_sum
                    self.dist_net_plus_prep_time_sum_per_cone += frame_dist_net_plus_prep_sum
                    self.dist_net_time_count_per_cone += len(dist_net_times_ms)

    def process_frame(self, frame, frame_number):
        # Start timing for YOLO
        yolo_start = time.time()

        # Run YOLO detection with a confidence threshold
        results = self.yolo_model.predict(source=frame, device=self.device, save=False, show=False, conf=0.5)
        yolo_end = time.time()
        # Update YOLO FPS
        self.yolo_counter += 1
        if yolo_end - self.yolo_timer >= 1.0:
            self.yolo_fps = self.yolo_counter / (yolo_end - self.yolo_timer)
            self.yolo_timer = yolo_end
            self.yolo_counter = 0

        yolo_time_ms = (yolo_end - yolo_start) * 1000.0

        detections = []
        dist_net_times_ms = []
        prep_times_ms = []
        dist_net_plus_prep_ms = []

        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates and other details
                xyxy = box.xyxy.tolist()[0]
                confidence = box.conf.item()
                yolo_class_id = int(box.cls.item())
                x1, y1, x2, y2 = xyxy

                # Retrieve mapping info
                mapping = self.class_mapping.get(yolo_class_id)
                if not mapping:
                    logging.warning(f"No mapping found for YOLO class ID {yolo_class_id}. Skipping.")
                    continue

                dist_net_class_id = mapping['dist_net_class']
                class_name = mapping['name']
                color = mapping['color']

                # Retrieve current pitch and roll safely
                with self.pitch_roll_lock:
                    current_pitch = self.pitch
                    current_roll = self.roll

                # Compute normalized bounding box parameters
                x_center = ((x1 + x2) / 2) / frame.shape[1]
                y_center = ((y1 + y2) / 2) / frame.shape[0]
                width = (x2 - x1) / frame.shape[1]
                height = (y2 - y1) / frame.shape[0]
                roll = current_roll  # Use dynamic value
                pitch = current_pitch  # Use dynamic value

                # Validate bounding box coordinates
                if not is_valid_bbox(int(x1), int(y1), int(x2), int(y2), frame.shape[1], frame.shape[0]):
                    logging.warning(f"Invalid bounding box: ({x1}, {y1}), ({x2}, {y2})")
                    continue

                # Add margin
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                margin_x = int(bbox_width * 0.1)
                margin_y = int(bbox_height * 0.1)
                x1_new = max(int(x1 - margin_x), 0)
                y1_new = max(int(y1 - margin_y), 0)
                x2_new = min(int(x2 + margin_x), frame.shape[1])
                y2_new = min(int(y2 + margin_y), frame.shape[0])

                # Crop the bounding box from the image
                prep_start = time.time()
                cropped_img = frame[y1_new:y2_new, x1_new:x2_new]

                # Check if cropped image is valid
                if cropped_img.size == 0:
                    logging.warning(f"Empty crop for bbox: ({x1_new}, {y1_new}), ({x2_new}, {y2_new})")
                    continue

                # Process image for CombinedNet
                square_cropped_img = crop_to_square(cropped_img)
                gray_img = cv2.cvtColor(square_cropped_img, cv2.COLOR_BGR2GRAY)
                resized_gray = cv2.resize(gray_img, (50, 50), interpolation=cv2.INTER_AREA)

                # Prepare tensors
                image_tensor = self.transform(resized_gray).unsqueeze(0).to(self.device)
                features = np.array([[yolo_class_id, x_center, y_center, width, height, roll, pitch]])
                features_scaled = self.feature_scaler.transform(features)
                features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
                prep_end = time.time()
                prep_time_ms = (prep_end - prep_start) * 1000.0

                # Run CombinedNet prediction
                combined_start = time.time()
                with torch.no_grad():
                    predictions = self.combined_model(features_tensor, image_tensor)
                combined_end = time.time()
                # Update CombinedNet FPS
                self.combined_counter += 1
                if combined_end - self.combined_timer >= 1.0:
                    self.combined_fps = self.combined_counter / (combined_end - self.combined_timer)
                    self.combined_timer = combined_end
                    self.combined_counter = 0

                dist_net_time_ms = (combined_end - combined_start) * 1000.0
                dist_net_plus_prep_time = dist_net_time_ms + prep_time_ms

                # Inverse scaling predictions
                predictions_original = self.target_scaler.inverse_transform(predictions.cpu().numpy())
                predicted_distance, predicted_angle = predictions_original[0]

                detections.append({
                    'yolo_class_id': yolo_class_id,
                    'dist_net_class_id': dist_net_class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [int(x1_new), int(y1_new), int(x2_new), int(y2_new)],
                    'distance': predicted_distance,
                    'angle': predicted_angle
                })

                dist_net_times_ms.append(dist_net_time_ms)
                prep_times_ms.append(prep_time_ms)
                dist_net_plus_prep_ms.append(dist_net_plus_prep_time)

        # Draw detections on frame if enabled
        if self.show_bboxes or self.show_bbox_labels or self.show_prediction_text:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']

                # Draw bounding box if enabled
                if self.show_bboxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Show class name and confidence if enabled
                if self.show_bbox_labels:
                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, self.bbox_text_size, (0, 255, 0), 2)

                # Show predicted distance and angle if enabled
                if self.show_prediction_text:
                    distance_label = f"D:{det['distance']:.2f}m"
                    angle_label = f"A:{det['angle']:.2f}"
                    cv2.putText(frame, distance_label, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, self.bbox_text_size, (255, 0, 0), 2)
                    cv2.putText(frame, angle_label, (x1, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, self.bbox_text_size, (255, 0, 0), 2)

        frame_height, frame_width = frame.shape[:2]
        frame_resized = cv2.resize(frame, (frame_width, frame_height))

        # Return timing info
        timings = {
            'yolo_time_ms': yolo_time_ms,
            'dist_net_times_ms': dist_net_times_ms,
            'prep_times_ms': prep_times_ms,
            'dist_net_plus_prep_ms': dist_net_plus_prep_ms
        }

        return frame_resized, detections, timings

    def set_pitch(self, pitch):
        with self.pitch_roll_lock:
            self.pitch = pitch
            logging.info(f"Pitch updated to {self.pitch} degrees.")

    def set_roll(self, roll):
        with self.pitch_roll_lock:
            self.roll = roll
            logging.info(f"Roll updated to {self.roll} degrees.")

    def stop(self):
        self.running = False
