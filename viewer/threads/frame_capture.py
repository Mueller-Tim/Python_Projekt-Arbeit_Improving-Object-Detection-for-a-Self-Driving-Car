import cv2
import logging
import threading
import queue

class FrameCapture(threading.Thread):
    def __init__(self, video_source, frame_queue):
        super(FrameCapture, self).__init__()
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Cannot open video source {video_source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame_queue = frame_queue
        self.running = True
        self.frame_number = 0

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                logging.warning("Failed to read frame from video source.")
                break
            self.frame_number += 1
            try:
                # Always keep only the latest frame
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()  # Remove old frame
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait((self.frame_number, frame))
            except queue.Full:
                logging.warning("Frame queue is full. Dropping frame.")

    def stop(self):
        self.running = False
        self.cap.release()
        logging.info("FrameCapture thread stopped.")
