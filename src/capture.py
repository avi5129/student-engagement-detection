import cv2
import threading
import time

class VideoSource:
    def __init__(self, source=0):
        """
        Initialize the video source.
        :param source: Video source index (int) or file path (str).
        """
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.lock = threading.Lock()
        self.running = True
        
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {self.source}")

    def get_frame(self):
        """
        Capture a frame from the video source.
        :return: (ret, frame) where ret is boolean indicating success.
        """
        with self.lock:
            if self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                return ret, frame
            return False, None

    def release(self):
        """
        Release the video source.
        """
        self.running = False
        with self.lock:
            if self.cap.isOpened():
                self.cap.release()
