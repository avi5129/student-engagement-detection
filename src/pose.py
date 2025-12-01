from ultralytics import YOLO
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self, model_path='yolov8n-pose.pt'):
        """
        Initialize the YOLOv8 Pose Estimator.
        :param model_path: Path to the YOLOv8 pose model.
        """
        self.model = YOLO(model_path)

    def process_frame(self, frame):
        """
        Process a frame to detect poses.
        :param frame: Input image frame.
        :return: Results object from YOLO.
        """
        results = self.model(frame, verbose=False)
        return results[0]

    def draw_keypoints(self, frame, results):
        """
        Draw keypoints and bounding boxes on the frame.
        :param frame: Input image frame.
        :param results: YOLO results object.
        :return: Annotated frame.
        """
        annotated_frame = results.plot()
        return annotated_frame

    def get_keypoints(self, results):
        """
        Extract keypoints from results.
        :param results: YOLO results object.
        :return: List of keypoints (x, y, conf) for each detected person.
        """
        if results.keypoints is not None:
            return results.keypoints.data.cpu().numpy()
        return []

    def get_face_box(self, keypoints, frame_shape):
        """
        Estimate face bounding box from keypoints (nose, eyes, ears).
        Keypoints indices: 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        :param keypoints: Array of keypoints for a single person.
        :param frame_shape: (height, width) of the frame.
        :return: (x1, y1, x2, y2) or None if not enough points.
        """
        # We need at least nose or eyes to estimate face
        face_indices = [0, 1, 2, 3, 4]
        valid_points = []
        
        for idx in face_indices:
            if idx < len(keypoints):
                x, y, conf = keypoints[idx]
                if conf > 0.5: # Confidence threshold
                    valid_points.append((x, y))
        
        if not valid_points:
            return None
            
        valid_points = np.array(valid_points)
        min_x = np.min(valid_points[:, 0])
        min_y = np.min(valid_points[:, 1])
        max_x = np.max(valid_points[:, 0])
        max_y = np.max(valid_points[:, 1])
        
        # Add padding
        width = max_x - min_x
        height = max_y - min_y
        
        pad_x = width * 0.5
        pad_y = height * 0.5
        
        x1 = max(0, int(min_x - pad_x))
        y1 = max(0, int(min_y - pad_y))
        x2 = min(frame_shape[1], int(max_x + pad_x))
        y2 = min(frame_shape[0], int(max_y + pad_y))
        
        return (x1, y1, x2, y2)
