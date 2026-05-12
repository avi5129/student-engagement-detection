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
        Process a frame to detect poses with tracking.
        :param frame: Input image frame.
        :return: Results object from YOLO.
        """
        # Enable tracking with persist=True to keep track IDs across frames
        results = self.model.track(frame, persist=True, verbose=False)
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
        
        # Reduce padding to 0.2 to match how the YOLO model was trained (tightly cropped faces)
        pad_x = width * 0.2
        pad_y = height * 0.2
        
        x1 = max(0, int(min_x - pad_x))
        y1 = max(0, int(min_y - pad_y))
        x2 = min(frame_shape[1], int(max_x + pad_x))
        y2 = min(frame_shape[0], int(max_y + pad_y))
        
        return (x1, y1, x2, y2)

    def estimate_head_pose(self, keypoints, frame_shape):
        """
        Estimate gaze direction/head pose from keypoints.
        :return: (gaze_direction, score)
        gaze_direction: "Center", "Left", "Right", "Up", "Down", "Unknown"
        """
        # Keypoints: 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear
        
        nose = keypoints[0][:2]
        l_eye = keypoints[1][:2]
        r_eye = keypoints[2][:2]
        l_ear = keypoints[3][:2]
        r_ear = keypoints[4][:2]
        
        # Check confidences
        if keypoints[0][2] < 0.4 or keypoints[1][2] < 0.4 or keypoints[2][2] < 0.4:
            return "Unknown", 0.0

        # Horizontal Ratio (Yaw)
        # Dist(Nose, Left Eye) vs Dist(Nose, Right Eye)
        # Note: In image coordinates, Left Eye is on the right side of the face from camera view if looking at camera.
        # But keypoints[1] is "left_eye" (person's left).
        
        d_nose_l_eye = np.linalg.norm(nose - l_eye)
        d_nose_r_eye = np.linalg.norm(nose - r_eye)
        
        # Avoid division by zero
        if d_nose_r_eye == 0: d_nose_r_eye = 0.001
        
        yaw_ratio = d_nose_l_eye / d_nose_r_eye
        
        # Vertical Ratio (Pitch)
        # Midpoint of eyes
        eye_mid = (l_eye + r_eye) / 2
        
        # We need ears to have a robust vertical estimate, or relative to bounding box?
        # A simple heuristic: Nose position relative to eyes.
        # If nose is very close to eye line -> Looking Up (or Center)
        # If nose is far below eye line -> Looking Down
        
        d_eye_dist = np.linalg.norm(l_eye - r_eye)
        d_nose_eye_line = nose[1] - eye_mid[1] # Positive if nose is below eyes
        
        pitch_ratio = d_nose_eye_line / (d_eye_dist + 0.001)
        
        # Thresholds (Tuned heuristically)
        # Yaw:
        # Center: roughly 1.0 (0.6 to 1.6)
        # Looking Left (Person's Left -> Camera Right): Ratio > 2.0
        # Looking Right (Person's Right -> Camera Left): Ratio < 0.5
        
        # Pitch:
        # Normal: 0.3 to 0.6
        # Looking Down: > 0.8
        # Looking Up: < 0.2
        
        direction = "Center"
        
        # Check Horizontal
        if yaw_ratio > 1.6: # Person looking far to their left (Camera Right)
            direction = "Right" 
        elif yaw_ratio > 1.2:
            direction = "Slightly Right"
        elif yaw_ratio < 0.6: # Person looking far to their right (Camera Left)
            direction = "Left"
        elif yaw_ratio < 0.85:
            direction = "Slightly Left"
            
        # Check Vertical (Override horizontal if strong)
        elif pitch_ratio > 0.8:
            direction = "Down"
        elif pitch_ratio < 0.15:
            direction = "Up"
            
        return direction, 1.0
