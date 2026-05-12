import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pose import PoseEstimator
from src.engagement import get_engagement_state

class TestGazeLogic(unittest.TestCase):
    def setUp(self):
        self.pose_estimator = PoseEstimator()

    def test_gaze_center(self):
        # Mock keypoints
        nose = np.array([100, 100, 0.9])
        l_eye = np.array([120, 90, 0.9]) 
        r_eye = np.array([80, 90, 0.9])
        
        keypoints = [nose, l_eye, r_eye, np.array([0,0,0]), np.array([0,0,0])] 
        direction, score = self.pose_estimator.estimate_head_pose(keypoints, (500, 500))
        self.assertEqual(direction, "Center")

    def test_gaze_left(self):
        # Left Result: Nose closer to Left Eye (Image Right). Small Ratio.
        nose = np.array([115, 100, 0.9]) 
        l_eye = np.array([120, 90, 0.9])
        r_eye = np.array([80, 90, 0.9])
        
        keypoints = [nose, l_eye, r_eye, np.array([0,0,0]), np.array([0,0,0])]
        direction, score = self.pose_estimator.estimate_head_pose(keypoints, (500, 500))
        self.assertEqual(direction, "Left")

    def test_gaze_down(self):
        # Looking Down
        nose = np.array([100, 130, 0.9])
        l_eye = np.array([120, 90, 0.9])
        r_eye = np.array([80, 90, 0.9])
        
        keypoints = [nose, l_eye, r_eye, np.array([0,0,0]), np.array([0,0,0])]
        direction, score = self.pose_estimator.estimate_head_pose(keypoints, (500, 500))
        self.assertEqual(direction, "Down")

    def test_engagement_states(self):
        # Gaze Priority
        self.assertEqual(get_engagement_state("Happy", "Left"), "Unattentive")
        self.assertEqual(get_engagement_state("Neutral", "Right"), "Unattentive")
        self.assertEqual(get_engagement_state("Neutral", "Up"), "Unattentive")
        
        # Down = Focused
        self.assertEqual(get_engagement_state("Unknown", "Down"), "Focused")
        self.assertEqual(get_engagement_state("Neutral", "Down"), "Focused")
        
        # Center = Emotion Logic
        self.assertEqual(get_engagement_state("Happy", "Center"), "Happy")
        self.assertEqual(get_engagement_state("Sad", "Center"), "Tired")
        self.assertEqual(get_engagement_state("Neutral", "Center"), "Focused")

if __name__ == '__main__':
    unittest.main()
