import unittest
import sys
import os
from types import SimpleNamespace

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engagement import calculate_engagement, get_engagement_status

class TestEngagement(unittest.TestCase):
    def test_no_pose_no_emotion(self):
        # No pose detected, unknown emotion
        score = calculate_engagement(None, "Unknown")
        self.assertEqual(score, 0)

    def test_pose_unknown_emotion(self):
        # Pose detected, unknown emotion
        # Score: 40 (presence) + 10 (unknown+pose) + 20 (attention) = 70
        mock_pose = SimpleNamespace(boxes=[1])
        score = calculate_engagement(mock_pose, "Unknown")
        self.assertEqual(score, 70)

    def test_pose_happy(self):
        # Pose detected, Happy
        # Score: 40 (presence) + 40 (positive) + 20 (attention) = 100
        mock_pose = SimpleNamespace(boxes=[1])
        score = calculate_engagement(mock_pose, "Happy")
        self.assertEqual(score, 100)

    def test_pose_sad(self):
        # Pose detected, Sad
        # Score: 40 (presence) + 20 (negative) + 20 (attention) = 80
        mock_pose = SimpleNamespace(boxes=[1])
        score = calculate_engagement(mock_pose, "Sad")
        self.assertEqual(score, 80)

    def test_status_labels(self):
        self.assertEqual(get_engagement_status(80), "High")
        self.assertEqual(get_engagement_status(50), "Medium")
        self.assertEqual(get_engagement_status(20), "Low")

if __name__ == '__main__':
    unittest.main()
