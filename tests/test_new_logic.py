import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.engagement import get_engagement_state, calculate_engagement

def test_engagement_logic():
    print("Testing Engagement Logic...")
    
    # Test Cases
    cases = [
        ('Happy', 'Happy'),
        ('Neutral', 'Focused'),
        ('Surprise', 'Focused'),
        ('Angry', 'Focused'),
        ('Sad', 'Tired'),
        ('Fear', 'Tired'),
        ('Disgust', 'Tired'),
        ('Unknown', 'Unattentive')
    ]
    
    for emotion, expected_state in cases:
        state = get_engagement_state(emotion)
        score = calculate_engagement(state)
        print(f"Emotion: {emotion:10} -> State: {state:12} -> Score: {score}")
        assert state == expected_state, f"Expected {expected_state}, but got {state}"
        
    print("\nAll logic tests passed!")

if __name__ == "__main__":
    test_engagement_logic()
