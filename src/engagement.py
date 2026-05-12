import numpy as np

def calculate_engagement(state, emotion_label=None):
    """
    Calculate engagement score based on state and original emotion.
    """
    if emotion_label == 'Angry' and state == 'Focused':
        return 70
        
    if state == 'Surprise':
        return 85

    scores = {
        'Focused': 90,
        'Talking': 85,
        'Happy': 95,
        'Surprise': 85,
        'Confused': 70,
        'Frustrated': 60,
        'Distracted': 50,
        'Bored': 40,
        'Tired': 30,
        'Unattentive': 20,
        'Unknown': 30
    }
    return scores.get(state, 50)

def get_engagement_level(score):
    if score >= 85:
        return 'High'
    elif score >= 50:
        return 'Moderate'
    else:
        return 'Low'

def get_engagement_state(emotion_label, gaze_direction="Center", is_talking=False):
    # 1. Strong Gaze Priority
    if gaze_direction in ["Left", "Right", "Up", "Unknown"]:
        return "Unattentive"
        
    # 1.5 Slight Gaze Priority
    if gaze_direction in ["Slightly Left", "Slightly Right"]:
        return "Distracted"

    # 1.8 Lip Movement Priority
    if is_talking:
        return "Talking"
        
    # 2. YOLOv12 Emotion Priority
    if emotion_label == 'Focused':
        return 'Focused'
    if emotion_label == 'Confused':
        return 'Confused'
    if emotion_label == 'Frustrated':
        return 'Frustrated'
    if emotion_label == 'Bored':
        return 'Bored'
    if emotion_label == 'Drowsy':
        return 'Tired'
    if emotion_label == 'Looking Away':
        return 'Unattentive'
    if emotion_label == 'Happy':
        return 'Happy'
    if emotion_label == 'Surprise':
        return 'Surprise'

    # 3. Heuristic Gaze Fallback
    if gaze_direction == "Down":
        return "Focused"
        
    return 'Focused'
