import numpy as np

def calculate_engagement(emotion_label):
    """
    Calculate engagement score based on emotion.
    Assumes presence is true (called for detected person).
    :param emotion_label: Predicted emotion label.
    :return: Engagement score (0-100).
    """
    score = 40 # Base score for presence
    
    # Emotion Score
    positive_emotions = ['Happy', 'Surprise', 'Neutral']
    negative_emotions = ['Sad', 'Angry', 'Fear', 'Disgust']
    
    if emotion_label in positive_emotions:
        score += 40
    elif emotion_label in negative_emotions:
        score += 20
    elif emotion_label == 'Unknown':
        score += 10
        
    # Head Tilt / Attention (Placeholder)
    # Assume attentive if detected
    score += 20
        
    return min(max(score, 0), 100)

def get_engagement_status(score):
    if score >= 75:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"
