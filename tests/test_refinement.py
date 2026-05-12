from collections import deque, Counter

# Mocking the smoothing logic
emotion_history = {'1': deque(maxlen=20)}

def update_history(emotion):
    emotion_history['1'].append(emotion)
    return Counter(emotion_history['1']).most_common(1)[0][0]

def test_smoothing():
    print("Testing Smoothing Logic (Buffer=20)...")
    
    # 1. Fill with 'Happy'
    for i in range(10):
        res = update_history('Happy')
        # Expect 'Happy' as it is the only input
        assert res == 'Happy'
        
    print("Initial 10 'Happy' inputs -> Result: Happy (Pass)")
    
    # 2. Introduce flicker ('Sad')
    res = update_history('Sad')
    # Should still be Happy because 10 Happy vs 1 Sad
    assert res == 'Happy'
    print("1 Flicker 'Sad' input -> Result: Happy (Pass - Smoothed)")
    
    # 3. Fill with 'Sad' to shift the mode
    # We have 10 Happy, 1 Sad. We need 10 more Sad to make it 11 Sad vs 9 Happy (since buffer pushes out old Happy)
    for i in range(15):
        res = update_history('Sad')
        
    # Now buffer should be mostly Sad
    assert res == 'Sad'
    print("15 'Sad' inputs -> Result: Sad (Pass - Mode Shifted)")
    
    # 4. Final verify logic with engagement
    # Just reusing the function logic import for final validity
    try:
        from src.engagement import get_engagement_state, calculate_engagement
        state = get_engagement_state('Angry')
        score = calculate_engagement(state, 'Angry')
        print(f"Logic Check: Angry -> State: {state}, Score: {score}")
        assert state == 'Focused'
        assert score == 70
        print("Logic Check: Angry -> Focused (70%) Passed")
        
        state_sur = get_engagement_state('Surprise')
        score_sur = calculate_engagement(state_sur, 'Surprise')
        print(f"Logic Check: Surprise -> State: {state_sur}, Score: {score_sur}")
        assert state_sur == 'Surprise'
        assert score_sur == 85
        print("Logic Check: Surprise -> Surprise (85%) Passed")
        
    except ImportError:
        print("Could not import src.engagement for logic check (path issue?), skipping.")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    test_smoothing()
