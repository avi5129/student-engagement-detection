
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from database import init_db, create_session, end_session, log_engagement, get_session_details
except ImportError:
    # Adjust path if running from root
    sys.path.append('src')
    from database import init_db, create_session, end_session, log_engagement, get_session_details

def reproduce():
    # 1. Init DB if needed
    init_db()
    
    # 2. Create a dummy session
    start_time = time.time()
    session_id = create_session(start_time)
    print(f"Created session {session_id}")
    
    # 3. Log some data
    # High
    log_engagement(session_id, start_time + 1, 90, "Happy", 1, "High")
    log_engagement(session_id, start_time + 2, 85, "Happy", 1, "High")
    # Moderate
    log_engagement(session_id, start_time + 3, 60, "Neutral", 1, "Moderate")
    # Low
    log_engagement(session_id, start_time + 4, 30, "Sad", 1, "Low")
    
    # End session
    end_session(session_id, start_time + 5, 66, "Moderate")
    
    # 4. Fetch details
    data = get_session_details(session_id)
    
    if not data:
        print("Error: Session not found")
        return

    logs = data['logs']
    if logs:
        scores = [l['score'] for l in logs if l['score'] is not None]
        levels = [l['engagement_level'] for l in logs if l['engagement_level']]
        
        from collections import Counter
        level_counts = Counter(levels)
        
        stats = {
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "distribution": {
                "High": level_counts.get("High", 0),
                "Moderate": level_counts.get("Moderate", 0),
                "Low": level_counts.get("Low", 0)
            }
        }
        
        print("\nCalculated Stats:")
        print(stats)
        
        expected_distribution = {'High': 2, 'Moderate': 1, 'Low': 1}
        
        if stats['distribution'] == expected_distribution:
            print("\nSUCCESS: Distribution logic is correct.")
        else:
            print(f"\nFAILURE: Expected {expected_distribution}, got {stats['distribution']}")

if __name__ == "__main__":
    reproduce()
