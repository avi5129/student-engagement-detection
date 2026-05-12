import sys
import os
import time
import random

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from database import init_db, create_session, end_session, log_engagement
except ImportError:
    # If running from src parent
    sys.path.append('src')
    from database import init_db, create_session, end_session, log_engagement

def populate():
    print("Initializing DB...")
    init_db()
    
    # Create 3 sessions
    for i in range(3):
        start_time = time.time() - ((i + 1) * 3600) # 1 hour ago, 2 hours ago...
        print(f"Creating Session {i+1}...")
        
        session_id = create_session(start_time)
        
        # Log 50 entries
        total_score = 0
        count = 0
        
        for j in range(50):
            ts = start_time + (j * 2)
            
            # Random score based on session type
            if i == 0: # High Engagement Session
                score = random.randint(70, 100)
                emotion = 'Happy'
                level = 'High'
            elif i == 1: # Mixed
                score = random.randint(30, 90)
                emotion = 'Neutral'
                level = 'Moderate' if score > 50 else 'Low'
            else: # Low
                score = random.randint(10, 60)
                emotion = 'Sad'
                level = 'Low'
                
            log_engagement(session_id, ts, score, emotion, 1, level)
            total_score += score
            count += 1
            
        avg = int(total_score / count)
        end_time = start_time + 100
        
        status = "High" if avg > 80 else ("Moderate" if avg > 50 else "Low")
        end_session(session_id, end_time, avg, status)
        print(f"Session {session_id} ended with avg {avg}")

    print("Done! Database populated.")

if __name__ == "__main__":
    populate()
