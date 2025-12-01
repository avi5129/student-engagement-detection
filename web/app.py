from flask import Flask, render_template, Response, jsonify
import cv2
import time
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.capture import VideoSource
from src.pose import PoseEstimator
from src.emotion import EmotionClassifier
from src.engagement import calculate_engagement, get_engagement_status

app = Flask(__name__)

# Global objects
# Try index 0 first, then 1 if failed (for built-in camera)
try:
    video_source = VideoSource(0)
    print("Opened camera 0")
except:
    try:
        video_source = VideoSource(1)
        print("Opened camera 1")
    except:
        print("Error: Could not open camera 0 or 1")
        video_source = None

pose_estimator = PoseEstimator()
emotion_classifier = EmotionClassifier()

# Global state
current_stats = {
    "engagement_score": 0,
    "emotion": "Waiting...",
    "status": "Waiting...",
    "people_count": 0
}

def generate_frames():
    global current_stats
    if not video_source:
        return

    while True:
        ret, frame = video_source.get_frame()
        if not ret:
            break
            
        # 1. Pose Estimation
        pose_results = pose_estimator.process_frame(frame)
        annotated_frame = pose_estimator.draw_keypoints(frame, pose_results)
        
        total_engagement = 0
        people_count = 0
        emotions_detected = []
        
        if pose_results.boxes:
            # Get keypoints
            keypoints = pose_estimator.get_keypoints(pose_results)
            
            for i, kps in enumerate(keypoints):
                people_count += 1
                
                # Get Face Box from Keypoints
                face_box = pose_estimator.get_face_box(kps, frame.shape)
                
                emotion_label = "Unknown"
                if face_box:
                    fx1, fy1, fx2, fy2 = face_box
                    face_crop = frame[fy1:fy2, fx1:fx2]
                    
                    # Predict Emotion
                    emotion_label = emotion_classifier.predict(face_crop)
                    
                    # Draw face box
                    cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)
                
                emotions_detected.append(emotion_label)
                
                # Calculate Engagement
                score = calculate_engagement(emotion_label)
                total_engagement += score
                
                # Draw Info on Person Box (if available)
                if i < len(pose_results.boxes):
                    box = pose_results.boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box[:4])
                    color = (0, 255, 0) if score > 50 else (0, 0, 255)
                    cv2.putText(annotated_frame, f"{emotion_label} ({score}%)", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Aggregate Stats
        avg_engagement = int(total_engagement / people_count) if people_count > 0 else 0
        status = get_engagement_status(avg_engagement)
        primary_emotion = max(set(emotions_detected), key=emotions_detected.count) if emotions_detected else "Waiting..."
        
        current_stats = {
            "engagement_score": avg_engagement,
            "emotion": primary_emotion,
            "status": status,
            "people_count": people_count
        }
        
        # Overlay Global Stats
        cv2.putText(annotated_frame, f"Avg Engagement: {avg_engagement}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"People: {people_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def stats():
    return jsonify(current_stats)

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)
