from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import time
import sys
import os
import numpy as np
from collections import deque, Counter

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.capture import VideoSource
from src.pose import PoseEstimator
from src.emotion import EmotionClassifier
from datetime import datetime
from src.engagement import calculate_engagement, get_engagement_state, get_engagement_level
from src.database import init_db, create_session, end_session, log_engagement, get_all_sessions, get_session_details, create_user, get_user_by_email, get_user_by_id

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
except Exception as e:
    print("Could not load face_landmarker.task:", e)
    face_landmarker = None

# Initialize Database
init_db()

app = Flask(__name__)
app.secret_key = 'super_secret_engagement_key' # In production, set this from environment variable

# Global objects
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

# Session State
session_active = False
current_session_id = None
session_start_time = None
last_log_time = 0

# Tracking history for smoothing
# Map: track_id -> deque of last 20 emotions
emotion_history = {}

# Map: track_id -> frames left to keep talking state (smoothing)
talking_buffer = {}

def generate_frames():
    global current_stats, emotion_history, talking_buffer, session_active, current_session_id, last_log_time
    if not video_source:
        return

    while True:
        ret, frame = video_source.get_frame()
        if not ret:
            break
            
        # 1. Pose Estimation with Tracking
        pose_results = pose_estimator.process_frame(frame)
        
        # CLEAN UI: Do NOT draw skeleton/keypoints.
        # Start with a clean copy of the frame.
        annotated_frame = frame.copy()
        
        total_engagement = 0
        people_count = 0
        emotions_detected = []
        states_detected = []
        
        if pose_results.boxes:
            # Get keypoints and Track IDs
            keypoints = pose_estimator.get_keypoints(pose_results)
            boxes = pose_results.boxes
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
            
            for i, kps in enumerate(keypoints):
                people_count += 1
                track_id = track_ids[i]
                
                # Get Face Box from Keypoints
                face_box = pose_estimator.get_face_box(kps, frame.shape)
                
                # Estimate Gaze/Head Pose
                gaze_dir, gaze_score = pose_estimator.estimate_head_pose(kps, frame.shape)
                
                raw_emotion = "Unknown"
                if face_box:
                    fx1, fy1, fx2, fy2 = face_box
                    face_crop = frame[fy1:fy2, fx1:fx2]
                    
                    # Predict Emotion
                    raw_emotion = emotion_classifier.predict(face_crop)
                    
                    # Detect Lip Movement using MediaPipe
                    is_talking_now = False
                    if face_landmarker and face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
                        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                        landmarker_result = face_landmarker.detect(mp_image)
                        
                        if landmarker_result.face_landmarks:
                            landmarks = landmarker_result.face_landmarks[0]
                            # MediaPipe FaceLandmarker: 13 is inner top lip, 14 is inner bottom lip
                            lip_dist_ratio = abs(landmarks[14].y - landmarks[13].y)
                            if lip_dist_ratio > 0.025: # threshold for open mouth
                                is_talking_now = True
                    
                    is_talking = False
                    if track_id != -1:
                        if is_talking_now:
                            talking_buffer[track_id] = 15 # Hold talking state for 15 frames
                        
                        if talking_buffer.get(track_id, 0) > 0:
                            is_talking = True
                            talking_buffer[track_id] -= 1
                    
                    # Draw Just Face Box (Thinner line)
                    # We will color it later based on state
                
                # --- Smoothing Logic ---
                if track_id != -1:
                    if track_id not in emotion_history:
                        emotion_history[track_id] = deque(maxlen=20)
                    emotion_history[track_id].append(raw_emotion)
                    
                    # Get smoothed emotion (Mode)
                    # Only smooth if we have enough history to be stable? 
                    # User asked for 20 frames "constant", so we take the mode of the last 20 frames.
                    history = emotion_history[track_id]
                    if len(history) >= 5: # Start showing something after 5 frames, but buffer size is 20
                        most_common = Counter(history).most_common(1)
                        smoothed_emotion = most_common[0][0]
                    else:
                        smoothed_emotion = raw_emotion # Fallback for first few frames
                else:
                    smoothed_emotion = raw_emotion

                emotions_detected.append(smoothed_emotion)
                
                # Calculate Engagement State & Score
                state = get_engagement_state(smoothed_emotion, gaze_direction=gaze_dir, is_talking=is_talking)
                score = calculate_engagement(state, emotion_label=smoothed_emotion)
                level = get_engagement_level(score)
                
                total_engagement += score
                states_detected.append(state)
                
                # --- VISUALIZATION ---
                if face_box:
                     fx1, fy1, fx2, fy2 = face_box
                     
                     # Determine color based on Level
                     if level == 'High':
                         color = (0, 255, 0) # Green
                     elif level == 'Moderate':
                         color = (0, 255, 255) # Yellow
                     else:
                         color = (0, 0, 255) # Red (Low)
                     
                     # Draw Face Box
                     cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), color, 2)
                     
                     # Draw Label OUTSIDE/ABOVE Box
                     # Format: [State] | Gaze | Level (Score)
                     display_gaze = "Away" if gaze_dir == 'Unknown' else gaze_dir
                     label_text = f"[{state}] {display_gaze} | {level}({score})"
                     
                     (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                     
                     text_y = max(fy1 - 10, h + 5)
                     # Background for text
                     cv2.rectangle(annotated_frame, (fx1, text_y - h - 5), (fx1 + w, text_y + 5), (0, 0, 0), -1)
                     cv2.putText(annotated_frame, label_text, (fx1, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Aggregate Stats
        avg_engagement = int(total_engagement / people_count) if people_count > 0 else 0
        
        # Simple status summary
        if avg_engagement >= 80: status_summ = "High"
        elif avg_engagement >= 50: status_summ = "Medium"
        else: status_summ = "Low"
        
        # Primary state
        # Primary state reflects actual states calculated per person, which includes gaze
        primary_state = max(set(states_detected), key=states_detected.count) if states_detected else "Waiting..."
        
        current_stats = {
            "engagement_score": avg_engagement,
            "emotion": primary_state,
            "status": status_summ,
            "people_count": people_count
        }
        
        # Log to Database if Session is Active (every 1 second)
        current_time = time.time()
        if session_active and current_session_id and (current_time - last_log_time >= 1.0):
            try:
                log_engagement(
                    current_session_id,
                    current_time,
                    avg_engagement,
                    primary_state,
                    people_count,
                    engagement_level=get_engagement_level(avg_engagement)
                )
                last_log_time = current_time
            except Exception as e:
                print(f"Error logging to DB: {e}")
        
        # Overlay Global Stats - Top Right
        stats_text = [
            f"Avg Engagement: {avg_engagement}%",
            f"Status: {status_summ}",
            f"People: {people_count}"
        ]
        
        overlay_w = 250
        overlay_h = 100
        cv2.rectangle(annotated_frame, (annotated_frame.shape[1] - overlay_w, 0), 
                      (annotated_frame.shape[1], overlay_h), (0, 0, 0), -1)
        
        for idx, line in enumerate(stats_text):
            cv2.putText(annotated_frame, line, 
                       (annotated_frame.shape[1] - overlay_w + 10, 30 + (idx * 30)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Real-time Timestamp (Bottom Right)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (tw, th), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(annotated_frame, ts, 
                   (annotated_frame.shape[1] - tw - 10, annotated_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = get_user_by_email(email)
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            return redirect(url_for('live'))
        else:
            flash('Invalid email or password')
            return redirect(url_for('login'))
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if get_user_by_email(email):
            flash('Email already registered')
            return redirect(url_for('register'))
            
        password_hash = generate_password_hash(password)
        if create_user(email, password_hash, name):
            user = get_user_by_email(email)
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            return redirect(url_for('live'))
        else:
            flash('Registration failed')
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/live')
@login_required
def live():
    return render_template('live.html', user_name=session.get('user_name'))

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
@login_required
def stats():
    return jsonify({
        **current_stats,
        "session_active": session_active
    })

@app.route('/api/start_session', methods=['POST'])
@login_required
def start_session_route():
    global session_active, current_session_id, session_start_time
    if not session_active:
        user_id = session.get('user_id')
        session_start_time = time.time()
        current_session_id = create_session(session_start_time, user_id=user_id)
        session_active = True
        return jsonify({"status": "started", "session_id": current_session_id})
    return jsonify({"status": "already_active"})

@app.route('/api/end_session', methods=['POST'])
@login_required
def end_session_route():
    global session_active, current_session_id, session_start_time
    if session_active:
        end_time = time.time()
        # Calculate final stats for session summary
        # For simplicity, taking current values, but ideally should be average of logs
        # Let frontend/dashboard handle aggregation or improve this later
        end_session(current_session_id, end_time, current_stats["engagement_score"], current_stats["status"])
        session_active = False
        current_session_id = None
        return jsonify({"status": "ended"})
    return jsonify({"status": "not_active"})

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user_name=session.get('user_name'))

@app.route('/api/sessions')
@login_required
def get_sessions():
    user_id = session.get('user_id')
    sessions = get_all_sessions(user_id=user_id)
    return jsonify(sessions)

@app.route('/api/session/<int:session_id>')
@login_required
def get_session_logs_route(session_id):
    data = get_session_details(session_id)
    if data:
        logs = data['logs']
        if logs:
            scores = [l['score'] for l in logs if l['score'] is not None]
            levels = [l['engagement_level'] for l in logs if l['engagement_level']]
            
            # Filter out "Unknown" if necessary, but keep for now
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
            data['stats'] = stats
            
        return jsonify(data)
    return jsonify({"error": "Session not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)
