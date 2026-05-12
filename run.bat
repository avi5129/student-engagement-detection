@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Starting application...
echo NOTE: First startup may take 1-2 minutes to download models and initialize TensorFlow. Please wait.
set TF_ENABLE_ONEDNN_OPTS=0
python web/app.py
pause
