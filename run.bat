@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Starting application...
python web/app.py
pause
