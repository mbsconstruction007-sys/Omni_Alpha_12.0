@echo off
echo 🚀 Starting Omni Alpha 5.0 Auto Git Update System
echo ================================================
echo.
echo This will automatically commit and push changes every 5 minutes
echo Press Ctrl+C to stop
echo.
python auto_git_update.py --interval 300
pause
