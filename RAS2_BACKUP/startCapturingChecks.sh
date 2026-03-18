#!/usr/bin/env bash
#
# run_brightness_checks.sh
#
# Sleeps for 10 s (so GUI’s up), then fires off two brightness-drop scripts
# completely detached, each logging to its own timestamped file.

set -x
sleep 10

# redirect wrapper’s own stdout+stderr into its trace log
exec >> /home/pi/logs/cron_trace.log 2>&1

# 1. Paths
PYTHON=/usr/bin/python3
BASEDIR=/home/pi/Code/ITCNaanCapture
LOGDIR=/home/pi/logs/brightness_checks
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# 2. Make sure log dirs exist
mkdir -p "$LOGDIR" /home/pi/logs

echo "[$(date)] Launching brightness checks..."

# 3. Launch first job completely detached
setsid "$PYTHON" "$BASEDIR/fetch.py" \
    >> "$LOGDIR/${TIMESTAMP}_fetch.log" 2>&1 < /dev/null &
PID0=$!
echo "[$(date)] Started picamera2 job with PID $PID0"

sleep 5
setsid "$PYTHON" "$BASEDIR/computeBrightnessDrop_center_picamera2.py" \
    >> "$LOGDIR/${TIMESTAMP}_picamera2.log" 2>&1 < /dev/null &
PID1=$!
echo "[$(date)] Started picamera2 job with PID $PID1"

# 4. Launch second job completely detached
setsid "$PYTHON" "$BASEDIR/computeBrightnessDrop_center_picamera2_16mp.py" \
    >> "$LOGDIR/${TIMESTAMP}_picamera2_16mp.log" 2>&1 < /dev/null &
PID2=$!
echo "[$(date)] Started picamera2_16mp job with PID $PID2"

echo "[$(date)] Wrapper done. Background PIDs: $PID1, $PID2, $PID0"
