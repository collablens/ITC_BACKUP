#!/bin/bash

source /home/itcfoods_collablens/miniconda3/etc/profile.d/conda.sh
conda activate itc

cd /home/itcfoods_collablens/Desktop/CODE

# Run both in background and capture their PIDs
python back.py
PID0=$!
python GUI_CODE.py &
PID1=$!
python GUI_CODE_down.py &
PID2=$!

# On Ctrl+C, kill both
trap "echo 'Terminating...'; kill $PID0 $PID1 $PID2; exit" SIGINT

# Wait for both processes to finish
wait $PID0
wait $PID1
wait $PID2

