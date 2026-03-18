from picamera2 import Picamera2
import cv2
import numpy as np
import time
import threading
import requests
from pathlib import Path

# ----------------------- CONFIG -----------------------------------
# Load UPLOAD_URL from upload_url.txt in the same directory as this script
SCRIPT_DIR = Path(__file__).resolve().parent
URL_FILE = "/home/pi/upload_gs_cam_url"
try:
    with open(URL_FILE, "r") as f:
        UPLOAD_URL = f.read().strip()
        if not UPLOAD_URL:
            raise ValueError("upload_url.txt is empty")
except Exception as e:
    raise RuntimeError(f"Failed to load upload URL from {URL_FILE}: {e}")

# ----------------------- CONFIG -----------------------------------
#UPLOAD_URL        = "http://192.168.68.126:2000/uploadLow"
W, H              = 1280, 720          # preview resolution
WIN               = 50                 # detection-window size (square)
SAMPLE_SECS       = 2                  # time to learn baseline
DELTA             = 20                 # trigger band around baseline
COOLDOWN_FRAMES   = 20                 # frames before next shot
JPEG_QUALITY      = 80
# ------------------------------------------------------------------

def send_jpeg_array(arr, name):
    ok, buf = cv2.imencode(".jpg", arr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        print("[ERROR] JPEG encode failed")
        return
    files = {"image": (name, buf.tobytes(), "image/jpeg")}
    try:
        r = requests.post(UPLOAD_URL, files=files, timeout=5)
        print(f"[UPLOAD] {name} → {r.status_code}")
    except Exception as e:
        print(f"[ERROR] upload {name}: {e}")

# -------------------- CAMERA SET-UP --------------------------------
picam2       = Picamera2(1)
video_config = picam2.create_video_configuration(main={"size": (W, H)})
still_config = picam2.create_still_configuration(main={"size": (1440, 1080)})

picam2.configure(video_config)
picam2.start(); time.sleep(1)

#picam2.set_controls({
#    "AfMode": 0, "LensPosition": 2.0,
#    "AeEnable": False, "ExposureTime": 500
#})
time.sleep(1.5)
# -------------------------------------------------------------------

# -------------------- MULTI-WINDOW VERTICAL SETUP ------------------
# horizontal center for all windows
CX = W // 2
# vertical centers at 1/4, 1/2, 3/4 of the frame height
y_centers     = [H // 4, H // 2, 3 * H // 4]
# compute top-left for each window
windows       = [(int(CX - WIN//2), int(y - WIN//2)) for y in y_centers]

# per-window baseline storage
baseline_values = [[] for _ in windows]
baseline_means  = [0.0] * len(windows)
baseline_ready  = False

cooldown   = 0
shot_index = 0
t_start    = time.time()
# -------------------------------------------------------------------

while True:
    loop_t0 = time.time()

    # 1) Grab preview frame & convert to grayscale
    frame = picam2.capture_array()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2) Compute mean luminance for each window
    avgs = []
    for (X0, Y0) in windows:
        roi = gray[Y0:Y0 + WIN, X0:X0 + WIN]
        avgs.append(float(roi.mean()))

    # 3) Build per-window baseline during initial SAMPLE_SECS
    if not baseline_ready:
        if time.time() - t_start < SAMPLE_SECS:
            for i, avg in enumerate(avgs):
                baseline_values[i].append(avg)
        else:
            for i, vals in enumerate(baseline_values):
                baseline_means[i] = float(np.mean(vals))
            baseline_ready = True
            print(f"[BASELINE] Learned means = {[f'{m:.1f}' for m in baseline_means]}")
    else:
        # 4) Trigger on any window whose avg deviates by > DELTA
        if cooldown == 0:
            for i, avg in enumerate(avgs):
                if abs(avg - baseline_means[i]) > DELTA:
                    fname = f"capture_{shot_index}_win{i}.jpg"
                    print(f"[TRIGGER] win{i} {fname}: avg={avg:.1f}, base={baseline_means[i]:.1f}")

                    hi_res = picam2.switch_mode_and_capture_array(still_config)
                    threading.Thread(target=send_jpeg_array,
                                     args=(hi_res, fname), daemon=True).start()
                    picam2.switch_mode(video_config)

                    cooldown = COOLDOWN_FRAMES
                    shot_index += 1
                    break  # only one capture per loop

    # decrement cooldown if active
    if cooldown:
        cooldown -= 1

    # 5) Overlay each window and its stats
    for i, (X0, Y0) in enumerate(windows):
        cv2.rectangle(frame, (X0, Y0), (X0 + WIN, Y0 + WIN), (0, 255, 0), 2)
        txt = f"W{i} μ:{avgs[i]:.1f}"
        if baseline_ready:
            delta = avgs[i] - baseline_means[i]
            txt += f"  Δ:{delta:+.1f}"
        cv2.putText(frame, txt, (X0, Y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 6) Draw center crosshair
    cv2.line(frame, (CX - 10, H // 2), (CX + 10, H // 2), (0, 0, 255), 1)
    cv2.line(frame, (CX, H // 2 - 10), (CX, H // 2 + 10), (0, 0, 255), 1)

    # 7) Display loop timing
    fps_ms = (time.time() - loop_t0) * 1000
    cv2.putText(frame, f"{fps_ms:.1f} ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Dynamic-Threshold Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
