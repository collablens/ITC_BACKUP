import cv2
import time
import threading
import queue
from PIL import Image
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request
import subprocess
import atexit
import numpy as np
from ultralytics import YOLO
from ShapeDetection1 import process_single_image
# from CookDetection import detect_overcooked_spots
from final_overcook_detection_down import analyze_image
# from HairDetection import hair_detection
from hairModified_down import hair_detection
from dataclasses import dataclass
from rembg import remove
import os
import datetime
from pyhid_usb_relay import find

time_total = 0

USE_GPU = True
DEVICE = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
if_image = True
# Thread-safe queue
pq = queue.Queue()
pq_s = queue.Queue()

# Frame buffer for live display (only overcooked and hair)
frame_buffer = {
    "overcooked": None,
    "hair": None,
    # "shape": None
}
frame_buffer_lock = threading.Lock()

# Globals
_SENTINEL = object()
t = time.time()
GLOBAL_COUNT = 1
GLOBAL_COUNT_S = 1

# Directories
os.makedirs("captures_down", exist_ok=True)
os.makedirs("low_captures_down", exist_ok=True)
os.makedirs("results_back", exist_ok=True)

# Flask app
app = Flask(__name__)
print("[INFO] Loading models …")
yolo_model = YOLO("yolov8l.pt").to(DEVICE)

# ─────────── Relay setup ───────────
relay = find()
if relay is None:
    raise RuntimeError("❌ Relay not found! Check USB connection/permissions.")
relay_lock = threading.Lock()

# ── utilities ───────────────────────────────────────────────────────────────────
def clear_buffers():
    """
    • Empty both work queues (pq  &  pq_s) atomically
    • Reset the live-preview images held in `frame_buffer`
    """
    # flush preview frames
    with frame_buffer_lock:
        for k in frame_buffer:
            frame_buffer[k] = None

    # flush pending jobs
    with pq.mutex:      # queue.Queue already protects its internals with this lock
        pq.queue.clear()
    with pq_s.mutex:
        pq_s.queue.clear()

    print("[INFO] Frame queues and preview buffer cleared")
# ────────────────────────────────────────────────────────────────────────────────

# def pulse_relay(
#     on: bool,
#     hold_time: float        = 2.0,
#     refresh_interval: float = 0.2
# ):
#     """
#     Drive channel #1 of a REES52 / USB-HID relay.

#     Parameters
#     ----------
#     on : bool
#         • True  → energise CH-1 for `hold_time` seconds  
#         • False → ensure CH-1 is OFF immediately
#     hold_time : float
#         Seconds to hold the relay on when `on` is True
#     refresh_interval : float
#         USB command repeat period while holding
#     """
#     channel_id  = 2
#     with relay_lock:
#         if not on:                         # “good” naan → OFF and return
#             try:
#                 relay.set_state(channel_id, False)
#             except Exception as e:
#                 print(f"[WARN] relay off failed: {e}")
#             return

#         end_ts = time.time() + hold_time
#         while time.time() < end_ts:
#             try:
#                 relay.set_state(channel_id, True)
#             except Exception as e:
#                 print(f"[ERROR] relay write failed: {e}")
#                 break
#             time.sleep(refresh_interval)

#         # leave de-energised
#         try:
#             relay.set_state(channel_id, False)
#         except Exception:
#             pass





# ───────── Relay-timing parameters ─────────
CHANNEL_ID      = 1
HOLD_TIME       = 0.2      # dwell time for every state change
ON_PULSE_SEC    = 1.0      # NEW – length of the “ON” pulse you want
REFRESH_INT     = 0.2
ACTUATION_DELAY = 0.0
# ------------------------------------------


# internal book-keeping (do not touch)
_last_state     = None     # None | True | False
_last_change_ts = 0.0      # epoch timestamp of the most recent flip


def pulse_relay(*, request_on: bool = None, on: bool = None):
    """
    If request_on=True  →  pulse ON for ON_PULSE_SEC, then back to OFF.
    If request_on=False →  stay OFF for HOLD_TIME.
    Everything is still fully thread-safe and dwell-guarded.
    """
    global _last_state, _last_change_ts

    # argument compatibility
    if request_on is None and on is None:
        raise TypeError("pulse_relay() needs request_on= (or legacy on=)")
    if request_on is None:
        request_on = on

    # optional pre-delay
    # time.sleep(ACTUATION_DELAY)

    with relay_lock:
        now = time.time()

        # 1) honour the previous state's dwell time
        elapsed = now - _last_change_ts
        if elapsed < HOLD_TIME:
            time.sleep(HOLD_TIME - elapsed)

        # ─────── PATCH START ──────────────────────────────────────
        if request_on:
            # ---  Pulse ON  ---------------------------------------------------
            relay.set_state(CHANNEL_ID, True)
            _last_state     = True
            _last_change_ts = time.time()

            # keep-alive while the coil is energised
            end_ts = _last_change_ts + ON_PULSE_SEC
            while time.time() < end_ts:
                relay.set_state(CHANNEL_ID, True)
                time.sleep(min(REFRESH_INT, end_ts - time.time()))

            # ---  Return to OFF  --------------------------------------------
            relay.set_state(CHANNEL_ID, False)
            _last_state     = False
            _last_change_ts = time.time()

            # guarantee the OFF dwell, then exit
            time.sleep(HOLD_TIME)
            return
        
        # ─────── PATCH END ────────────────────────────────────────

        # request_on is False → just make sure we stay OFF
        if _last_state != False:
            relay.set_state(CHANNEL_ID, False)
            _last_state     = False
            _last_change_ts = time.time()

        # keep OFF for the mandatory dwell time
        time.sleep(HOLD_TIME)




@dataclass
class FrameProfile:
    idx: int = 0
    t_enqueue: float = 0.0
    t_dequeue: float = 0.0
    t_shape: float = 0.0
    t_cook: float = 0.0
    t_hair: float = 0.0

def start_external_script():
    global external_proc
    try:
        # external_proc = subprocess.Popen(
        #     ["python", "/home/itcfoods_collablens/Code/ITCTesting_03_07_2025/computeBrightnessDrop_center_usbcam_nD_old.py"],
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE
        # )
        # time.sleep(3)
        # if external_proc.poll() is None:
        #     print("External process started and is running.")
        # else:
        #     stderr_output = external_proc.stderr.read().decode()
        #     print("External process failed to start.")
        #     print("Error:", stderr_output)
        print("done")
    except Exception as e:
        print(f"Failed to start external process: {e}")

@atexit.register
def cleanup_external_script():
    global external_proc
    if external_proc and external_proc.poll() is None:
        print("Terminating external subprocess…")
        external_proc.terminate()
        try:
            external_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force killing external subprocess…")
            external_proc.kill()

def crop_yolo(frame):
    results = yolo_model.predict(frame, device=DEVICE, verbose=False, conf=0.1)
    for result in results:
        for box in result.boxes:
            if int(box.cls) in (53, 55):
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                return frame[ymin:ymax, xmin:xmax]
    print("No image detected")
    return None

def crop_bgr_image_using_rembg(bgr_image: np.ndarray):
    """
    Remove background with `rembg` and tightly crop the remaining foreground.
    If the foreground covers ≥ 80 % of the original frame, treat it as
    “no meaningful object” and return None.

    Parameters
    ----------
    bgr_image : np.ndarray
        Input image in BGR (OpenCV) format.

    Returns
    -------
    np.ndarray | None
        Cropped BGR image, or None when no significant foreground detected.
    """
    # Convert to RGBA for rembg
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_rgba = Image.fromarray(rgb).convert("RGBA")

    # Background removal
    fg_removed = remove(pil_rgba)
    fg_np = np.array(fg_removed)

    # Need an alpha channel to know foreground
    if fg_np.shape[2] < 4:
        print("[WARN] Alpha channel missing – skipping. No image detected.")
        return None

    alpha = fg_np[:, :, 3]

    # Fraction of pixels that are foreground
    fg_ratio = np.count_nonzero(alpha) / alpha.size
    if fg_ratio >= 0.80:
        print("[INFO] Foreground ≈ original (ratio {:.0%}) – no image detected."
              .format(fg_ratio))
        return None

    # Locate bounding box of foreground
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        print("[INFO] No foreground mask – no image detected.")
        return None

    l, t, r, b = xs.min(), ys.min(), xs.max(), ys.max()
    return bgr_image[t:b+1, l:r+1]

@app.route('/')
def home():
    return "Flask server is running!", 200

@app.route("/upload", methods=["POST"])
def upload():
    global GLOBAL_COUNT
    if 'image' not in request.files or request.files['image'].filename == '':
        return "No image part", 400

    arr = np.frombuffer(request.files['image'].read(), np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return "Decode error", 415
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")   # e.g. 20251015
    time_str = now.strftime("%H%M%S")   # e.g. 134502# create   low_captures/<date>/   if it doesn’t exist
    folder_path = os.path.join("captures_down", date_str)
    os.makedirs(folder_path, exist_ok=True)# save as  HHMMSS_low_capture.jpg  (adjust the name if you prefer)
    filename = os.path.join(folder_path, f"{time_str}_capture.jpg")
    cv2.imwrite(filename, frame_bgr)

    frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
    crop = crop_bgr_image_using_rembg(frame_bgr)
    if crop is not None:
        profile = FrameProfile(idx=GLOBAL_COUNT)
        profile.t_enqueue = time.time() - t
        pq.put((profile, crop))
        GLOBAL_COUNT += 1
        return "OK", 200
    else:
        clear_buffers()
        return "No image detected", 204

@app.route("/uploadLow", methods=["POST"])
def uploadLow():
    global GLOBAL_COUNT_S
    if 'image' not in request.files or request.files['image'].filename == '':
        return "No image part", 400

    arr = np.frombuffer(request.files['image'].read(), np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return "Decode error", 415
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")   # e.g. 20251015
    time_str = now.strftime("%H%M%S")   # e.g. 134502# create   low_captures/<date>/   if it doesn’t exist
    folder_path = os.path.join("low_captures_down", date_str)
    os.makedirs(folder_path, exist_ok=True)# save as  HHMMSS_low_capture.jpg  (adjust the name if you prefer)
    filename = os.path.join(folder_path, f"{time_str}_low_capture.jpg")
    cv2.imwrite(filename, frame_bgr)

    crop = frame_bgr
    if crop is not None:
        profile = FrameProfile(idx=GLOBAL_COUNT_S)
        profile.t_enqueue = time.time() - t
        pq_s.put((profile, crop))
        GLOBAL_COUNT_S += 1
        return "OK", 200
    else:
        clear_buffers()
        return "No image detected", 204

# def worker_allocator():
#     print("Worker thread started")
#     while True:
#         item = pq.get()
#         item_s = pq_s.get()
#         if item is _SENTINEL:
#             pq.put(_SENTINEL)
#             break

#         if item_s is _SENTINEL:
#             pq_s.put(_SENTINEL)
#             break

#         prof, frame = item
#         prof_s, frame_s = item_s
#         threading.Thread(target=dispatcher_thread, args=(prof, frame, prof_s, frame_s), daemon=True).start()

from queue import Empty          # ➊ add this near the other imports

# ── worker thread ───────────────────────────────────────────────────────────────
def worker_allocator():
    print("Worker thread started")
    while True:
        item = pq.get()                          # blocks until a high-res frame
        if item is _SENTINEL:                    # graceful shutdown
            pq.put(_SENTINEL)
            break

        # try to fetch the matching low-res frame; give up after 1 s
        # try:
        #     item_s = pq_s.get(timeout=1)         # ➋ the 1-second window
        # except Empty:
        #     # nothing arrived in time → discard the high-res frame and continue
        #     print("[WARN] Low-res mate missing for >1 s – discarding high-res frame")
        #     continue

        # if item_s is _SENTINEL:                  # shutdown requested on low queue
        #     pq_s.put(_SENTINEL)
        #     break

        # Unpack the paired items and dispatch
        prof, frame     = item
        # prof_s, frame_s = item_s                # (was the bug you fixed earlier)

        threading.Thread(
            target=dispatcher_thread,
            args=(prof, frame, None, None),
            daemon=True
        ).start()


def dispatcher_thread(prof, frame, prof_s, frame_s):

    if frame is None or np.mean(frame) < 5:
        print(f"[WARN] Invalid frame input at index {prof.idx}")
        return
    
    # if frame_s is None or np.mean(frame_s) < 5:
    #     print(f"[WARN] Invalid frame input at index {prof_s.idx}")
    #     return

    frame_dup = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    flag = True  # all detectors OK until proven otherwise
    time_total = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_to_key = {
            pool.submit(analyze_image, frame.copy()): "overcooked",
            pool.submit(hair_detection, frame.copy()): "hair",
            # pool.submit(process_single_image, frame_s.copy()): "shape"
        }

    batch_ts  = datetime.datetime.now()
    date_str  = batch_ts.strftime("%Y%m%d")
    time_str  = batch_ts.strftime("%H%M%S")

    # prepare the folders once (instead of per key)
    results_dir = os.path.join("results_back", date_str, time_str)
    not_ok_dir  = os.path.join("NOT_OK_back", date_str, time_str)
    os.makedirs(results_dir, exist_ok=True)
    

    for fut in as_completed(future_to_key):
        key = future_to_key[fut]
        try:
            result = fut.result(timeout=10)

            if result.get("Result") == "NOT OK":
                flag = False
                os.makedirs(not_ok_dir,  exist_ok=True)
                cv2.imwrite(os.path.join(not_ok_dir, f"{key}.jpg"),
                            result["annotated_image"])

            if result and "annotated_image" in result:
                with frame_buffer_lock:
                    frame_buffer[key] = result["annotated_image"]

                cv2.imwrite(os.path.join(results_dir, f"{key}.jpg"),
                            result["annotated_image"])
            else:
                print(f"[WARN] {key} returned invalid result")

        except Exception as e:
            print(f"[ERROR] {key} detection failed: {e}")

    # Relay actuation (single channel)
    try:
        pulse_relay(request_on=(not flag))
    except Exception as e:
        print(f"[ERROR] Relay signal failed: {e}")
               


def display_thread():
    win_name = "Live Results"
    display_width = 360
    display_height = 420
    
    while True:
        canvas = np.zeros((display_height, display_width * 2, 3), dtype=np.uint8)
        with frame_buffer_lock:
            for idx, key in enumerate(["overcooked", "hair"]):
                img = frame_buffer[key]
                if img is not None:
                    try:
                        resized = cv2.resize(img, (display_width, display_height))
                        canvas[:, idx*display_width:(idx+1)*display_width] = resized
                    except Exception as e:
                        print(f"[WARN] Failed to render {key} image: {e}")
        cv2.imshow(win_name, canvas)
        cv2.moveWindow(win_name, 0, 600)
        if cv2.waitKey(10) & 0xFF == 27:
            break

if __name__ == "__main__":
    try:
        start_external_script()
        worker = threading.Thread(target=worker_allocator, daemon=True)
        server = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 2001, "debug": False, "use_reloader": False}, daemon=True)
        display = threading.Thread(target=display_thread, daemon=True)
        worker.start()
        server.start()
        display.start()
        worker.join()
        server.join()
        display.join()
    except KeyboardInterrupt:
        pq.put(_SENTINEL)
        cleanup_external_script()
    finally:
        cv2.destroyAllWindows()
