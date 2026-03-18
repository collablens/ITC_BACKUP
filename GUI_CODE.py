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
from final_overcook_detection_up import analyze_image
from hairModified_EDGE import hair_detection
from hair_UnetPP import infer_single
# from crack_detection import crack_detect
from HoleDetection import detect_hole_spots
from dataclasses import dataclass
from rembg import remove
import os
import datetime
from pyhid_usb_relay import find
from rembg import remove, new_session

USE_GPU = True
DEVICE = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
if_image = True
# Thread-safe queue
pq = queue.Queue()
pq_s = queue.Queue()
u2p_session = new_session("u2netp")

# Frame buffer for live display (only overcooked and hair)
frame_buffer = {
    "overcooked": None,
    "hair": None,
    "shape": None,
    # "hole": None,
    # "hair2": None,
}

# ── module states ──────────────────────────────────────────────
module_states = {
    "overcooked": True,
    "hair": True,
    "shape": True,
    # "hole": False,
    # "hair2": False,
}

frame_buffer_lock = threading.Lock()

# Globals
_SENTINEL = object()
t = time.time()
GLOBAL_COUNT = 1
GLOBAL_COUNT_S = 1

# Directories
os.makedirs("captures", exist_ok=True)
os.makedirs("low_captures", exist_ok=True)
os.makedirs("results", exist_ok=True)
# os.makedirs("captures_cropped", exist_ok = True)

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
import tkinter as tk
from tkinter import simpledialog, messagebox

PASSWORD = "1234"
def toggle_module_with_password(module):
    # Ask for password
    entered = simpledialog.askstring("Password", "Enter password:", show="*")
    if entered == PASSWORD:
        toggle_module(module)
    else:
        messagebox.showerror("Error", "Incorrect password!")

def toggle_module(module):
    module_states[module] = not module_states[module]
    buttons[module].config(
        text=f"{module.upper()}: {'ON' if module_states[module] else 'OFF'}",
        bg="green" if module_states[module] else "red"
    )
    print(f"[INFO] {module} set to {module_states[module]}")

def control_gui():
    root = tk.Tk()
    root.title("Module Control")
    root.geometry("350x350")   # Increase window size (width x height)

    global buttons
    buttons = {}

    font_cfg = ("Arial", 18, "bold")   # Larger font for buttons

    for i, module in enumerate(module_states.keys()):
        btn = tk.Button(
            root,
            text=f"{module.upper()}: {'ON' if module_states[module] else 'OFF'}",
            width=18,
            height=2,          # Taller buttons
            font=font_cfg,     # Bigger text
            bg="green" if module_states[module] else "red",
            command=lambda m=module: toggle_module_with_password(m)
        )
        btn.grid(row=i, column=0, padx=20, pady=20)  # More padding
        buttons[module] = btn

    root.mainloop()


# ─────────── Relay-timing parameters (edit here only) ───────────
CHANNEL_ID      = 2      # 1-based index of the coil you wired
HOLD_TIME       = 1 # minimum seconds the relay must stay in ANY state
REFRESH_INT     = 0.2     # resend ON command every 0.5 s (avoids USB timeout)
ACTUATION_DELAY = 0   # wait this long BEFORE we touch the relay
# ----------------------------------------------------------------

# internal book-keeping (do not touch)
_last_state     = None     # None | True | False
_last_change_ts = 0.0      # epoch timestamp of the most recent flip


# ───────── Relay-timing parameters ─────────
CHANNEL_ID      = 2
HOLD_TIME       = 0.2      # dwell time for every state change
ON_PULSE_SEC    = 2.5      # NEW – length of the “ON” pulse you want
REFRESH_INT     = 0.2
ACTUATION_DELAY = 0.5
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

def remove_background(image_bgr: np.ndarray) -> np.ndarray:
    """
    Removes background using rembg and fills the transparent background with white.
    Returns a BGR image with white background.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image received")

    # Convert BGR → RGB → PIL
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")

    # Remove background using rembg
    output_pil = remove(pil_img, session=u2p_session)  # or use default session
    output_pil = output_pil.convert("RGBA")

    # Convert back to BGR (OpenCV format)
    output_bgr = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
    return output_bgr


def get_binary_mask(image_bgr: np.ndarray, threshold: int = 20) -> np.ndarray:
    """
    Converts BGR image to binary mask using fixed grayscale threshold.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image for thresholding.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary_mask


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
    h, w = bgr_image.shape[:2]
    print("[CROP] in:", bgr_image.shape)

    # safer ROI crop (optional)
    if w > 4050:
        bgr_roi = bgr_image[:, 256:4050]
    elif w > 256:
        bgr_roi = bgr_image[:, 256:]     # don’t try to go to 4050
    else:
        print("[CROP] too narrow:", w)
        return None

    rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
    pil_rgba = Image.fromarray(rgb).convert("RGBA")

    fg_removed = remove(pil_rgba, session=u2p_session)  # use same session
    fg_np = np.array(fg_removed)

    if fg_np.shape[2] < 4:
        print("[CROP] alpha missing")
        return None

    alpha = fg_np[:, :, 3]
    fg_ratio = np.count_nonzero(alpha) / alpha.size
    print(f"[CROP] fg_ratio={fg_ratio:.3f}")

    # only reject "almost empty"
    if fg_ratio < 0.01:
        print("[CROP] almost empty mask")
        return None

    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        print("[CROP] empty mask")
        return None

    l, t, r, b = xs.min(), ys.min(), xs.max(), ys.max()
    bbox_area = (r - l) * (b - t)

    # relative min area: 3% of ROI area (tune 0.02–0.08)
    min_object_area = int(0.03 * bgr_roi.shape[0] * bgr_roi.shape[1])
    print(f"[CROP] bbox_area={bbox_area}, min={min_object_area}")

    if bbox_area < min_object_area:
        print("[CROP] bbox too small")
        return None

    return bgr_roi[t:b+1, l:r+1]

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
    ct = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captures/capture_{ct}.jpg"
    save_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
    folder_path = os.path.join("low_captures", date_str)
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
        try:
            item_s = pq_s.get(timeout=1)         # ➋ the 1-second window
        except Empty:
            # nothing arrived in time → discard the high-res frame and continue
            print("[WARN] Low-res mate missing for >1 s – discarding high-res frame")
            continue

        if item_s is _SENTINEL:                  # shutdown requested on low queue
            pq_s.put(_SENTINEL)
            break

        # Unpack the paired items and dispatch
        prof, frame     = item
        prof_s, frame_s = item_s                # (was the bug you fixed earlier)

        threading.Thread(
            target=dispatcher_thread,
            args=(prof, frame, prof_s, frame_s),
            daemon=True
        ).start()


def dispatcher_thread(prof, frame, prof_s, frame_s):

    # Storing the cropped image
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")   # e.g. 20251015
    time_str = now.strftime("%H%M%S")   # e.g. 134502# create   low_captures/<date>/   if it doesn’t exist
    folder_path = os.path.join("captures", date_str)
    os.makedirs(folder_path, exist_ok=True)# save as  HHMMSS_low_capture.jpg  (adjust the name if you prefer)
    filename = os.path.join(folder_path, f"{time_str}_capture.jpg")
    cv2.imwrite(filename, frame)

    if frame is None or np.mean(frame) < 5:
        print(f"[WARN] Invalid frame input at index {prof.idx}")
        return
    
    if frame_s is None or np.mean(frame_s) < 5:
        print(f"[WARN] Invalid frame input at index {prof_s.idx}")
        return
    
    # cv2.imwrite("image_inside_dispatcher.png", frame)
    frame_bg_rem = frame.copy()
    frame_bg_rem = remove_background(frame_bg_rem)

    flag = True  # all detectors OK until proven otherwise

    with ThreadPoolExecutor(max_workers=3) as pool:
        future_to_key = {}        
        if module_states["overcooked"]:
            future_to_key[pool.submit(analyze_image, frame.copy())] = "overcooked"
        if module_states["hair"]:
            future_to_key[pool.submit(hair_detection, frame.copy())] = "hair"
        if module_states["shape"]:
            future_to_key[pool.submit(process_single_image, frame_s.copy())] = "shape"
        # if module_states["hole"]:
        #     future_to_key[pool.submit(detect_hole_spots, frame_bg_rem.copy())] = "hole"

        # compute timestamp once
        batch_ts  = datetime.datetime.now()
        date_str  = batch_ts.strftime("%Y%m%d")
        time_str  = batch_ts.strftime("%H%M%S")

        # prepare the folders once (instead of per key)
        results_dir = os.path.join("results", date_str, time_str)
        not_ok_dir  = os.path.join("NOT_OK", date_str, time_str)
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
    display_width = 480
    display_height = 420
    
    while True:
        canvas = np.zeros((display_height, display_width * 3, 3), dtype=np.uint8)
        with frame_buffer_lock:
            for idx, key in enumerate(["overcooked", "hair","shape"]):
                img = frame_buffer[key]
                if img is not None:
                    try:
                        resized = cv2.resize(img, (display_width, display_height))
                        canvas[:, idx*display_width:(idx+1)*display_width] = resized
                    except Exception as e:
                        print(f"[WARN] Failed to render {key} image: {e}")
        cv2.imshow(win_name, canvas)
        cv2.moveWindow(win_name, 0, 0)
        if cv2.waitKey(10) & 0xFF == 27:
            break
        
import os
import json
import shutil

LAST_RUN_FILE = 'last_cleanup_run.json'

def load_last_run_date():
    """Load the last run date from a file."""
    if os.path.exists(LAST_RUN_FILE):
        with open(LAST_RUN_FILE, 'r') as file:
            data = json.load(file)
            return datetime.datetime.strptime(data['last_run_date'], "%Y-%m-%d").date()
    return None  # If no previous run, return None

def save_last_run_date():
    """Save the current date as the last run date."""
    today = datetime.datetime.now().date()
    with open(LAST_RUN_FILE, 'w') as file:
        json.dump({"last_run_date": today.strftime("%Y-%m-%d")}, file)

def cleanup_old_files():
    """Deletes files older than 7 days in the specified directories."""
    global LAST_RUN_DATE
    today = datetime.datetime.now().date()
    
    # Check if the script has already run today
    if LAST_RUN_DATE == today:
        print("[INFO] Cleanup already ran today.")
        return

    # Update the last run date to today
    LAST_RUN_DATE = today
    save_last_run_date()  # Save the date to the file

    # Set the cutoff date for deletion (7 days ago)
    cutoff_date = today - datetime.timedelta(days=7)

    # Define the root folders to clean up
    root_folders = ["captures", "captures_down", "low_captures", "low_captures_down", "results", "results_back", "NOT_OK", "NOT_OK_back"]
    
    for root in root_folders:
        root_path = os.path.join(os.getcwd(), root)
        
        if not os.path.exists(root_path):
            continue
        
        for date_dir in os.listdir(root_path):
            try:
                # Convert directory name to date (YYYYMMDD format)
                dir_date = datetime.datetime.strptime(date_dir, "%Y%m%d").date()
                
                # If the directory is older than the cutoff date, remove it
                if dir_date < cutoff_date:
                    full_dir_path = os.path.join(root_path, date_dir)
                    print(f"[INFO] Deleting {full_dir_path} (older than {cutoff_date})")
                    shutil.rmtree(full_dir_path)
            except ValueError:
                continue
        
def schedule_cleanup():
    """Function to run the cleanup every day (runs in a background thread)."""
    while True:
        cleanup_old_files()
        time.sleep(86400)  # Sleep for 24 hours (86400 seconds)

# Loading the last run date from the file
LAST_RUN_DATE = load_last_run_date()


if __name__ == "__main__":
    try:
        # start_external_script()
        worker = threading.Thread(target=worker_allocator, daemon=True)
        server = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 2000, "debug": False, "use_reloader": False}, daemon=True)
        display = threading.Thread(target=display_thread, daemon=True)
        gui = threading.Thread(target=control_gui, daemon=True)
        cleanup = threading.Thread(target=schedule_cleanup, daemon=True)
        cleanup.start()
        gui.start()
        worker.start()
        server.start()
        display.start()
        worker.join()
        server.join()
        display.join()
        gui.join()
    except KeyboardInterrupt:
        pq.put(_SENTINEL)
        # cleanup_external_script()
    finally:
        cv2.destroyAllWindows()
