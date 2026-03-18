import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp
from rembg import remove
from PIL import Image
import time
from pathlib import Path
import datetime
import argparse, csv
from hair_UnetPP_mask import infer_single
import io
from rembg import remove, new_session

# ----------------------
# CONFIG
# ----------------------
IMG_SIZE = 1824
PIECES = 3
PATCH_SIZE = IMG_SIZE // PIECES  # 800×800
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "best_model.pth"
# os.makedirs("masks_results", exist_ok=True)
# os.makedirs("Combined_TestResults", exist_ok=True)
u2p_session = new_session("u2netp")

# ----------------------
# LOAD MODEL
# ----------------------
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(DEVICE)

assert os.path.exists(model_path), f"❌ Model not found: {model_path}"
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print(f"✅ Loaded best model from {model_path}")
# ----------------------
# Coriander Removal of areas
# ----------------------

#!/usr/bin/env python3

# ---------- DEFAULTS (formerly CLI args) ----------
INPUT_DIR = r"C:\Users\singh\Downloads\Pass_Check"         # folder of images OR a single file path
EXT = "png"                  # used when INPUT_DIR is a folder
OUTPUT_DIR = Path("Cliantro_Results")
MIN_AREA = 20
OPEN_K = 3
CLOSE_K = 5
NEAR_PX = 12
CLOSE_AFTER_UNION = 3
PAD_PX = 40
SAVE_MASKS = True
SAVE_CROPS = False
SAVE_CSV = False
BLUR = True  # equivalent to not using --no-blur
# HSV ranges (pass 1 and pass 2)
P1_H_LO, P1_H_HI = 19, 46
P1_S_LO, P1_S_HI = 152, 230
P1_V_LO, P1_V_HI = 24, 125
P2_H_LO, P2_H_HI = 22, 44
P2_S_LO, P2_S_HI = 106, 240
P2_V_LO, P2_V_HI = 20, 125

p1_lo = np.array([P1_H_LO, P1_S_LO, P1_V_LO], np.uint8)
p1_hi = np.array([P1_H_HI, P1_S_HI, P1_V_HI], np.uint8)
p2_lo = np.array([P2_H_LO, P2_S_LO, P2_V_LO], np.uint8)
p2_hi = np.array([P2_H_HI, P2_S_HI, P2_V_HI], np.uint8)
# --------------------------------------------------

def remove_bg(image_bgr):
    
    # new_size = (640, 480)  
    # resized_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_LANCZOS4)

    if image_bgr is None:
        raise ValueError("Empty image received")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    output_pil = remove(pil_img, session=u2p_session)
    output_rgb = np.array(output_pil)
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    return output_bgr

def extract_binary(image_bgr, t, pad: int = 20):
    if t == 0:
        cv2.imwrite("Glare_Images/original.png", image_bgr)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if gray is None or gray.size == 0:
        raise ValueError("Image could not be loaded or is empty.")

    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Threshold mask is empty – no foreground found.")

    largest = max(contours, key=cv2.contourArea)

    clean_mask = np.zeros_like(thresh)
    cv2.drawContours(clean_mask, [largest], contourIdx=-1,
                     color=255, thickness=cv2.FILLED)

    # ❗ return the full-size gray + mask, not cropped
    return gray, clean_mask

def extract_foreground_mask_rembg(rgb_image, erosion_pixels=250):
    # Step 1: Load image using PIL for rembg
    
    input_pil = Image.fromarray(rgb_image).convert("RGBA")

    # Step 2: Use rembg to remove background and get the alpha channel as mask
    output_pil = remove(input_pil)  # Returns RGBA image with transparent background
    alpha_channel = np.array(output_pil.split()[-1])  # Extract the alpha channel

    # Step 3: Threshold the alpha to get binary mask
    _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)

    # Step 4: Erode the mask by N pixels to remove outer edge
    kernel = np.ones((erosion_pixels, erosion_pixels), np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

    return binary_mask, eroded_mask

def make_mask(hsv, lo, hi):
    return cv2.inRange(hsv, lo, hi)

def clean_mask(mask, open_k=3, close_k=5):
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def make_final_box_mask(shape, rects):

    H, W = shape[:2]
    mask = np.full((H, W), 255, np.uint8)  # white background

    for (x0, y0, x1, y1) in rects:
        # Clamp & guard
        x0 = max(0, min(W, x0)); y0 = max(0, min(H, y0))
        x1 = max(0, min(W, x1)); y1 = max(0, min(H, y1))
        if x1 > x0 and y1 > y0:
            # x1,y1 are exclusive -> draw to (x1-1, y1-1)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 0, thickness=-1)

    return mask

def dilate(mask, radius_px):
    if radius_px <= 0:
        return mask
    ksz = 2*int(radius_px) + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    return cv2.dilate(mask, k)

def expand_bbox(x, y, w, h, pad_px, W, H):
    x0 = max(0, x - pad_px)
    y0 = max(0, y - pad_px)
    x1 = min(W, x + w + pad_px)   # exclusive
    y1 = min(H, y + h + pad_px)   # exclusive
    return x0, y0, x1, y1

def cliantro(img, out_dir = OUTPUT_DIR, p1_lo = p1_lo, p1_hi = p1_hi, p2_lo = p2_lo, p2_hi = p2_hi,
                  open_k = OPEN_K, close_k = CLOSE_K, near_px = NEAR_PX, close_after_union = CLOSE_AFTER_UNION,
                  min_area = MIN_AREA, pad_px = PAD_PX, blur = BLUR):
    
    # pth = Path(pth)
    # out_dir = Path(out_dir)

    # img = cv2.imread(str(pth), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Could not read image")
        return np.zeros((1, 1), dtype=np.uint8)  # safe fallback
    H, W = img.shape[:2]

    work = cv2.GaussianBlur(img, (3,3), 0) if blur else img
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    # --- Pass 1 ---
    m1 = clean_mask(make_mask(hsv, p1_lo, p1_hi), open_k, close_k)

    # --- Pass 2 ---
    m2 = clean_mask(make_mask(hsv, p2_lo, p2_hi), open_k, close_k)

    # Only keep pass-2 pixels near pass-1
    m1_dil  = dilate(m1, near_px)
    m2_near = cv2.bitwise_and(m2, m1_dil)

    # Combine
    combined = cv2.bitwise_or(m1, m2_near)

    # Optional closing after union
    if close_after_union > 0:
        ksz = 2*int(close_after_union) + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)

    # Find final contours
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare outputs
    annotated = img.copy()

    rows = []
    kept = 0
    padded_rects = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        x0, y0, x1, y1 = expand_bbox(x, y, w, h, pad_px, W, H)
        padded_rects.append((x0, y0, x1, y1))

        kept += 1
        # tag = f"{pth.stem}_speck_{kept:03d}"

        # Draw contour + tight bbox + padded bbox
        cv2.drawContours(annotated, [c], -1, (0,255,255), 2)              # yellow contour
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,165,255), 2)      # orange tight bbox
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (255,255,0), 1)      # cyan padded bbox
        cv2.putText(annotated, f"#{kept}", (x, max(0, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        
    # Save annotated
    # out_dir.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite(str(out_dir / f"{pth.stem}_annotated.png"), annotated)

    # Save masks (optional)
    ct = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_box_mask = make_final_box_mask(img.shape, padded_rects)
    # cv2.imwrite(str(out_dir / f"{ct}_final_box_mask.png"), final_box_mask)

    return final_box_mask

# ----------------------
# HELPERS (unchanged)
# ----------------------
def boxes_overlap_or_close(box1, box2, threshold=30):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max + threshold < x2_min or x2_max + threshold < x1_min or
                y1_max + threshold < y2_min or y2_max + threshold < y1_min)

def merge_boxes(boxes, threshold=30):
    merged = []
    for box in boxes:
        matched = False
        for group in merged:
            if any(boxes_overlap_or_close(box, other, threshold) for other in group):
                group.append(box)
                matched = True
                break
        if not matched:
            merged.append([box])

    unioned = []
    for group in merged:
        x_min = min(box[0] for box in group)
        y_min = min(box[1] for box in group)
        x_max = max(box[2] for box in group)
        y_max = max(box[3] for box in group)
        unioned.append((x_min, y_min, x_max, y_max))
    return unioned

def detect_grooves(image_bgr):
    """
    Detect grooves / creases on naan.
    Returns:
        groove_mask: uint8 mask (255 = groove)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)    # 1) Strong blur to suppress hair-like thin lines
    gray_blur = cv2.GaussianBlur(gray, (11, 11), 0)    # 2) Black-hat to highlight dark concave structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    blackhat = cv2.morphologyEx(gray_blur, cv2.MORPH_BLACKHAT, kernel)    # 3) Normalize then threshold
    bh_norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(
        bh_norm, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )    # 4) Morphological open to remove tiny specks
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )

     # 5) Keep only larger regions (to avoid tiny noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = 500  # tweak per dataset
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255    
    inverted_cleaned = cv2.bitwise_not(cleaned)    
    return inverted_cleaned
# ----------------------
# FUNCTION TO PROCESS IMAGE
# ----------------------
def hair_detection(image: np.ndarray) -> dict:
    """
    Processes an image to detect hair, merge bounding boxes, and overlay zoomed-in views.
    Args:
        image (np.ndarray): Input image in RGB format (height, width, channels).

    Returns:
        dict: Contains 'Result' (status) and 'annotated_image' (final processed image).
    """

    print(type(image))
    start_time = time.time()
    naan_type = None

    if image.shape[0] * image.shape[1] < 4000000:
        # A mini naan spotted
        IMG_SIZE = 1824
        PIECES = 3
        naan_type = "MINI"
    else:
        IMG_SIZE = 2400
        PIECES = 3
        naan_type = "MAXI"

    print(naan_type)
    # cv2.imwrite("received_img_in_hair.png", image)
    # mask_Unet = infer_single(image)
    # mask_Unet = cv2.resize(mask_Unet, (IMG_SIZE,IMG_SIZE))

    # input_image = cv2.imread("received_img_in_hair.png")
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_image = cv2.resize(image_RGB, (IMG_SIZE, IMG_SIZE))  # Resize image to fit the model input
    image_bgr  = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) 
    pred_patches = []

    # Process 3x3 patches
    for patch_idx in range(PIECES * PIECES):
        row = patch_idx // PIECES
        col = patch_idx % PIECES
        y1, y2 = row * PATCH_SIZE, (row + 1) * PATCH_SIZE
        x1, x2 = col * PATCH_SIZE, (col + 1) * PATCH_SIZE

        patch = orig_image[y1:y2, x1:x2]
        # resized_patch = cv2.resize(patch, (1024, 1024)).astype(np.float32) / 255.0
        resized_patch = patch.astype(np.float32)/255.0
        resized_patch = (resized_patch - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        # resized_patch = (patch - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor_patch = torch.tensor(resized_patch, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        pred = torch.sigmoid(model(tensor_patch)) > 0.90
        pred_patch = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
        pred_patch = cv2.resize(pred_patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
        pred_patches.append(pred_patch)

    full_pred = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for patch_idx, patch in enumerate(pred_patches):
        row = patch_idx // PIECES
        col = patch_idx % PIECES
        y1, y2 = row * PATCH_SIZE, (row + 1) * PATCH_SIZE
        x1, x2 = col * PATCH_SIZE, (col + 1) * PATCH_SIZE
        full_pred[y1:y2, x1:x2] = patch

    # full_pred = cv2.bitwise_or(full_pred, mask_Unet)

    mask_cliantro = cliantro(image)           # uint8 mask, 0/255
    mask_cliantro = cv2.resize(mask_cliantro, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    full_pred = cv2.bitwise_and(full_pred, full_pred, mask=mask_cliantro)

    groove_mask = detect_grooves(image) 
    groove_mask = cv2.resize(groove_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    full_pred = cv2.bitwise_and(full_pred, full_pred, mask=groove_mask)

    image_bgr_nobg = remove_bg(image_bgr)   # same size as image_bgr (IMG_SIZE x IMG_SIZE)
    _, binary_mask = extract_binary(image_bgr_nobg, 0)  # returns full-size mask
    kernel = np.ones((50, 50), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

    # ensure type is correct
    binary_mask = binary_mask.astype(np.uint8)

    # apply to prediction
    full_pred = cv2.bitwise_and(full_pred, full_pred, mask=binary_mask)

    if naan_type=="MINI":
        _, mask_edges = extract_foreground_mask_rembg(orig_image)
        full_pred = cv2.bitwise_and(full_pred, full_pred, mask=mask_edges)

    #Save the mask-
    mask = cv2.cvtColor(full_pred, cv2.COLOR_GRAY2BGR)
    ct = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # cv2.imwrite(f"masks_results/mask_{ct}.png", mask)

    contours, _ = cv2.findContours(full_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500:  # Ignore small noise
            x1 = max(x - w // 2, 0)
            y1 = max(y - h // 2, 0)
            x2 = min(x + (3 * w) // 2, IMG_SIZE)
            y2 = min(y + (3 * h) // 2, IMG_SIZE)
            raw_boxes.append((x1, y1, x2, y2))

    merged_boxes = merge_boxes(raw_boxes, threshold=30)

    # ---------- NEW: ZOOM-IN OVERLAY (integrated from first script) ----------
    overlay = orig_image.copy()
    margin = 50  # px between zooms & border
    zoom_scale = 2  # Magnification factor
    zoom_y = margin  # current vertical offset for next zoom

    for box_id, (x1, y1, x2, y2) in enumerate(merged_boxes):
        # Draw bounding box on main image
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Crop & enlarge the detected hair region
        crop = overlay[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        zoom = cv2.resize(crop, ((x2 - x1) * zoom_scale, (y2 - y1) * zoom_scale), interpolation=cv2.INTER_LINEAR)

        # Compute paste position (right-hand side, stacked vertically)
        px = IMG_SIZE - zoom.shape[1] - margin
        py = zoom_y

        # Stop if the zoom would overflow bottom edge
        if py + zoom.shape[0] > IMG_SIZE:
            print(f"Skipping zoom for box #{box_id+1}: not enough vertical space.")
            continue

        overlay[py:py + zoom.shape[0], px:px + zoom.shape[1]] = zoom
        # Red frame around zoomed view
        cv2.rectangle(overlay, (px, py), (px + zoom.shape[1], py + zoom.shape[0]), (0, 0, 255), 6)

        zoom_y += zoom.shape[0] + margin

    combined_array = np.concatenate((orig_image, overlay, mask), axis=1)
    ct = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 3. Save the final, combined array to a file using cv2.imwrite.
    # cv2.imwrite(f"Combined_TestResults/combined_image_{ct}.png", cv2.cvtColor(combined_array, cv2.COLOR_RGB2BGR))

    # Add padding (same as in code 1)
    PAD_TOP, PAD_BOTTOM, PAD_LR = 100, 100, 60
    annotated_padded = cv2.copyMakeBorder(
        overlay,
        PAD_TOP, PAD_BOTTOM, PAD_LR, PAD_LR,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # Text strings for Header and Result
    HEADER_TEXT = "HAIR ANALYSIS"
    RESULT_TEXT = "OK" if len(merged_boxes) == 0 else "NOT OK"

    # Colors for the text (same as code 1)
    HEADER_COLOR = (255, 255, 255)  # white
    RESULT_COLOR = (0, 255, 0) if RESULT_TEXT == "OK" else (0, 0, 255)  # green for OK, red for NOT OK

    # Helper function to draw centered text (same as in code 1)
    def put_centered(img, text, y, color, scale, thickness):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, _), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (img.shape[1] - w) // 2
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    # Draw header text in the top padding
    put_centered(
        annotated_padded,
        HEADER_TEXT,
        y=PAD_TOP - 10,  # 10 px down from very top
        color=HEADER_COLOR,
        scale=2.0,        # larger font
        thickness=3
    )

    # Draw the result text (OK/NOT OK) in the bottom padding
    put_centered(
        annotated_padded,
        RESULT_TEXT,
        y=annotated_padded.shape[0] - 10,  # 10 px above the bottom
        color=RESULT_COLOR,
        scale=2.0,        # larger font
        thickness=4
    )

    # Timer end (for completion time)

    print("Hair inference time:", time.time() - start_time)
    # Final result including the annotated image
    annotated_padded = cv2.cvtColor(annotated_padded, cv2.COLOR_RGB2BGR)

    result = {
        "Result": RESULT_TEXT,
        "annotated_image": annotated_padded
    }

    print("Hair_Inference_Completed")
    return result


# def process_images(input_folder, output_folder):
#     """
#     Processes all images in the input folder, applies the hair detection model,
#     and saves the annotated results to the output folder.
    
#     Args:
#         input_folder (str): Path to the input directory containing images.
#         output_folder (str): Path to the output directory to save annotated images.
#     """
#     # Ensure output folder exists
#     Path(output_folder).mkdir(parents=True, exist_ok=True)
    
#     # List all image files in the input folder
#     image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

#     for image_file in image_files:
#         # Read image
#         input_image_path = os.path.join(input_folder, image_file)
#         input_image = cv2.imread(input_image_path)

#         # Apply hair detection
#         result = hair_detection(input_image)

#         # Save the annotated image (in RGB format)
#         output_image_path = os.path.join(output_folder, f"annotated_{image_file}")
#         # annotated_image_bgr = cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR)
#         cv2.imwrite(output_image_path, result["annotated_image"])

#         print(f"Processed {image_file} and saved the result to {output_image_path}")

# if __name__ == "__main__":
#     # Set paths for input and output directories
#     input_folder = r"C:\Users\singh\Downloads\Grooves_Pro\MiniNaansData"  # Path to the input folder
#     output_folder = r"C:\Users\singh\Downloads\Grooves_Pro\MiniNaansData\Results_600X600_on_1800_EDGE"  # Path to the output folder
    
#     # Process all images in the input folder and save the results in the output folder
#     process_images(input_folder, output_folder)

# path = "/home/itcfoods_collablens/Desktop/CODE/captures_cropped/capture_20250915_113447.png"
# im = cv2.imread(path, cv2.IMREAD_COLOR)
# # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 

# r = hair_detection(im)
# anno = r["annotated_image"]
# cv2.imwrite("result2.png", anno)