# hair_basic_unetpp.py
import os, cv2, math, json, random, numpy as np
from glob import glob
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
from tqdm import tqdm
import datetime

# os.makedirs("result_mask_U2Net", exist_ok=True)

# ========================== CONFIG ==========================
@dataclass
class CFG:
    images_dir: str = "HairRandom"          # RGB images
    masks_dir:  str = "cropped_masks"           # 0/255 single-channel masks (hair=255)
    out_dir:    str = "UnetPP_Inference"        # training artifacts + inference
    tile: int = 1024
    stride: int = 512
    pos_ratio: float = 0.7                        # 70% positive, 30% negative
    epochs: int = 120
    batch_size: int = 8
    lr: float = 6e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    val_split: float = 0.2                        # split by image files
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    IMNET_MEAN = (0.485, 0.456, 0.406)
    IMNET_STD  = (0.229, 0.224, 0.225)
    IMG_SIZE = 2048

ckpt_path = "latest.ckpt"
model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None,
                             in_channels=3, classes=1).to(CFG.device)
state = torch.load(ckpt_path, map_location=CFG.device, weights_only=False)['model']
model.load_state_dict(state)

# ========================== PATCH INDEXING ==========================
def make_grid(h, w, tile, stride) -> List[Tuple[int,int,int,int]]:
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] != h - tile: ys.append(h - tile)
    if xs[-1] != w - tile: xs.append(w - tile)
    coords = [(y, x, y+tile, x+tile) for y in ys for x in xs]
    return coords   # iterates from left to right [row, col]

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

# ========================== TILED INFERENCE ==========================
@torch.no_grad()
def infer_single(img, tile=1024, stride=512):

    # Assumption : Receives a RGB image
    start_time = time.time()
    cv2.imwrite("received_img_in_hair2.png", img)
    input_image = cv2.imread("received_img_in_hair2.png", cv2.IMREAD_COLOR)
    img = cv2.resize(input_image, (CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model.eval()
    H, W, _ = img.shape
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    tf = A.Compose([
        A.Normalize(mean=CFG.IMNET_MEAN, std=CFG.IMNET_STD),
        ToTensorV2()
    ])

    coords = make_grid(H, W, tile, stride)
    for (y0,x0,y1,x1) in coords:
        crop = img[y0:y1, x0:x1]
        if crop.shape[0] != tile or crop.shape[1] != tile:
            pad = np.zeros((tile, tile, 3), dtype=img.dtype)
            pad[:crop.shape[0], :crop.shape[1]] = crop
            crop = pad
        tens = tf(image=crop)['image'].unsqueeze(0).to(CFG.device)
        logit = model(tens)
        prob = torch.sigmoid(logit)[0,0].float().cpu().numpy()
        prob = prob[:(y1-y0), :(x1-x0)]
        acc[y0:y1, x0:x1] += prob
        cnt[y0:y1, x0:x1] += 1.0

    prob = acc / np.clip(cnt, 1e-6, None)
    prob = np.clip(prob, 0.0, 1.0)
    mask = (prob >= 0.1).astype(np.uint8)*255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100:  # Ignore small noise
            x1 = max(x - w // 2, 0)
            y1 = max(y - h // 2, 0)
            x2 = min(x + (3 * w) // 2, CFG.IMG_SIZE)
            y2 = min(y + (3 * h) // 2, CFG.IMG_SIZE)
            raw_boxes.append((x1, y1, x2, y2))

    merged_boxes = merge_boxes(raw_boxes, threshold=30)

    # ---------- NEW: ZOOM-IN OVERLAY (integrated from first script) ----------
    overlay = img.copy()
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
        px = CFG.IMG_SIZE - zoom.shape[1] - margin
        py = zoom_y

        # Stop if the zoom would overflow bottom edge
        if py + zoom.shape[0] > CFG.IMG_SIZE:
            print(f"Skipping zoom for box #{box_id+1}: not enough vertical space.")
            continue

        overlay[py:py + zoom.shape[0], px:px + zoom.shape[1]] = zoom
        # Red frame around zoomed view
        cv2.rectangle(overlay, (px, py), (px + zoom.shape[1], py + zoom.shape[0]), (0, 0, 255), 6)

        zoom_y += zoom.shape[0] + margin

    # Add padding (same as in code 1)
    PAD_TOP, PAD_BOTTOM, PAD_LR = 100, 100, 60
    annotated_padded = cv2.copyMakeBorder(
        overlay,
        PAD_TOP, PAD_BOTTOM, PAD_LR, PAD_LR,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # Text strings for Header and Result
    HEADER_TEXT = "HAIR ANALYSIS 2"
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

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    ct = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"result_mask_U2Net/mask_{ct}.png", mask)

    return result

# path = "/home/itcfoods_collablens/Desktop/CODE/HairTest/capture_20250913_154521.png"
# im = cv2.imread(path, cv2.IMREAD_COLOR)
# # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 

# r = infer_single(im)
# anno = r["annotated_image"]
# cv2.imwrite("result1.png", anno)
