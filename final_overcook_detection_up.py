import os
from pathlib import Path
import cv2
import numpy as np
from rembg import remove
from PIL import Image
from datetime import datetime
from glob import glob
import math

# ct = 0

def analyze_image(
    image: np.ndarray,
    area_thresh: int = 3000,
    expansion: float = 2.0,
    conc_thresh: float = 23.0,
    overcooked_area_thresh: int = 30000
):

    if image is None or image.size == 0:
        raise ValueError("Empty image passed to analyze_image().")

    annotated_image = image.copy()

    # -------------------------
    # Helpers
    # -------------------------
    def remove_background(image_bgr: np.ndarray) -> np.ndarray:
        """Removes background using rembg and fills the transparent background with white."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).convert("RGBA")
        output_pil = remove(pil_img).convert("RGBA")
        # Convert back to BGR (drop alpha)
        output_bgr = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
        return output_bgr

    def compute_strict_circle(contour):
        """Computes a strict circular approximation for a given contour."""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        distances = [np.linalg.norm([cx - pt[0][0], cy - pt[0][1]]) for pt in contour]
        mean_radius = math.ceil(np.mean(distances))
        return (cx, cy), mean_radius

    def count_non_zero_in_circle(brown_mask, binary_mask, center, radius):
        """Counts non-zero pixels from brown_mask within the circular ROI defined by (center, radius)."""
        circle_mask = np.zeros_like(brown_mask, dtype=np.uint8)
        cv2.circle(circle_mask, center, radius, 255, -1)
        roi_mask = cv2.bitwise_and(binary_mask, circle_mask)
        brown_pixels_in_roi = cv2.bitwise_and(brown_mask, roi_mask)
        non_zero_pixels = cv2.countNonZero(brown_pixels_in_roi)
        total_pixels_in_roi = cv2.countNonZero(roi_mask)
        return non_zero_pixels, total_pixels_in_roi
    
    # global ct

    image_no_bg = remove_background(image)
    # cv2.imwrite(f"masks_overcook/no_bg_{ct}.png", image_no_bg)
    gray = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)   # removes small specks
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  # smooth edges
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # cv2.imwrite(f"masks_overcook/save_binary_{ct}.png", cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, (0, 0, 0), (25, 255, 30))      # Charred areas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    dark_brown_mask = cv2.inRange(hsv, (0, 200, 0), (30, 255, 70))  # Overcooked

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    has_charred = False
    total_overcooked_brown_pixels = cv2.countNonZero(dark_brown_mask)   
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_thresh:
            continue
        
        has_charred = True

        center, radius = compute_strict_circle(cnt)

        if center is None:
            print("center not found")
            continue

        # Inner circle (red): tight around contour
        cv2.circle(annotated_image, center, radius, (0, 0, 255), 2)

    if has_charred:
        print("inside has_charred")
        final_status = "NOT OK"

    elif total_overcooked_brown_pixels > overcooked_area_thresh:
        print("inside total_overcook_brown_pixels")
        final_status = "NOT OK" 

    else:
        print("inside else")
        final_status = "OK"

    # -------------------------
    # Padded banner + bottom-centered status (your style)
    # -------------------------
    PAD_TOP, PAD_BOTTOM, PAD_LR = 60, 60, 30

    HEADER_TEXT  = "OVERCOOKED ANALYSIS"
    RESULT_TEXT  = final_status.strip().upper()

    HEADER_COLOR = (255, 255, 255)                       # white
    RESULT_COLOR = (0, 255, 0) if RESULT_TEXT == "OK" else (0, 0, 255)

    annotated_padded = cv2.copyMakeBorder(
        annotated_image,
        PAD_TOP, PAD_BOTTOM, PAD_LR, PAD_LR,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    def put_centered(img, text, y, color, scale, thickness):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (img.shape[1] - w) // 2
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    # Top heading
    put_centered(
        annotated_padded,
        HEADER_TEXT,
        y=PAD_TOP - 10,
        color=HEADER_COLOR,
        scale=1.5,
        thickness=3
    )

    # Bottom result
    put_centered(
        annotated_padded,
        RESULT_TEXT,
        y=annotated_padded.shape[0] - 10,
        color=RESULT_COLOR,
        scale=2,
        thickness=4
    )

    # ct += 1
    return {
        "Result": final_status,
        "annotated_image": annotated_padded
    }

# if __name__ == "__main__":
#     folder_path = r"C:\Users\singh\OverCookedDetection\AllCasesOvercooked"
#     output_folder = "overcooked_results_charred"
#     os.makedirs(output_folder, exist_ok=True)
#     # os.makedirs("masks_overcook_charred", exist_ok=True)

#     # Loop over all jpg and png images in the folder
#     for image_path in glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.png")):
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"Could not read {image_path}")
#             continue

#         result = analyze_image(img)  # Assuming it returns a dict with "annotated_image"

#         # Save with the same filename in output folder
#         filename = os.path.basename(image_path)
#         save_path = os.path.join(output_folder, f"annotated_{filename}")
#         cv2.imwrite(save_path, result["annotated_image"])
#         print(f"Processed: {filename}")
