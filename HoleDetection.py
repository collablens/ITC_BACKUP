import cv2
import numpy as np
import time
def detect_hole_spots(image):
    ttt=time.time()
    image = cv2.resize(image, (800, 800))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_min = 12
    h_max = 30
    s_min = 11
    s_max = 65
    v_min = 143
    v_max = 193
    total_area = 0
    lower_black = np.array([h_min, s_min, v_min])
    upper_black = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_image, lower_black, upper_black)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 9))
    # print(type(kernel))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # print(type(mask_cleaned))
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = image.copy()
    final_status = "OK"
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        total_area+=area
        x, y, w, h = cv2.boundingRect(cnt)
        if 14 <= w <= 1000 and 14 <= h <= 1000:
            # cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(result_img, (x - w, y - h), (x + 2*w, y + 2*h), (0, 255, 255), 2)
            # cv2.putText(result_img, f"Area: {int(area)}", (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            final_status = "NOT OK"
            
    # Padding (pixels)
    annotated = result_img
    PAD_TOP, PAD_BOTTOM, PAD_LR = 60, 60, 30

    # Text strings
    HEADER_TEXT  = "HOLE ANALYSIS"
    RESULT_TEXT  = str(final_status).strip().upper()

    # Colours (BGR)
    HEADER_COLOR = (255, 255, 255)                    # white
    RESULT_COLOR = (0, 255, 0) if RESULT_TEXT == "OK" else (0, 0, 255)

    # 1) Surround the image with a black border
    annotated_padded = cv2.copyMakeBorder(
        annotated,
        PAD_TOP, PAD_BOTTOM, PAD_LR, PAD_LR,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # 2) Helper: draw centred text
    def put_centered(img, text, y, color, scale, thickness):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, _), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (img.shape[1] - w) // 2
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    # 3) Draw heading in top padding
    put_centered(
        annotated_padded,
        HEADER_TEXT,
        y = PAD_TOP - 10,          # 10 px down from very top
        color = HEADER_COLOR,
        scale = 1.2,               # larger font
        thickness = 3
    )

    # 4) Draw GOOD/BAD in bottom padding
    put_centered(
        annotated_padded,
        RESULT_TEXT,
        y = annotated_padded.shape[0] - 10,   # 10 px above bottom
        color = RESULT_COLOR,
        scale = 1.6,               # even bigger
        thickness = 4
    )
    print(f"{time.time()-ttt:.2f}s")
    # 5) Return as before
    return {
        "Result": final_status,
        "annotated_image": annotated_padded
    }
