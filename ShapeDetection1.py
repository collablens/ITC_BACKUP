import cv2
import time
from PIL import Image
import numpy as np
from rembg import remove, new_session
import onnxruntime as ort
import math
import os

u2p_session = new_session("u2netp")
#os.makedirs("thresholded_images", exist_ok = True)
#os.makedirs("GrayScaled_Images", exist_ok = True)
#os.makedirs("Rotated_image", exist_ok = True)
#os.makedirs("Received", exist_ok=True)
#os.makedirs("results", exist_ok=True)
#os.makedirs("Glare_Images", exist_ok=True)


def process_single_image(image_bgr):
    print("inside shape")
    # Threshold constants
    MAX_ANGLE = 30        # degrees (left-right angle difference)
    MAX_KINK = 30         # degrees (absolute change between consecutive angles)
    MAX_FOLD = -30     # degrees (minimum allowable signed change)
    MAX_DENTS = 50       # maximum allowable base dents
    TEMPLATE_BIG = r"IMG_0513.png"
    TEMPLATE_MINI = r"IMG_0590.png"

    # Dictionary to record timing for each function
    times = {
        "extract_binary": [],
        "get_largest_contour": [],
        "find_tip": [],
        "find_base_points": [],
        "fit_line": [],
        "perpendicular_line": [],
        "rotate_image": [],
        "rotate_points": [],
        "sample_extrema_from_binary_image": [],
        "compute_angles": [],
        "show_base_curve_with_stats": [],
        "process_image": [],
        "align_naan_by_template_matching_fast": [],
        "Total_Rotation_Time": []
    }

    def process_images_for_triangle_check(contour, threshold=0.65):
        
        retval, triangle = cv2.minEnclosingTriangle(contour)
        triangle = np.intp(triangle).reshape((3, 2))

        contour_area = cv2.contourArea(contour)
        triangle_area = cv2.contourArea(triangle)
        fill_ratio = contour_area / triangle_area if triangle_area > 0 else 0
        passed = "PASSED" if fill_ratio >= threshold else "FAILED"

        return passed

        # ────────────────────────────────────────────────────────────────
    #  1. Utility – make a colour canvas that mirrors the mask canvas
    # ────────────────────────────────────────────────────────────────
    def _make_bgr_canvas(img_bgr: np.ndarray,
                        mask: np.ndarray,
                        padding: int = 10) -> np.ndarray:
        """Centre-crop `img_bgr` by the mask’s bbox and paste onto a
        black square canvas sized identically to the mask canvas."""
        # bbox of white pixels in the binary mask
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        crop = img_bgr[y:y + h, x:x + w]

        max_dim   = max(h, w)
        canvas_sz = int(math.ceil(math.sqrt(2) * max_dim)) + 2 * padding
        canvas    = np.zeros((canvas_sz, canvas_sz, 3), dtype=img_bgr.dtype)

        cy, cx = canvas_sz // 2, canvas_sz // 2
        canvas[cy - h // 2: cy - h // 2 + h,
            cx - w // 2: cx - w // 2 + w] = crop
        return canvas


    # ────────────────────────────────────────────────────────────────
    #  2. Helper – minimal template matcher that ALSO returns matrix
    # ────────────────────────────────────────────────────────────────
    def _template_align_and_matrix(area,
                                mask,
                                boundary,
                                scale_factor=0.13,
                                coarse_step=5,
                                fine_range=5,
                                template_big=TEMPLATE_BIG,
                                template_small=TEMPLATE_MINI):
        """A trimmed-down copy of your template matcher that
        additionally returns the 2×3 affine matrix for reuse."""
        # --- load + centre template (reuse existing helpers) ----------
        tpl_path = template_small  # big / mini choice simplified
        tpl_mask = cv2.imread(tpl_path, cv2.IMREAD_UNCHANGED)
        _, tpl_mask = extract_binary(tpl_mask, 1)
        tpl_mask   = prepare_canvas_with_centered_mask(tpl_mask)

        # --- scale both masks down ------------------------------------
        def _down(img):
            return cv2.resize(img, None, fx=scale_factor, fy=scale_factor,
                            interpolation=cv2.INTER_NEAREST)
        tpl_s  = _down(tpl_mask)
        inp_s  = _down(mask)

        # --- place on equal-sized canvases ----------------------------
        h_t, w_t = tpl_s.shape
        h_i, w_i = inp_s.shape
        diag     = int(np.sqrt(max(h_t, h_i) ** 2 + max(w_t, w_i) ** 2) * 1.5)
        cx = cy  = diag // 2
        def _to_canvas(sm):
            can = np.zeros((diag, diag), np.uint8)
            h, w = sm.shape
            can[cy - h // 2: cy - h // 2 + h,
                cx - w // 2: cx - w // 2 + w] = sm
            return can
        tpl_c = _to_canvas(tpl_s)
        inp_c = _to_canvas(inp_s)

        def _iou(a, b):
            inter = cv2.bitwise_and(a, b)
            union = cv2.bitwise_or(a, b)
            return np.sum(inter) / (np.sum(union) + 1e-6)

        best_ang  = 0
        best_iou  = -1
        for ang in range(0, 360, coarse_step):
            M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
            rot = cv2.warpAffine(inp_c, M, (diag, diag),
                                flags=cv2.INTER_NEAREST)
            iou = _iou(tpl_c, rot)
            if iou > best_iou:
                best_iou, best_ang = iou, ang

        for off in range(-fine_range, fine_range + 1):
            ang = (best_ang + off) % 360
            M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
            rot = cv2.warpAffine(inp_c, M, (diag, diag),
                                flags=cv2.INTER_NEAREST)
            iou = _iou(tpl_c, rot)
            if iou > best_iou:
                best_iou, best_ang = iou, ang

        print("Best IOU found is: {best_iou}")
        # --- final affine on the *original-res* mask ------------------
        h0, w0 = mask.shape
        M0 = cv2.getRotationMatrix2D((w0 // 2, h0 // 2), best_ang, 1.0)
        mask_rot  = cv2.warpAffine(mask,     M0, (w0, h0),
                                flags=cv2.INTER_NEAREST)
        bound_rot = cv2.warpAffine(boundary, M0, (w0, h0),
                                flags=cv2.INTER_NEAREST)
        return mask_rot, bound_rot, M0


    # ────────────────────────────────────────────────────────────────
    #  3. Public API – one call, colour out
    # ────────────────────────────────────────────────────────────────
    def rotate_original_like_mask(frame_bgr: np.ndarray) -> np.ndarray:
        """
        Rotate & pad `frame_bgr` **exactly** as the final binary mask
        produced inside `process_single_image`.
        Returns the aligned colour image.
        """
        # A. reproduce early pipeline, but keep colour
        bgr_nobg       = remove_bg(frame_bgr)
        _, mask_raw    = extract_binary(bgr_nobg, t=99)           # t=99 ⇒ no debug files
        mask_canvas    = prepare_canvas_with_centered_mask(mask_raw)
        colour_canvas  = _make_bgr_canvas(bgr_nobg, mask_raw)

        boundary       = extract_boundary(mask_canvas)
        _, area        = get_largest_contour(boundary)

        # B. coarse + fine template rotation
        mask_rot1, bound_rot1, M_tpl = _template_align_and_matrix(area,
                                                                mask_canvas,
                                                                boundary)

        # C. second pass – base line horizontal
        contour, _     = get_largest_contour(mask_rot1)
        contour    = contour.reshape(-1, 2).astype(np.float32)
        base_pts       = find_base_points(contour)
        m_base, _      = fit_line(base_pts)
        ang_base       = np.degrees(np.arctan(m_base))

        h, w           = mask_rot1.shape
        M_base         = cv2.getRotationMatrix2D((w // 2, h // 2), ang_base, 1.0)

        # D. compose the two affine matrices
        M_tpl_3  = np.vstack([M_tpl,  [0, 0, 1]])
        M_base_3 = np.vstack([M_base, [0, 0, 1]])
        M_total  = (M_base_3 @ M_tpl_3)[:2, :]

        # E. final warp on the colour canvas
        aligned = cv2.warpAffine(colour_canvas, M_total, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
        return aligned


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

        if t==0:
            cv2.imwrite("Glare_Images/original.png", image_bgr)

        st = time.time()

        # 1. Gray + fixed threshold -------------------------------------------------
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if gray is None or gray.size == 0:
            raise ValueError("Image could not be loaded or is empty.")

        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        # 2. Find all external contours and keep the largest ------------------------
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Threshold mask is empty – no foreground found.")

        largest = max(contours, key=cv2.contourArea)

        # 3. Build a clean mask containing ONLY that contour ------------------------
        clean_mask = np.zeros_like(thresh)
        cv2.drawContours(clean_mask, [largest], contourIdx=-1,
                        color=255, thickness=cv2.FILLED)

        # 4. Bounding box of that contour (+ padding, clamped to image size) --------
        x, y, w, h = cv2.boundingRect(largest)
        H, W = gray.shape
        x0, y0, x1, y1 = x, y, x+w, y+h
    
        # 5. Crop gray image and binary mask ----------------------------------------
        gray_crop = gray[  y0:y1, x0:x1 ]
        mask_crop = clean_mask[y0:y1, x0:x1]

        # 6. (Optional) write debug images ------------------------------------------
        if t==0:
            cv2.imwrite("GrayScaled_Images/image.png", gray_crop)
            cv2.imwrite("thresholded_images/image.png", mask_crop)

        times['extract_binary'].append(time.time() - st)
        return gray_crop, mask_crop # bbox is handy if you need it
        
    def extract_boundary(binary_mask):

        if binary_mask is None:
            raise ValueError("Empty Binary Mask")
        kernel = np.ones((3, 3), np.uint8)
        boundary = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)
        return boundary

    def prepare_canvas_with_centered_mask(binary_mask, padding=10):

        if binary_mask is None:
            raise ValueError("Empty Binary_Mask")

        # Step 1: Find bounding box of the white region
        coords = cv2.findNonZero(binary_mask)
        x, y, w, h = cv2.boundingRect(coords)

        # Crop the object
        cropped = binary_mask[y:y+h, x:x+w]

        # Step 2: Compute square canvas size
        max_dim = max(h, w)
        canvas_size = int(math.ceil(math.sqrt(2) * max_dim)) + 2 * padding

        # Step 3: Calculate center of cropped mask
        center_y, center_x = h // 2, w // 2

        # Step 4: Create new canvas and compute paste coordinates
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        canvas_center_y, canvas_center_x = canvas_size // 2, canvas_size // 2

        y_start = canvas_center_y - center_y
        x_start = canvas_center_x - center_x

        # Step 5: Paste cropped mask on the canvas
        canvas[y_start:y_start+h, x_start:x_start+w] = cropped

        return canvas

    def _load_and_prep_template(template_path):

        template_full = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

        if template_full is None:
            raise FileNotFoundError(f"Template image '{template_path}' not found.")

        h_full, w_full = template_full.shape[:2]
        _, template_full = extract_binary(template_full, 1)
        template_full = prepare_canvas_with_centered_mask(template_full)

        if np.sum(template_full) < 10:
            raise ValueError("Failed to create a valid binary mask from template.")

        return template_full

    def _calculate_iou(mask1, mask2):

        """Calculates Intersection over Union (IoU) between two binary masks."""
        intersection = cv2.bitwise_and(mask1, mask2)
        union = cv2.bitwise_or(mask1, mask2)
        iou = np.sum(intersection) / (np.sum(union) + 1e-6)
        return iou

    def align_naan_by_template_matching_fast(area, binary_mask, boundary,
                                                template_path_big=TEMPLATE_BIG,
                                                template_path_mini=TEMPLATE_MINI,
                                                scale_factor=0.13,  #size reduction to reduce pixels
                                                coarse_step=5,   # angle range it will check on 45, 90,... if coarse_step=45
                                                fine_range=5):    # +- in coarse step angle to check in case get better allign
        st = time.time()

        try:
            # template_mask_full = _load_and_prep_template(template_path_big if area > 2000000 else template_path_mini)
            template_mask_full = _load_and_prep_template(template_path_mini)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error preparing template: {e}")
            return binary_mask, boundary

        def _normalise_scale(input_mask, template_mask):
            h_i, w_i = input_mask.shape
            h_t, w_t = template_mask.shape

            # pick the longer dimension for each
            long_i = max(h_i, w_i)
            long_t = max(h_t, w_t)

            scale = long_t / long_i
            if abs(scale - 1.0) < 1e-3:         # already the same size
                return input_mask, 1.0

            new_w = int(round(w_i * scale))
            new_h = int(round(h_i * scale))
            resized = cv2.resize(
                input_mask, (new_w, new_h),
                interpolation=cv2.INTER_NEAREST
            )
            return resized, scale

        input_small, scale_used = _normalise_scale(binary_mask, template_mask_full)
        # --- Downscale Masks ---
        template_small = cv2.resize(template_mask_full, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        input_small = cv2.resize(input_small, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        # template_small = template_mask_full

        # --- Find Centroids (on small masks) ---
        M_template = cv2.moments(template_small)
        if M_template['m00'] == 0:
            print("Warning: Template mask empty after scaling.")
            return binary_mask, boundary
        ct_x_s = int(M_template['m10'] / M_template['m00'])
        ct_y_s = int(M_template['m01'] / M_template['m00'])

        M_input = cv2.moments(input_small)
        if M_input['m00'] == 0:
            print("Warning: Input mask empty after scaling.")
            return binary_mask, boundary
        ci_x_s = int(M_input['m10'] / M_input['m00'])
        ci_y_s = int(M_input['m01'] / M_input['m00'])

        # --- Create Small Canvases ---
        h_t_s, w_t_s = template_small.shape
        h_i_s, w_i_s = input_small.shape
        diag_s = np.sqrt(max(h_t_s, h_i_s)**2 + max(w_t_s, w_i_s)**2)
        canvas_size_s = int(diag_s * 1.5)
        canvas_center_x_s, canvas_center_y_s = canvas_size_s // 2, canvas_size_s // 2
        canvas_center_s = (canvas_center_x_s, canvas_center_y_s)

        template_canvas_s = np.zeros((canvas_size_s, canvas_size_s), dtype=np.uint8)
        template_canvas_s[canvas_center_y_s - ct_y_s : canvas_center_y_s - ct_y_s + h_t_s,
                        canvas_center_x_s - ct_x_s : canvas_center_x_s - ct_x_s + w_t_s] = template_small

        input_canvas_s = np.zeros((canvas_size_s, canvas_size_s), dtype=np.uint8)
        input_canvas_s[canvas_center_y_s - ci_y_s : canvas_center_y_s - ci_y_s + h_i_s,
                    canvas_center_x_s - ci_x_s : canvas_center_x_s - ci_x_s + w_i_s] = input_small

        
        def debug_print(template_canvas_s: np.ndarray,
                        rotated_input:     np.ndarray,
                        angle:             float,
                        iou:               float) -> None:
        
            # os.makedirs(out_dir, exist_ok=True)

            # --- build a colour overlay ----------------------------------------------
            overlay = np.zeros((*template_canvas_s.shape, 3), dtype=np.uint8)
            overlay[..., 2] = template_canvas_s            # red   = template
            overlay[..., 1] = rotated_input                # green = candidate

            # --- annotate angle + IoU -------------------------------------------------
            text = f"{angle:.1f}° | IoU {iou:.3f}"
            cv2.putText(
                overlay, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            # --- write to disk --------------------------------------------------------
            fname = f"ang_{int(round(angle)) :03d}_iou_{iou:.3f}.png"
            # cv2.imwrite(os.path.join(out_dir, fname), overlay)


        # --- Coarse Search ---
        best_iou = -1.0
        best_angle = 0.0
        for angle_int in range(0, 360, coarse_step):
            angle = float(angle_int)
            M_rot = cv2.getRotationMatrix2D(canvas_center_s, angle, 1.0)
            rotated_input = cv2.warpAffine(input_canvas_s, M_rot, (canvas_size_s, canvas_size_s), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            iou = _calculate_iou(template_canvas_s, rotated_input)
            debug_print(template_canvas_s, rotated_input, angle_int, iou)
            if iou > best_iou:
                best_iou = iou
                best_angle = angle

        # --- Fine Search ---
        for angle_offset in range(-fine_range, fine_range + 1):
            angle = (best_angle + angle_offset) % 360.0 # Handle wrap around
            M_rot = cv2.getRotationMatrix2D(canvas_center_s, angle, 1.0)
            rotated_input = cv2.warpAffine(input_canvas_s, M_rot, (canvas_size_s, canvas_size_s), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            iou = _calculate_iou(template_canvas_s, rotated_input)
            if iou > best_iou:
                best_iou = iou
                best_angle = angle

        print(f"Best angle found: {best_angle:.2f} degrees with IoU: {best_iou:.4f}")

        # --- Final Rotation (Full Res) ---
        # M_input_orig = cv2.moments(binary_mask)
        # if M_input_orig['m00'] == 0:
        #     print("Warning: Original input mask empty.")
        #     return image, binary_mask, boundary
        # ci_x_orig = int(M_input_orig['m10'] / M_input_orig['m00'])
        # ci_y_orig = int(M_input_orig['m01'] / M_input_orig['m00'])
        h_orig, w_orig = binary_mask.shape[:2]
        ci_x_orig = h_orig // 2
        ci_y_orig = w_orig // 2
        M_final = cv2.getRotationMatrix2D((ci_x_orig, ci_y_orig), best_angle, 1.0)

        # rotated_image = cv2.warpAffine(image, M_final, (w_orig, h_orig), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_mask = cv2.warpAffine(binary_mask, M_final, (w_orig, h_orig), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_boundary = cv2.warpAffine(boundary, M_final, (w_orig, h_orig), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        et = time.time()
        times["align_naan_by_template_matching_fast"].append(et-st)

        return rotated_mask, rotated_boundary


    def get_largest_contour(boundary_img):

        st = time.time()

        contours, _ = cv2.findContours(boundary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours found in image.")
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        et = time.time()
        times['get_largest_contour'].append(et - st)
        return largest, area


    def find_tip(binary_image):

        st = time.time()

        rows = np.any(binary_image > 0, axis=1)
        rows_with_pixel = np.where(rows)[0]

        if len(rows_with_pixel) == 0:
            raise ValueError("No visible pixels found in the image.")

        tip_y = rows_with_pixel[0]
        tip_row = binary_image[tip_y]

        tip_x_coords = np.where(tip_row > 0)[0]
        tip_x = int(np.mean(tip_x_coords))

        et = time.time()
        times['find_tip'].append(et - st)
        return np.array([tip_x, tip_y])


    def find_base_points(contour, horiz_thresh=10, bottom_frac=0.3):

        st = time.time()

        diffs = np.roll(contour, -1, axis=0) - contour
        angles = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
        is_horiz = (np.abs(angles) < horiz_thresh) | (np.abs(np.abs(angles) - 180) < horiz_thresh)
        ys = contour[:, 1]
        min_y, max_y = np.min(ys), np.max(ys)
        is_bottom = ys >= min_y + (1 - bottom_frac) * (max_y - min_y)
        base_mask = is_horiz & is_bottom

        et = time.time()
        times['find_base_points'].append(et - st)
        return contour[base_mask]


    def fit_line(points):

        st = time.time()

        x = points[:, 0]
        y = points[:, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        et = time.time()
        times['fit_line'].append(et - st)
        return m, b

    def perpendicular_line(m, tip):

        st = time.time()

        if m == 0:
            et = time.time()
            times['perpendicular_line'].append(et - st)
            return None, tip[0]

        elif np.isinf(m):
            et = time.time()
            times['perpendicular_line'].append(et - st)
            return 0, tip[1]

        m_perp = -1 / m
        b_perp = tip[1] - m_perp * tip[0]

        et = time.time()
        times['perpendicular_line'].append(et - st)
        return m_perp, b_perp

    def rotate_image(image, angle):

        st = time.time()

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

        et = time.time()
        times['rotate_image'].append(et - st)
        return rotated, M

    def rotate_points(points, M):

        st = time.time()

        points = np.array(points, dtype=np.float32)
        if points.ndim == 1:
            points = np.expand_dims(points, axis=0)
        points = np.hstack([points, np.ones((points.shape[0], 1))])
        rotated = (M @ points.T).T

        et = time.time()
        times['rotate_points'].append(et - st)
        return rotated

    def sample_extrema_from_binary_image(boundary, tip, num_lines=7, tolerance=10):

        st = time.time()

        max_y = np.max(boundary[:, 1])
        y_start = tip[1] + int((max_y - tip[1]) * 0.1)
        y_end = max_y - int((max_y - tip[1]) * 0.1)
        y_levels = np.linspace(y_start, y_end, num=num_lines + 1, dtype=int)
        boundary_x = boundary[:, 0]
        boundary_y = boundary[:, 1]
        left_points = []
        right_points = []
        for y in y_levels:
            mask = np.abs(boundary_y - y) <= tolerance
            candidates = boundary[mask]
            if candidates.shape[0] == 0:
                continue
            left_mask = candidates[:, 0] < tip[0]
            right_mask = ~left_mask
            if np.any(left_mask):
                left_candidates = candidates[left_mask]
                left_points.append(tuple(left_candidates[np.argmin(left_candidates[:, 0])]))
            if np.any(right_mask):
                right_candidates = candidates[right_mask]
                right_points.append(tuple(right_candidates[np.argmax(right_candidates[:, 0])]))

        et = time.time()
        times['sample_extrema_from_binary_image'].append(et - st)
        return left_points, right_points, y_levels

    def compute_angles(points, side):

        st = time.time()

        angles = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            if side == "right":
                angle = angle % 360
            elif side == "left":
                angle = (angle + 180) % 360
            if angle > 180:
                angle = 360 - angle
            angles.append(round(angle, 2))

        et = time.time()
        times['compute_angles'].append(et - st)
        return angles

    def show_base_curve_with_stats(contour,
                                area,
                                deg: int = 2,
                                cluster_eps: int = 1):
        """
        Reports:
        • n_dents – count of bottom-edge points whose deviation > depth_thresh
        """
        st = time.time()
        bottom_frac = 0.075 if area>2000000 else 0.05
        depth_thresh = 20 if area>2000000 else 20

        cnt = np.asarray(contour).reshape(-1, 2).astype(np.int32)
        y_min, y_max = cnt[:, 1].min(), cnt[:, 1].max()
        band_y = y_max - bottom_frac * (y_max - y_min)
        base_pts = cnt[cnt[:, 1] >= band_y]
        if len(base_pts) <= deg:
            raise ValueError("Not enough base points to fit polynomial.")
        base_pts = base_pts[np.argsort(base_pts[:, 0])]
        xs, ys = base_pts[:, 0], base_pts[:, 1]
        coeffs = np.polyfit(xs, ys, deg)

        band_y1 = y_max - 0.2 * (y_max - y_min)
        base_pts1 = cnt[cnt[:, 1] >= band_y1]
        base_pts1 = base_pts1[np.argsort(base_pts1[:, 0])]
        xs1, ys1 = base_pts1[:, 0], base_pts1[:, 1]
        y_fit = np.polyval(coeffs, xs1)

        residual = ys1 - y_fit
        over = residual > depth_thresh
        under = residual < -depth_thresh
        dent_mask = over | under
        dent_pts = base_pts1[dent_mask]

        # Filter out dent points in the leftmost and rightmost 20%
        x_min, x_max = xs1.min(), xs1.max()
        x_left_thresh = x_min + 0.15 * (x_max - x_min)
        x_right_thresh = x_max - 0.15 * (x_max - x_min)
        valid_dent_pts = dent_pts[(dent_pts[:, 0] >= x_left_thresh) & (dent_pts[:, 0] <= x_right_thresh)]

        clusters = []
        remaining = valid_dent_pts.copy()
        while len(remaining):
            seed = remaining[0]
            dists = np.linalg.norm(remaining - seed, axis=1)
            grp = remaining[dists < cluster_eps]
            clusters.append(tuple(np.mean(grp, axis=0)))
            remaining = remaining[dists >= cluster_eps]
        n_dents = len(clusters)

        changes = 0
        if clusters:
            first_x = int(round(clusters[0][0]))
            first_y = clusters[0][1]
            crnt_dir = first_y > np.polyval(coeffs, first_x)

            for x_f, y_f in clusters:
                x_f_int = int(round(x_f))
                y_f_fit = np.polyval(coeffs, x_f_int)
                new_dir = y_f > y_f_fit
                if new_dir != crnt_dir:
                    changes += 1
                    crnt_dir = new_dir

        et = time.time()
        times['show_base_curve_with_stats'].append(et - st)
        return n_dents, xs, y_fit, clusters, coeffs, changes

    def process_image(image_bgr):

        st = time.time()
        cv2.imwrite("Received/img.png", image_bgr)
        image_bgr = remove_bg(image_bgr)
        image_gray, binary_mask = extract_binary(image_bgr, 0)
        binary_mask = prepare_canvas_with_centered_mask(binary_mask)
        boundary = extract_boundary(binary_mask)
        _, area = get_largest_contour(boundary)
        print("I have reached Step 1")

        total_rotation_time = time.time()
        binary_mask, boundary = align_naan_by_template_matching_fast(area, binary_mask, boundary)
        contour, _ = get_largest_contour(binary_mask)
        print("I have reached Step 2")

        if contour is None:
            raise ValueError("Contour couldnt be detected")
        
        contour = contour.squeeze().astype(np.float32)

        base_points = find_base_points(contour)
        if len(base_points) < 2:
            raise ValueError("Base Points detected")

        # cv2.imwrite("Rotated_image/image_mask_before.png", binary_mask)
        # cv2.imwrite("Rotated_image/image_boundary_before.png", boundary)

        m_base, b_base = fit_line(base_points)
        angle_rad = math.atan(m_base)
        angle_deg = np.degrees(angle_rad)
        print(angle_deg)
        stacked = np.stack([binary_mask, boundary], axis=2)  # shape: H×W×3

        # Compute rotation matrix M once:
        h, w = binary_mask.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        rotated_all = cv2.warpAffine(stacked, M, (w, h), flags=cv2.INTER_LINEAR)

        # rotated_image = rotated_all[:, :, 0]
        rotated_mask  = rotated_all[:, :, 0]
        rotated_boundary = rotated_all[:, :, 1]
        # rotated_gray = cv2.warpAffine(image_gray, M, (w, h), flags=cv2.INTER_LINEAR)
        # cv2.imwrite(output_path, rotated_mask)
        # cv2.imwrite("Rotated_image/image_boundary.png", rotated_boundary)

        rotated_contour = rotate_points(contour, M)
        print("I have reached Step 3")

        times["Total_Rotation_Time"].append(time.time() - total_rotation_time)

        status_angles = "OK"

        triangle_check = process_images_for_triangle_check(rotated_contour.astype(np.int32))
        if triangle_check == "FAILED":
            status_angles = "NOT OK"

        # Find tip in rotated mask and perpendicular axis
        tip = find_tip(rotated_mask)
        m_perp, b_perp = perpendicular_line(0, tip)

        # Sample left/right edge points and compute angles
        boundary_points = np.argwhere(rotated_boundary > 0)
        boundary_points = np.array([[pt[1], pt[0]] for pt in boundary_points], dtype=np.float32)
        left_pts, right_pts, _ = sample_extrema_from_binary_image(boundary_points, tip)
        left_angles = compute_angles(left_pts, "left")
        right_angles = compute_angles(right_pts, "right")

        # Pad left/right angles to exactly 6 values (if fewer points found, pad with 0)
        if len(left_angles) < 7:
            left_angles += [0] * (7 - len(left_angles))
        if len(right_angles) < 7:
            right_angles += [0] * (7 - len(right_angles))
        L = left_angles[:7]
        R = right_angles[:7]
        angles = L + R  # Combined list of 12 angles

        print("I have reached Step 4")

        # 1. Left-Right angle differences
        angle_diffs = [abs(L[i] - R[i]) for i in range(7)]
        max_angle_diff = max(angle_diffs)

        # 2. Kinks: absolute change between consecutive angles
        to = [1, 3]
        left_kinks = [abs(L[i + 1] - L[i]) for i in range(to[0],to[1])]
        right_kinks = [abs(R[i + 1] - R[i]) for i in range(to[0],to[1])]
        all_kinks = left_kinks + right_kinks
        max_kink = max(all_kinks) if all_kinks else 0

        # 3. Folds: signed change between consecutive angles
        left_folds = [L[i + 1] - L[i] for i in range(1, 3)]
        right_folds = [R[i + 1] - R[i] for i in range(1, 3)]
        all_folds = left_folds + right_folds
        min_fold = min(all_folds) if all_folds else 0

        # Determine intermediate status after angle-based checks
        status_angles = "OK" if (max_angle_diff < MAX_ANGLE and min_fold > MAX_FOLD and max_kink < MAX_KINK and status_angles == "OK") else "NOT OK"
        #status_angles = "OK"

        # Initialize values for base analysis
        n_dents = None
        xs = y_fit = clusters = coeffs = changes = None

        # If angle checks passed, perform base dent analysis
        n_dents, xs, y_fit, clusters, coeffs, changes = show_base_curve_with_stats(rotated_contour, area)

        # Determine final status after checking dents
        final_status = status_angles
        if status_angles == "OK" and (n_dents is not None and n_dents < MAX_DENTS):
            final_status = "OK"
        else:
            final_status = "NOT OK"

        et = time.time()
        times['process_image'].append(et - st)

        print("I have reached annotated step")

        # Convert binary mask to BGR for annotation
        annotated = cv2.cvtColor(rotated_mask, cv2.COLOR_GRAY2BGR)

        # === Dynamic Text + Spacing ===
        h_img, w_img = annotated.shape[:2]
        font_scale_main = h_img / 900.0                # For angles and boxes
        font_scale_info = font_scale_main * 0.7        # For top-left summary
        font_thickness = max(2, int(h_img / 300))      # Increased thickness
        line_spacing = int(h_img / 20)                 # Adaptive line spacing
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Text anchor
        text_x, text_y = 5, 25

        # Colors
        white = (255, 255, 255)
        green = (0, 255, 0)
        red = (0, 0, 255)
        blue = (255, 0, 0)
        yellow = (0, 255, 255)
        orange = (0, 165, 255)
        box_color = yellow  # Yellow for boxes, not cyan

        # --- Draw clusters ---
        if clusters is not None:
            for pt in clusters:
                pt_int = tuple(map(int, pt))
                cv2.circle(annotated, pt_int, 5, red, -1)

            # Draw polynomial curve
            x_curve = np.linspace(0, rotated_mask.shape[1] - 1, 400)
            y_curve = np.polyval(coeffs, x_curve)
            curve_pts = np.column_stack((x_curve, y_curve)).astype(np.int32)
            cv2.polylines(annotated, [curve_pts.reshape(-1, 1, 2)], False, orange, 1)  # thin curve

        print("I have reached step 6")

        # --- Draw angles and connecting lines ---
        angle_i = 0
        pts = left_pts + right_pts
        for i, pt in enumerate(pts):
            color = blue if i < len(left_pts) else green
            pt_int = tuple(map(int, pt))
            cv2.circle(annotated, pt_int, 5, color, -1)

            if i > 0 and i != len(left_pts):
                prev_pt = tuple(map(int, pts[i - 1]))
                cv2.line(annotated, prev_pt, pt_int, color, 2)

            if i == 0 or i == len(left_pts):
                continue

            # Text with outline
            angle_txt = f"{angles[angle_i]:.1f}"
            cv2.putText(annotated, angle_txt, pt_int, font, font_scale_main, (0, 0, 0), font_thickness + 2)
            cv2.putText(annotated, angle_txt, pt_int, font, font_scale_main, color, font_thickness)
            angle_i += 1

        # === Overlay summary text top-left ===
        # summary_lines = [
        #     ("Symmetric" if max_angle_diff < MAX_ANGLE else "Unsymmetric", green if max_angle_diff < MAX_ANGLE else red),
        #     ("Not Bent Inside" if min_fold > MAX_FOLD else "Bent Inside", green if min_fold > MAX_FOLD else red),
        #     (f"Base {'Smooth' if n_dents is not None and n_dents < MAX_DENTS else 'Not Smooth'}", 
        #     green if n_dents is not None and n_dents < MAX_DENTS else red if n_dents is not None else white),
        #     (f"{final_status}", green if final_status == "OK" else red)
        # ]

        # for i, (line_text, color) in enumerate(summary_lines):
        #     y = text_y + i * line_spacing
        #     cv2.putText(annotated, line_text, (text_x, y), font, font_scale_info, color, font_thickness)

        print("I have reached Step 7")

        # === Box for asymmetric region (no cyan used) ===
        for i in range(7):
            if abs(L[i] - R[i]) > MAX_ANGLE:
                ptL, ptR = tuple(map(int, left_pts[i])), tuple(map(int, right_pts[i]))
                ptLA, ptRA = tuple(map(int, left_pts[i+1])), tuple(map(int, right_pts[i+1]))
                x1, y1 = min(ptL[0], ptLA[0]) - 50, min(ptL[1], ptR[1]) - 50
                x2, y2 = max(ptR[0], ptRA[0]) + 50, max(ptLA[1], ptRA[1]) + 50
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 5)
                cv2.putText(annotated, "Asymmetric", (x1, y1 - 10), font, font_scale_main, box_color, font_thickness)
                break

        # === 3. Fold Failures (Bent Inwards) ===
        for i in range(1, 5):
            if i < len(left_pts) - 2 and (L[i+1] - L[i]) < MAX_FOLD:
                pts = [tuple(map(int, left_pts[j])) for j in (i, i+1, i+2)]
                x1, y1 = min(p[0] for p in pts) - 50, min(p[1] for p in pts) - 50
                x2, y2 = max(p[0] for p in pts) + 50, max(p[1] for p in pts) + 50
                cv2.rectangle(annotated, (x1, y1), (x2, y2), red, 5)
                cv2.putText(annotated, "Bent Outwards", (x1, y1 - 10), font, font_scale_main, red, font_thickness)
                break

            if i < len(right_pts) - 2 and (R[i+1] - R[i]) < MAX_FOLD:
                pts = [tuple(map(int, right_pts[j])) for j in (i, i+1, i+2)]
                x1, y1 = min(p[0] for p in pts) - 50, min(p[1] for p in pts) - 50
                x2, y2 = max(p[0] for p in pts) + 50, max(p[1] for p in pts) + 50
                cv2.rectangle(annotated, (x1, y1), (x2, y2), red, 5)
                cv2.putText(annotated, "Bent Outwards", (x1, y1 - 10), font, font_scale_main, red, font_thickness)
                break

        for i in range(1, 5):
            if i < len(left_pts) - 2 and abs(L[i+1] - L[i]) > MAX_KINK:
                pts = [tuple(map(int, left_pts[j])) for j in (i, i+1, i+2)]
                x1, y1 = min(p[0] for p in pts) - 50, min(p[1] for p in pts) - 50
                x2, y2 = max(p[0] for p in pts) + 50, max(p[1] for p in pts) + 50
                cv2.rectangle(annotated, (x1, y1), (x2, y2), red, 5)
                cv2.putText(annotated, "Bent Inwards", (x1, y1 - 10), font, font_scale_main, red, font_thickness)
                break

            if i < len(right_pts) - 2 and abs(R[i+1] - R[i]) > MAX_KINK:
                pts = [tuple(map(int, right_pts[j])) for j in (i, i+1, i+2)]
                x1, y1 = min(p[0] for p in pts) - 50, min(p[1] for p in pts) - 50
                x2, y2 = max(p[0] for p in pts) + 50, max(p[1] for p in pts) + 50
                cv2.rectangle(annotated, (x1, y1), (x2, y2), red, 5)
                cv2.putText(annotated, "Bent Inwards", (x1, y1 - 10), font, font_scale_main, red, font_thickness)
                break

        print("I have reached Step 8")
        # aligned_colour = rotate_original_like_mask(image_bgr)
        # cv2.imwrite("original_bgr_rotated.png", aligned_colour)
        # Padding (pixels)
        PAD_TOP, PAD_BOTTOM, PAD_LR = 60, 60, 30

        # Text strings
        HEADER_TEXT  = "SHAPE ANALYSIS"
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

        # 5) Return as before
        return {
            "Result": final_status,
            "annotated_image": annotated_padded
        }

    return process_image(image_bgr)

# def process_images_in_folder(input_folder, output_folder):
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)
#     os.makedirs(r"c:\Users\singh\Downloads\rough_output", exist_ok=True)

#     # Loop through all files in the input folder
#     for filename in os.listdir(input_folder):
#         # Build full file path
#         input_path = os.path.join(input_folder, filename)

#         # Only process files that look like images
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#             # Read the image in BGR format
#             image_bgr = cv2.imread(input_path)

#             if image_bgr is not None:
#                 # Build output filename path
#                 output_path = os.path.join(output_folder, filename)

#                 # Call the provided processing function
#                 out = process_single_image(image_bgr, output_path)
#                 cv2.imwrite(fr"c:\Users\singh\Downloads\rough_output\{filename}.png", out["annotated_image"])
#             else:
#                 print(f"Warning: Couldn't read image {input_path}")
#         else:
#             print(f"Skipping non-image file: {filename}")

# process_images_in_folder(r"C:\Users\singh\Downloads\complete_shape_dataset", r"C:\Users\singh\Downloads\Rough_Triangle_Templates")
