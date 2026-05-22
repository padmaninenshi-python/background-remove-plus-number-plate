import os
import uuid
import base64
import io
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import cv2

try:
    from flask_cors import CORS
    _has_cors = True
except ImportError:
    _has_cors = False

try:
    from rembg import remove as rembg_remove, new_session
    _has_rembg = True
except ImportError:
    _has_rembg = False

app = Flask(__name__)
if _has_cors:
    CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///autolens.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER']    = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

db = SQLAlchemy(app)

class CarImage(db.Model):
    id             = db.Column(db.String(50),  primary_key=True)
    filename       = db.Column(db.String(255))
    original_path  = db.Column(db.String(500))
    processed_path = db.Column(db.String(500))
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()
    os.makedirs(app.config['UPLOAD_FOLDER'],    exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  DETECTION ENGINE  —  5 independent methods, best-score wins
# ═══════════════════════════════════════════════════════════

def _score_candidate(x, y, w, h, img_w, img_h):
    if w <= 0 or h <= 0:
        return 0
    aspect      = w / h
    area_ratio  = (w * h) / (img_w * img_h)
    cx_ratio    = (x + w / 2) / img_w
    cy_ratio    = (y + h / 2) / img_h
    score = 0
    if cy_ratio < 0.55:
        return 0
    if aspect < 1.8 or aspect > 7.0:
        return 0
    if w > img_w * 0.55 or h > img_h * 0.15:
        return 0
    if w < max(60, img_w * 0.06) or h < max(14, img_h * 0.018):
        return 0
    if   3.0 <= aspect <= 5.5:  score += 40
    elif 2.5 <= aspect <= 6.0:  score += 22
    elif 1.8 <= aspect <= 7.0:  score += 8
    if   0.008 <= area_ratio <= 0.07: score += 25
    elif 0.004 <= area_ratio <= 0.12: score += 10
    min_w = max(60, img_w * 0.06)
    max_w = img_w * 0.55
    min_h = max(14, img_h * 0.018)
    max_h = img_h * 0.12
    if min_w <= w <= max_w and min_h <= h <= max_h:
        score += 20
    if   0.70 <= cy_ratio <= 0.95: score += 15
    elif 0.55 <= cy_ratio <= 0.70: score += 5
    if   0.20 <= cx_ratio <= 0.80: score += 8
    elif 0.10 <= cx_ratio <= 0.90: score += 3
    return score


def _method_edge(img_cv, img_w, img_h):
    candidates = []
    gray    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur    = cv2.bilateralFilter(gray, 9, 17, 17)
    edges   = cv2.Canny(blur, 25, 180)
    edges2  = cv2.Canny(blur, 50, 250)
    combined = cv2.bitwise_or(edges, edges2)
    for e in [combined, edges]:
        cnts, _ = cv2.findContours(e, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:40]
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            s = _score_candidate(x, y, w, h, img_w, img_h)
            if s >= 45:
                candidates.append((s, x, y, w, h, 'edge'))
    return candidates


def _method_morph(img_cv, img_w, img_h):
    candidates = []
    gray  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat  = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kern)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blackhat)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed     = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kern)
    closed     = cv2.dilate(closed, close_kern, iterations=1)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        s = _score_candidate(x, y, w, h, img_w, img_h)
        if s >= 45:
            candidates.append((s, x, y, w, h, 'morph'))
    return candidates


def _method_color(img_cv, img_w, img_h):
    candidates = []
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    masks = []
    masks.append(cv2.inRange(hsv, np.array([0,  0,  170]), np.array([180, 45, 255])))
    masks.append(cv2.inRange(hsv, np.array([18, 60, 100]), np.array([38, 255, 255])))
    r1 = cv2.inRange(hsv, np.array([0,  130, 70]),  np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([158,130, 70]),  np.array([180,255, 255]))
    masks.append(cv2.bitwise_or(r1, r2))
    masks.append(cv2.inRange(hsv, np.array([8, 150, 80]),  np.array([22, 255, 255])))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 6))
    for mask in masks:
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  kern, iterations=2)
        m = cv2.dilate(m, kern, iterations=1)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            s = _score_candidate(x, y, w, h, img_w, img_h)
            if s >= 40:
                px = int(w * 0.35); py = int(h * 0.55)
                nx = max(0, x-px);  ny = max(0, y-py)
                nw = min(img_w-nx, w+px*2); nh = min(img_h-ny, h+py*2)
                s2 = _score_candidate(nx, ny, nw, nh, img_w, img_h)
                candidates.append((max(s, s2), nx, ny, nw, nh, 'color'))
    return candidates


def _method_sobel(img_cv, img_w, img_h):
    candidates = []
    gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelX = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    absX   = cv2.convertScaleAbs(sobelX)
    _, thr = cv2.threshold(absX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    thr  = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kern, iterations=2)
    thr  = cv2.dilate(thr, kern, iterations=1)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        s = _score_candidate(x, y, w, h, img_w, img_h)
        if s >= 45:
            candidates.append((s, x, y, w, h, 'sobel'))
    return candidates


def _method_white_rect(img_cv, img_w, img_h):
    candidates = []
    gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    roi_y  = int(img_h * 0.30)
    roi    = gray[roi_y:, :]
    _, thr = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
    kern   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    thr    = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kern, iterations=2)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        y += roi_y
        fill = cv2.countNonZero(thr[y - roi_y:y - roi_y + h, x:x + w]) / max(w * h, 1)
        if fill < 0.35:
            continue
        if w > img_w * 0.60:
            continue
        s = _score_candidate(x, y, w, h, img_w, img_h)
        if s >= 42:
            cx = (x + w / 2) / img_w
            if 0.15 <= cx <= 0.85:
                candidates.append((s, x, y, w, h, 'white_rect'))
    return candidates


def detect_number_plate(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[detect] Cannot read image: {image_path}")
            return None
        img_h, img_w = img.shape[:2]
        print(f"\n🔍 Detecting plate in {img_w}×{img_h} image...")
        all_candidates = []
        all_candidates += _method_edge(img,       img_w, img_h)
        all_candidates += _method_morph(img,      img_w, img_h)
        all_candidates += _method_color(img,      img_w, img_h)
        all_candidates += _method_sobel(img,      img_w, img_h)
        all_candidates += _method_white_rect(img, img_w, img_h)
        if not all_candidates:
            print("[detect] No candidates found")
            return None
        all_candidates.sort(key=lambda c: c[0], reverse=True)
        for i, (s, x, y, w, h, method) in enumerate(all_candidates[:5]):
            print(f"  #{i+1} score={s:3d} [{method:10s}] ({x},{y},{w},{h}) ar={w/max(h,1):.2f}")
        best = all_candidates[0]
        s, x, y, w, h, method = best
        x = max(0, x); y = max(0, y)
        w = min(img_w - x, w); h = min(img_h - y, h)
        print(f"\n✅ Best plate: ({x},{y},{w},{h}) score={s} via [{method}]")
        return (x, y, w, h)
    except Exception as e:
        print(f"[detect] Error: {e}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════
#  BACKGROUND REMOVAL — ENHANCED CLEAN FUNCTION
# ═══════════════════════════════════════════════════════════

def clean_background_alpha(img_pil):
    """
    rembg ke baad alpha channel aggressive clean karo:
    - LARGEST connected component = car → keep karo
    - Baaki sab stray blobs (objects, shadows, floor patches) → fully transparent
    - Car ke andar ke holes fill karo (windows, sunroof)
    - Edges smooth rakho
    """
    arr = np.array(img_pil).copy()
    alpha = arr[:, :, 3].copy()
    h, w = alpha.shape

    # ── Step 1: Binary mask from rembg ──────────────────────
    car_mask = (alpha > 128).astype(np.uint8)

    # ── Step 2: Morphological close — holes fill karo ───────
    # Car windows, sunroof, gaps ko band karo
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    car_closed = cv2.morphologyEx(car_mask, cv2.MORPH_CLOSE, kernel_close, iterations=5)

    # ── Step 3: CONNECTED COMPONENTS — sirf sabse bada blob rakho ──
    # Yahi asli fix hai: side objects, floor patches, shadows
    # sab alag connected components hain — unhe hatao
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        car_closed, connectivity=8
    )

    if num_labels > 1:
        # Component 0 = background, skip karo
        # Baaki mein se sabse bada area dhundo
        areas = stats[1:, cv2.CC_STAT_AREA]  # background chhod ke
        largest_label = int(np.argmax(areas)) + 1  # +1 because we skipped bg

        # Sirf largest component rakho, sab transparent
        main_car_mask = (labels == largest_label).astype(np.uint8)

        # Small secondary blobs bhi check karo — agar car ka part hain
        # (jaise side mirror jo disconnect ho gaya) toh rakho
        # Rule: agar blob ka area > 2% of largest, aur car ke paas hai, rakho
        largest_area = areas[largest_label - 1]
        for label_idx in range(1, num_labels):
            if label_idx == largest_label:
                continue
            blob_area = stats[label_idx, cv2.CC_STAT_AREA]
            if blob_area < largest_area * 0.02:
                continue  # bahut chota — stray pixel, ignore

            # Blob ka bounding box
            bx = stats[label_idx, cv2.CC_STAT_LEFT]
            by = stats[label_idx, cv2.CC_STAT_TOP]
            bw = stats[label_idx, cv2.CC_STAT_WIDTH]
            bh = stats[label_idx, cv2.CC_STAT_HEIGHT]
            blob_cx = bx + bw / 2
            blob_cy = by + bh / 2

            # Main car ka bounding box
            car_bb = stats[largest_label]
            car_left   = car_bb[cv2.CC_STAT_LEFT]
            car_top    = car_bb[cv2.CC_STAT_TOP]
            car_right  = car_left + car_bb[cv2.CC_STAT_WIDTH]
            car_bottom = car_top  + car_bb[cv2.CC_STAT_HEIGHT]

            # Blob car ke bounding box ke andar ya bahut paas hai?
            margin = max(w, h) * 0.05  # 5% margin
            inside_or_near = (
                (car_left - margin) <= blob_cx <= (car_right + margin) and
                (car_top  - margin) <= blob_cy <= (car_bottom + margin)
            )

            if inside_or_near and blob_area > largest_area * 0.05:
                # Likely car part (mirror, bumper piece) — rakho
                main_car_mask = cv2.bitwise_or(main_car_mask, (labels == label_idx).astype(np.uint8))
                print(f"[clean] Kept secondary blob label={label_idx} area={blob_area}")
            else:
                print(f"[clean] Removed stray blob label={label_idx} area={blob_area} at ({blob_cx:.0f},{blob_cy:.0f})")
    else:
        main_car_mask = car_closed

    # ── Step 4: Final close — koi last gap fill karo ────────
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    main_car_mask = cv2.morphologyEx(main_car_mask, cv2.MORPH_CLOSE, kernel_final, iterations=3)

    # ── Step 5: Alpha apply ──────────────────────────────────
    # Car region = solid 255, outside = 0
    final_alpha = (main_car_mask * 255).astype(np.uint8)

    # rembg ke edge pixels preserve karo — natural boundary ke liye
    edge_zone = (alpha > 5) & (alpha <= 128)
    # Sirf wahan jahan main_car_mask ke paas hain
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near_car = cv2.dilate(main_car_mask, kernel_edge, iterations=3).astype(bool)
    edge_preserve = edge_zone & near_car
    final_alpha[edge_preserve] = alpha[edge_preserve]

    arr[:, :, 3] = final_alpha
    print(f"[clean] Alpha cleaned: {num_labels-1} components found, kept car + valid parts")
    return Image.fromarray(arr, 'RGBA')


def remove_background_clean(image_bytes):
    """
    rembg use karke background remove karo, phir AGGRESSIVELY clean karo.
    Returns RGBA PIL Image — sirf car, koi bhi stray object/shadow nahi.
    """
    if not _has_rembg:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        return img

    print("[bg_remove] Running rembg...")
    removed = rembg_remove(image_bytes)
    img_pil = Image.open(io.BytesIO(removed)).convert('RGBA')
    print(f"[bg_remove] rembg done, size={img_pil.size}")

    # Aggressive clean — stray blobs, floor patches, side objects sab remove
    img_pil = clean_background_alpha(img_pil)
    print("[bg_remove] Alpha aggressively cleaned ✅")
    return img_pil


def remove_floor_shadow_strip(img_pil):
    """
    Car ke bottom se neeche jo bhi connected floor/shadow pixels hain unhe hatao.
    Car ka lowest solid row dhundo — us row ke neeche sab transparent.
    """
    arr = np.array(img_pil).copy()
    alpha = arr[:, :, 3]
    h, w = alpha.shape

    # Car ka lowest solid row (alpha > 200)
    car_rows = np.where(np.any(alpha > 200, axis=1))[0]
    if len(car_rows) == 0:
        return img_pil

    car_bottom_row = int(car_rows[-1])
    r_all = arr[:, :, 0].astype(int)
    g_all = arr[:, :, 1].astype(int)
    b_all = arr[:, :, 2].astype(int)

    # Pure tyre/rubber pixels — bahut dark (safe to keep)
    is_pure_tyre = (r_all < 45) & (g_all < 45) & (b_all < 45)

    # Car bottom ke neeche sab transparent (floor, shadow, road)
    if car_bottom_row + 1 < h:
        below = arr[car_bottom_row + 1:, :, :]
        below_r = r_all[car_bottom_row + 1:, :]
        below_g = g_all[car_bottom_row + 1:, :]
        below_b = b_all[car_bottom_row + 1:, :]
        not_pure_tyre = ~((below_r < 45) & (below_g < 45) & (below_b < 45))
        below[not_pure_tyre, 3] = 0
        arr[car_bottom_row + 1:, :, :] = below

    print(f"[floor_strip] Cleared floor/shadow below row {car_bottom_row}")
    return Image.fromarray(arr.astype(np.uint8), 'RGBA')


def restore_tyre_pixels(orig_path, img_pil):
    """
    rembg kabhi kabhi pure black tyre pixels galat remove kar deta hai.
    Sirf PURE BLACK (RGB < 45) pixels restore karo — car bbox ke andar sirf.
    Floor/shadow (jo gray/brown hote hain) RESTORE NAHI honge.
    """
    try:
        orig_pil = Image.open(orig_path).convert('RGBA')
        orig_arr = np.array(orig_pil)
        rmbg_arr = np.array(img_pil).copy()

        # Car bounding box dhundo
        alpha_check = rmbg_arr[:, :, 3]
        car_rows = np.where(np.any(alpha_check > 128, axis=1))[0]
        car_cols = np.where(np.any(alpha_check > 128, axis=0))[0]

        if len(car_rows) == 0 or len(car_cols) == 0:
            return img_pil

        car_top    = int(car_rows[0])
        car_bottom = int(car_rows[-1])
        car_left   = int(car_cols[0])
        car_right  = int(car_cols[-1])

        # Only bottom 30% of car bbox (tyre zone)
        tyre_top = car_top + int((car_bottom - car_top) * 0.70)

        wheel_orig = orig_arr[tyre_top:car_bottom, car_left:car_right, :]
        wheel_rmbg = rmbg_arr[tyre_top:car_bottom, car_left:car_right, :]

        r = wheel_orig[:, :, 0].astype(int)
        g = wheel_orig[:, :, 1].astype(int)
        b = wheel_orig[:, :, 2].astype(int)

        # ONLY pure black — actual tyre rubber
        is_pure_tyre = (r < 45) & (g < 45) & (b < 45)
        wrongly_removed = (wheel_rmbg[:, :, 3] < 30) & is_pure_tyre

        wheel_rmbg[wrongly_removed, 0] = wheel_orig[wrongly_removed, 0]
        wheel_rmbg[wrongly_removed, 1] = wheel_orig[wrongly_removed, 1]
        wheel_rmbg[wrongly_removed, 2] = wheel_orig[wrongly_removed, 2]
        wheel_rmbg[wrongly_removed, 3] = 255

        rmbg_arr[tyre_top:car_bottom, car_left:car_right, :] = wheel_rmbg
        restored = int(wrongly_removed.sum())
        print(f"[tyre] Restored {restored} pure tyre pixels inside car bbox")

        return Image.fromarray(rmbg_arr.astype(np.uint8), 'RGBA')
    except Exception as e:
        print(f"[tyre] Wheel restore skipped: {e}")
        return img_pil




# ═══════════════════════════════════════════════════════════
#  PLATE REMOVAL ENGINE
# ═══════════════════════════════════════════════════════════

def _load_font(size):
    paths = [
        "arial.ttf", "Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, int(size))
        except Exception:
            pass
    return ImageFont.load_default()


def apply_removal(image_path, output_path, x, y, w, h, mode='caryanams'):
    try:
        img  = Image.open(image_path).convert('RGB')
        iw, ih = img.size
        x = max(0, x); y = max(0, y)
        w = min(iw - x, w); h = min(ih - y, h)

        if mode == 'blur':
            region = img.crop((x, y, x+w, y+h))
            block  = max(8, h // 4)
            small  = region.resize((max(1, w//block), max(1, h//block)), Image.NEAREST)
            pix    = small.resize((w, h), Image.NEAREST)
            pix    = pix.filter(ImageFilter.GaussianBlur(radius=2))
            img.paste(pix, (x, y))

        elif mode == 'white':
            draw = ImageDraw.Draw(img)
            draw.rectangle([x, y, x+w, y+h], fill=(255, 255, 255))

        elif mode == 'black':
            draw = ImageDraw.Draw(img)
            draw.rectangle([x, y, x+w, y+h], fill=(0, 0, 0))

        elif mode == 'caryanams':
            logo_pasted = False
            logo_paths = [
                'caryanams_logo_clean.png',
                os.path.join(os.path.dirname(__file__), 'caryanams_logo_clean.png'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'caryanams_logo_clean.png'),
                'static/caryanams_logo_clean.png',
                os.path.join('static', 'caryanams_logo_clean.png'),
            ]
            for logo_path in logo_paths:
                if os.path.exists(logo_path):
                    try:
                        logo_raw = Image.open(logo_path).convert('RGBA')
                        arr_l    = np.array(logo_raw)
                        white_mask        = (arr_l[:,:,0]>240) & (arr_l[:,:,1]>240) & (arr_l[:,:,2]>240)
                        arr_l[:,:,3]      = np.where(white_mask, 0, 255)
                        content_mask = arr_l[:,:,3] > 10
                        rows_m = np.any(content_mask, axis=1)
                        cols_m = np.any(content_mask, axis=0)
                        rmin, rmax = np.where(rows_m)[0][[0, -1]]
                        cmin, cmax = np.where(cols_m)[0][[0, -1]]
                        logo_t = Image.fromarray(arr_l).crop((cmin, rmin, cmax + 1, rmax + 1))
                        draw = ImageDraw.Draw(img)
                        draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))
                        margin   = max(3, h // 10)
                        target_w = w - margin * 2
                        target_h = h - margin * 2
                        lw, lh   = logo_t.size
                        scale_f  = min(target_w / lw, target_h / lh)
                        nw = max(1, int(lw * scale_f))
                        nh = max(1, int(lh * scale_f))
                        logo_s   = logo_t.resize((nw, nh), Image.LANCZOS)
                        base_rgba = img.convert('RGBA')
                        px = x + (w - nw) // 2
                        py = y + (h - nh) // 2
                        base_rgba.paste(logo_s, (px, py), logo_s)
                        img = base_rgba.convert('RGB')
                        logo_pasted = True
                        break
                    except Exception as le:
                        print(f"[apply] Logo paste failed ({logo_path}): {le}")
                        continue

            if not logo_pasted:
                draw = ImageDraw.Draw(img)
                draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))
                f_main = _load_font(max(9, h * 0.38))
                tmp_d  = ImageDraw.Draw(Image.new("RGB", (1, 1)))
                try:
                    bb = tmp_d.textbbox((0, 0), "Caryanams", font=f_main)
                    tw, th = bb[2]-bb[0], bb[3]-bb[1]
                except Exception:
                    tw = w // 2; th = h // 3
                draw.text((x + (w - tw) // 2, y + (h - th) // 2),
                          "Caryanams", fill=(17, 56, 110), font=f_main)

        img.save(output_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"[apply] Error: {e}")
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════
#  SHOWROOM BACKGROUND COMPOSITE
# ═══════════════════════════════════════════════════════════

SHOWROOM_BG_PATHS = [
    'showroom_bg.jpeg',
    'showroom_bg.jpg',
    'showroom_bg.png',
    'static/showroom_bg.jpeg',
    'static/showroom_bg.jpg',
    'static/showroom_bg.png',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'showroom_bg.jpeg'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'showroom_bg.jpg'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'showroom_bg.png'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'showroom_bg.jpeg'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'showroom_bg.jpg'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'showroom_bg.png'),
]

def _load_showroom_bg():
    for p in SHOWROOM_BG_PATHS:
        if os.path.exists(p):
            print(f"[bg] Loaded showroom background: {p}")
            return Image.open(p).convert('RGBA')
    print("[bg] Showroom BG not found — using white fallback")
    return None


def apply_showroom_background(car_rgba, car_size_pct=75):
    """
    Transparent car ko fixed showroom floor background pe composite karo.
    car_rgba: RGBA PIL Image — sirf car, background fully transparent
    """
    bg_orig = _load_showroom_bg()

    if bg_orig:
        bg_w, bg_h = bg_orig.size
        # White base banao pehle, phir BG paste karo
        base   = Image.new('RGB', (bg_w, bg_h), (255, 255, 255))
        bg_rgb = bg_orig.convert('RGBA')
        base.paste(bg_rgb, mask=bg_rgb.split()[3])
        canvas = base.convert('RGBA')
    else:
        bg_w, bg_h = 1382, 752
        canvas = Image.new('RGBA', (bg_w, bg_h), (255, 255, 255, 255))

    floor_y = int(bg_h * 0.95)

    # Car size scale
    size_scale   = max(0.30, min(1.00, car_size_pct / 100.0))
    car_w, car_h = car_rgba.size
    target_car_w = int(bg_w * size_scale)
    scale        = target_car_w / car_w
    target_car_h = int(car_h * scale)
    print(f"[bg] car_size_pct={car_size_pct}% target_w={target_car_w}px")

    max_car_h = floor_y - 5
    if target_car_h > max_car_h:
        scale        = max_car_h / car_h
        target_car_w = int(car_w * scale)
        target_car_h = max_car_h

    if target_car_w > bg_w:
        scale        = bg_w / car_w
        target_car_w = bg_w
        target_car_h = int(car_h * scale)

    car_scaled = car_rgba.resize((target_car_w, target_car_h), Image.LANCZOS)

    # ── Position (CENTERED SHOWROOM LOOK) ─────────────────────
    car_x = max(0, (bg_w - target_car_w) // 2)
    center_offset = int(bg_h * 0.08)
    car_y = max(0, (bg_h - target_car_h) // 2 + center_offset)
    max_y = floor_y - target_car_h + 25
    if car_y > max_y:
        car_y = max_y

    # ── Contact shadow ────────────────────────────────────────
    shadow_layer = Image.new('RGBA', (bg_w, bg_h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow_layer)
    sh_cx = bg_w // 2
    sh_cy = floor_y - int(target_car_h * 0.01)
    sh_rx = int(target_car_w * 0.42)
    sh_ry = int(target_car_h * 0.045)
    shadow_layers = [
        (sh_rx + 50, sh_ry + 18, 12),
        (sh_rx + 30, sh_ry + 11, 22),
        (sh_rx + 14, sh_ry + 6,  35),
        (sh_rx,      sh_ry,      50),
        (sh_rx - 16, sh_ry - 3,  65),
        (sh_rx - 32, sh_ry - 6,  50),
        (sh_rx - 46, sh_ry - 9,  35),
    ]
    for rx, ry, alpha in shadow_layers:
        if rx <= 0 or ry <= 0:
            continue
        sdraw.ellipse([sh_cx-rx, sh_cy-ry, sh_cx+rx, sh_cy+ry], fill=(30, 30, 35, alpha))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=14))
    canvas = Image.alpha_composite(canvas, shadow_layer)

    # ── Car paste ─────────────────────────────────────────────
    canvas.paste(car_scaled, (car_x, car_y), car_scaled)

    # ── Final RGB output — no black areas ─────────────────────
    final = Image.new('RGB', (bg_w, bg_h), (255, 255, 255))
    final.paste(canvas.convert('RGB'), mask=canvas.split()[3])

    return final


# ═══════════════════════════════════════════════════════════
#  MAIN PIPELINE: Plate Remove + BG Remove + Showroom BG
# ═══════════════════════════════════════════════════════════

def process_all_in_one(image_path, output_path, mode='caryanams', manual=None, car_size_pct=60):
    """
    ONE-CLICK pipeline:
      Step 1 → Number plate detect & remove
      Step 2 → Background FULLY remove (rembg + clean) — sirf car bachegi
      Step 3 → Car ko showroom floor background pe composite
    """
    try:
        print(f"\n{'='*55}")
        print(f"  ONE-CLICK PIPELINE START")
        print(f"{'='*55}")

        # ── STEP 1: Number Plate Remove ──────────────────────
        temp_plate_path = output_path.replace('.png', '_step1.jpg')
        plate_info = None

        if manual:
            x = int(manual['x']); y = int(manual['y'])
            w = int(manual['w']); h = int(manual['h'])
            ok1 = apply_removal(image_path, temp_plate_path, x, y, w, h, mode)
            plate_info = {'x': x, 'y': y, 'width': w, 'height': h}
            print(f"[step1] Manual plate remove: ok={ok1}")
        else:
            plate = detect_number_plate(image_path)
            if plate:
                x, y, w, h = plate
                ok1 = apply_removal(image_path, temp_plate_path, x, y, w, h, mode)
                plate_info = {'x': x, 'y': y, 'width': w, 'height': h}
                print(f"[step1] Auto plate remove: ok={ok1} at ({x},{y},{w},{h})")
            else:
                import shutil
                shutil.copy(image_path, temp_plate_path)
                ok1 = True
                print("[step1] No plate detected — skipping plate step, continuing BG remove")

        source_for_bg = temp_plate_path if os.path.exists(temp_plate_path) else image_path

        # ── STEP 2: Background FULLY Remove ──────────────────
        # Car ke alawa SAB KUCH remove hoga — sky, road, trees, showroom walls
        print("[step2] Removing ALL background — sirf car bachegi...")

        with open(source_for_bg, 'rb') as f:
            raw = f.read()

        # rembg se background remove karo
        img_pil = remove_background_clean(raw)

        # Floor/shadow strip remove karo — car ke neeche ka sab hatao
        img_pil = remove_floor_shadow_strip(img_pil)

        # Tyre pixels restore karo jo accidentally remove ho gaye
        img_pil = restore_tyre_pixels(source_for_bg, img_pil)

        # ── VERIFY: Background sach mein gaya? ───────────────
        arr_check = np.array(img_pil)
        alpha_check = arr_check[:, :, 3]
        transparent_pct = (alpha_check < 10).sum() / alpha_check.size * 100
        car_pct = (alpha_check > 200).sum() / alpha_check.size * 100
        print(f"[step2] Background removed: {transparent_pct:.1f}% transparent, {car_pct:.1f}% solid car")

        # ── STEP 3: Showroom Background Apply ────────────────
        print("[step3] Car ko showroom floor pe composite kar raha hoon...")
        final_img = apply_showroom_background(img_pil, car_size_pct=car_size_pct)
        final_img.save(output_path, 'PNG', optimize=True)
        print(f"[step3] Saved final → {output_path}")

        # Cleanup temp
        try:
            if os.path.exists(temp_plate_path):
                os.remove(temp_plate_path)
        except Exception:
            pass

        print(f"{'='*55}\n  PIPELINE DONE ✅\n{'='*55}\n")
        return True, plate_info

    except Exception as e:
        print(f"[pipeline] ERROR: {e}")
        traceback.print_exc()
        return False, None


# ═══════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('plate_remover.html')


@app.route('/api/upload-car', methods=['POST'])
def upload_car():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'Empty filename'}), 400

        uid = str(uuid.uuid4())[:8]
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
            ext = '.jpg'

        fname         = f"car_{uid}{ext}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(original_path)

        car = CarImage(id=uid, filename=fname, original_path=original_path)
        db.session.add(car)
        db.session.commit()

        return jsonify({
            'success': True,
            'id': uid,
            'original_url': '/' + original_path.replace('\\', '/')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-plate/<image_id>', methods=['GET'])
def detect_plate_api(image_id):
    try:
        car = CarImage.query.get_or_404(image_id)
        plate = detect_number_plate(car.original_path)
        if plate:
            x, y, w, h = plate
            iw, ih = Image.open(car.original_path).size
            return jsonify({
                'detected': True,
                'x': x, 'y': y, 'width': w, 'height': h,
                'img_width': iw, 'img_height': ih,
                'message': f'Plate at ({x},{y}) size {w}×{h}'
            })
        else:
            iw, ih = Image.open(car.original_path).size
            return jsonify({
                'detected': False,
                'img_width': iw, 'img_height': ih,
                'message': 'Auto-detection failed. Use manual selection.'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process-car/<image_id>', methods=['POST'])
def process_car_api(image_id):
    """
    ONE-CLICK endpoint: plate remove + BG FULL remove + showroom BG apply.
    """
    try:
        car = CarImage.query.get(image_id)
        if not car:
            return jsonify({'success': False, 'message': 'Image not found'}), 404

        data          = request.get_json() or {}
        mode          = data.get('mode', 'caryanams')
        manual        = data.get('manual')
        car_size_pct  = int(data.get('car_size_pct', 60))

        output_path = os.path.join(
            app.config['PROCESSED_FOLDER'],
            f'final_{image_id}.png'
        )

        ok, plate_info = process_all_in_one(car.original_path, output_path, mode, manual, car_size_pct)

        if ok and os.path.exists(output_path):
            car.processed_path = output_path
            db.session.commit()

            try:
                with open(output_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
            except Exception:
                b64 = None

            if _has_rembg:
                msg = '✅ Plate removed + Background FULLY removed + Showroom floor applied!'
            else:
                msg = '⚠️ Plate removed + Showroom applied (rembg install karo BG removal ke liye)'

            return jsonify({
                'success': True,
                'processed_url': '/' + output_path.replace('\\', '/'),
                'preview_b64': b64,
                'plate': plate_info,
                'message': msg
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Processing failed. Try manual plate selection.'
            })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/remove-plate/<image_id>', methods=['POST'])
def remove_plate_api(image_id):
    return process_car_api(image_id)


@app.route('/api/download/<image_id>')
def download_image(image_id):
    car  = CarImage.query.get_or_404(image_id)
    path = car.processed_path or car.original_path
    fname = f'caryanams_{car.id[:8]}.png'
    try:
        hd_path = path.replace('.png', '_hd.png')
        if not os.path.exists(hd_path):
            img = Image.open(path)
            img.save(hd_path, 'PNG', optimize=False, compress_level=1)
        return send_file(hd_path, as_attachment=True, download_name=fname)
    except Exception:
        return send_file(path, as_attachment=True, download_name=fname)


@app.route('/api/gallery', methods=['GET'])
def gallery_api():
    try:
        cars = CarImage.query.filter(CarImage.processed_path != None).order_by(CarImage.created_at.desc()).all()
        result = []
        for car in cars:
            if car.processed_path and os.path.exists(car.processed_path):
                result.append({
                    'id': car.id,
                    'filename': car.filename,
                    'processed_url': '/' + car.processed_path.replace('\\', '/'),
                    'original_url':  '/' + car.original_path.replace('\\', '/'),
                    'created_at': car.created_at.strftime('%d %b %Y, %I:%M %p') if car.created_at else ''
                })
        return jsonify({'success': True, 'cars': result, 'total': len(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/gallery')
def gallery_page():
    gallery_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gallery.html')
    if os.path.exists(gallery_path):
        return send_file(gallery_path)
    return "<h2>gallery.html not found</h2>", 404


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚗  CARYANAMS — Number Plate + Background Remover  v3")
    print("="*60)
    print("✅  Car ka background FULLY remove hoga (sky/road/walls sab)")
    print("✅  Multi-method plate detection (5 algorithms)")
    print("✅  Manual selection fallback")
    print("✅  Showroom floor background apply hoga")
    print("\n🌐  Open: http://localhost:5055")
    print("="*60 + "\n")
    app.run(debug=True, port=5055)
