import cv2
import mediapipe as mp
import numpy as np
import math

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_draw = mp.solutions.drawing_utils
hand_conn_spec = mp_draw.DrawingSpec(color=(180, 120, 255), thickness=2)
hand_dot_spec  = mp_draw.DrawingSpec(color=(255, 200, 255), thickness=4, circle_radius=3)

# ── Camera ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame0 = cap.read()
H, W = (frame0.shape[:2] if ret else (720, 1280))

# ── Palette (BGR) ────────────────────────────────────────────────────────────
PALETTE = [
    {"name": "Rose",   "color": (100,  80, 255)},
    {"name": "Peach",  "color": ( 80, 160, 255)},
    {"name": "Lemon",  "color": ( 60, 230, 240)},
    {"name": "Mint",   "color": (130, 220, 120)},
    {"name": "Sky",    "color": (235, 180,  80)},
    {"name": "Lilac",  "color": (220, 130, 190)},
    {"name": "White",  "color": (240, 240, 240)},
    {"name": "Erase",  "color": (  0,   0,   0)},
]

PALETTE_W    = 80
PALETTE_PAD  = 12
SWATCH_GAP   = 8
SWATCH_COUNT = len(PALETTE)
STATUS_H     = 100

# ── State ────────────────────────────────────────────────────────────────────
xp, yp      = 0, 0
draw_color  = PALETTE[0]["color"]
color_idx   = 0
smoothening = 4
canvas      = np.zeros((H, W, 3), dtype=np.uint8)
mode_text   = "READY"
mode_color  = (200, 200, 200)

# ── Drag state ───────────────────────────────────────────────────────────────
drag_active = False
drag_anchor = (0, 0)

# ── Helpers ──────────────────────────────────────────────────────────────────
def fingers_up(lm):
    tips = [4, 8, 12, 16, 20]
    up = []
    up.append(1 if lm[tips[0]][0] > lm[tips[0]-1][0] else 0)
    for i in range(1, 5):
        up.append(1 if lm[tips[i]][1] < lm[tips[i]-2][1] else 0)
    return up

def is_fist(lm):
    tips = [8, 12, 16, 20]
    for tip in tips:
        if lm[tip][1] < lm[tip - 2][1]:
            return False
    return True

def lerp(a, b, t):
    return int(a + (b - a) / t)

def draw_rounded_rect(img, x0, y0, x1, y1, r, color, filled=True):
    thick = cv2.FILLED if filled else 1
    cv2.rectangle(img, (x0+r, y0), (x1-r, y1), color, thick)
    cv2.rectangle(img, (x0, y0+r), (x1, y1-r), color, thick)
    for cx, cy in [(x0+r, y0+r),(x1-r, y0+r),(x0+r, y1-r),(x1-r, y1-r)]:
        cv2.circle(img, (cx, cy), r, color, thick)

def draw_palette(img, sel_idx):
    strip_x = W - PALETTE_W
    overlay = img.copy()
    cv2.rectangle(overlay, (strip_x, 0), (W, H - STATUS_H), (20, 16, 28), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.line(img, (strip_x, 0), (strip_x, H - STATUS_H), (120, 80, 200), 2)

    avail_h = H - STATUS_H - 2 * PALETTE_PAD
    sw_h    = (avail_h - SWATCH_GAP * (SWATCH_COUNT - 1)) // SWATCH_COUNT
    sw_w    = PALETTE_W - 2 * PALETTE_PAD

    for i, p in enumerate(PALETTE):
        y0 = PALETTE_PAD + i * (sw_h + SWATCH_GAP)
        y1 = y0 + sw_h
        x0 = strip_x + PALETTE_PAD
        x1 = x0 + sw_w

        cv2.rectangle(img, (x0+4, y0+4), (x1+4, y1+4), (0,0,0), cv2.FILLED)
        draw_rounded_rect(img, x0, y0, x1, y1, 8, p["color"], filled=True)
        if i == sel_idx:
            draw_rounded_rect(img, x0-3, y0-3, x1+3, y1+3, 10, (255,255,255), filled=False)
            draw_rounded_rect(img, x0-5, y0-5, x1+5, y1+5, 11, (180,140,255), filled=False)
        if p["name"] == "Erase":
            cx2, cy2 = (x0+x1)//2, (y0+y1)//2
            cv2.putText(img, "X", (cx2-8, cy2+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,120,120), 2)
    return img

def draw_status_hud(img, fingers, mode_txt, mode_col, cur_color):
    hy = H - STATUS_H
    overlay = img.copy()
    cv2.rectangle(overlay, (0, hy), (W, H), (14, 10, 22), cv2.FILLED)
    cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)
    cv2.line(img, (0, hy), (W - PALETTE_W, hy), (120, 80, 200), 2)

    finger_names = ["Thumb","Index","Middle","Ring","Pinky"]
    total_w = W - PALETTE_W
    slot_w  = total_w // 5

    for i, (fname, fup) in enumerate(zip(finger_names, fingers)):
        cx      = i * slot_w + slot_w // 2
        cy_base = hy + STATUS_H - 18
        bar_h   = 52 * fup
        bar_col = (min(255,cur_color[0]+60), min(255,cur_color[1]+60), min(255,cur_color[2]+60)) if fup else (40,35,55)
        draw_rounded_rect(img, cx-15, hy+16, cx+15, cy_base, 7, (30,25,45), filled=True)
        if fup:
            draw_rounded_rect(img, cx-13, cy_base-bar_h, cx+13, cy_base, 6, bar_col, filled=True)
        out_col = (180,140,255) if fup else (60,55,75)
        draw_rounded_rect(img, cx-15, hy+16, cx+15, cy_base, 7, out_col, filled=False)
        cv2.putText(img, fname[0], (cx-6, hy+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200,180,255) if fup else (80,70,100), 1)

    lx = W - PALETTE_W + 4
    cv2.putText(img, "MODE", (lx, hy + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100,90,130), 1)
    words = mode_txt.split()
    for wi, word in enumerate(words):
        cv2.putText(img, word, (lx, hy + 50 + wi*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, mode_col, 2)
    return img

def draw_cursor(img, x, y, color, mode):
    if mode == "DRAW":
        cv2.circle(img, (x, y), 10, color, cv2.FILLED)
        cv2.circle(img, (x, y), 12, (255,255,255), 1)
    elif mode == "SELECT":
        cv2.circle(img, (x, y), 14, (255,255,255), 2)
        cv2.circle(img, (x, y),  5, (255,255,255), cv2.FILLED)
    elif mode == "DRAG":
        cv2.circle(img, (x, y), 18, (80, 220, 255), 2)
        cv2.line(img, (x-10, y), (x+10, y), (80,220,255), 2)
        cv2.line(img, (x, y-10), (x, y+10), (80,220,255), 2)
    return img

def get_palette_hover(x, y):
    strip_x = W - PALETTE_W
    if x < strip_x:
        return -1
    avail_h = H - STATUS_H - 2 * PALETTE_PAD
    sw_h    = (avail_h - SWATCH_GAP * (SWATCH_COUNT - 1)) // SWATCH_COUNT
    for i in range(SWATCH_COUNT):
        y0 = PALETTE_PAD + i * (sw_h + SWATCH_GAP)
        y1 = y0 + sw_h
        if y0 <= y <= y1:
            return i
    return -1

# ── Main loop ────────────────────────────────────────────────────────────────
while True:
    success, img = cap.read()
    if not success:
        break

    img    = cv2.flip(img, 1)
    img    = cv2.resize(img, (W, H))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lm_list = []
    f_up    = [0, 0, 0, 0, 0]

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS,
                                   hand_dot_spec, hand_conn_spec)
            for lm in handLms.landmark:
                lm_list.append((int(lm.x * W), int(lm.y * H)))

        if lm_list:
            f_up     = fingers_up(lm_list)
            rx, ry   = lm_list[8]
            x1       = lerp(xp if xp else rx, rx, smoothening)
            y1       = lerp(yp if yp else ry, ry, smoothening)
            total_up = sum(f_up)
            fist     = is_fist(lm_list)
            palm_open = total_up == 5

            # FIST → drag entire canvas
            if fist:
                mode_text  = "DRAG"
                mode_color = (80, 220, 255)
                cx, cy = lm_list[9]
                if not drag_active:
                    drag_active = True
                    drag_anchor = (cx, cy)
                else:
                    dx = cx - drag_anchor[0]
                    dy = cy - drag_anchor[1]
                    M  = np.float32([[1, 0, dx], [0, 1, dy]])
                    canvas[:] = cv2.warpAffine(canvas, M, (W, H))
                    drag_anchor = (cx, cy)
                draw_cursor(img, cx, cy, (80,220,255), "DRAG")
                xp, yp = 0, 0

            # OPEN PALM → release drag
            elif palm_open:
                mode_text   = "RELEASED"
                mode_color  = (180, 255, 180)
                drag_active = False
                xp, yp = 0, 0

            # 4 fingers → erase all
            elif total_up == 4:
                mode_text   = "ERASE ALL"
                mode_color  = (80, 80, 255)
                canvas[:] = 0
                drag_active = False
                xp, yp = 0, 0

            # 2 fingers (index + middle) → colour select
            elif f_up[1] == 1 and f_up[2] == 1 and f_up[3] == 0:
                mode_text   = "SELECT"
                mode_color  = (150, 220, 255)
                drag_active = False
                xp, yp = 0, 0
                hit = get_palette_hover(x1, y1)
                if hit >= 0:
                    color_idx  = hit
                    draw_color = PALETTE[hit]["color"]
                draw_cursor(img, x1, y1, draw_color, "SELECT")

            # 1 finger → draw
            elif f_up[1] == 1 and f_up[2] == 0:
                mode_text   = "DRAW"
                mode_color  = (min(255,draw_color[0]+80), min(255,draw_color[1]+80), min(255,draw_color[2]+80))
                drag_active = False
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                is_eraser = PALETTE[color_idx]["name"] == "Erase"
                thick = 48 if is_eraser else 12
                col   = (0,0,0) if is_eraser else draw_color
                cv2.line(canvas, (xp,yp), (x1,y1), col, thick)
                draw_cursor(img, x1, y1, draw_color, "DRAW")
                xp, yp = x1, y1

            else:
                mode_text   = "HOLD"
                mode_color  = (140, 120, 180)
                drag_active = False
                xp, yp = 0, 0

    else:
        xp, yp      = 0, 0
        drag_active = False
        mode_text   = "READY"
        mode_color  = (160, 140, 200)

    # ── Composite canvas onto camera ─────────────────────────────────────────
    gray     = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg   = cv2.bitwise_and(img, img, mask=mask_inv)
    output   = cv2.add(img_bg, canvas)

    # ── UI overlays ──────────────────────────────────────────────────────────
    output = draw_palette(output, color_idx)
    output = draw_status_hud(output, f_up, mode_text, mode_color, draw_color)

    cv2.putText(output, "AirCanvas", (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (80,60,120), 3)
    cv2.putText(output, "AirCanvas", (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (220,200,255), 1)

    legend = "1:Draw  2:Select  4:EraseAll  Fist:Drag  OpenPalm:Release  [C]Clear  [ESC]Quit"
    cv2.putText(output, legend, (W//2 - 340, H - STATUS_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130,110,170), 1)

    cv2.imshow("AirCanvas", output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        canvas[:] = 0

cap.release()
cv2.destroyAllWindows()