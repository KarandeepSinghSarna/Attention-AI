"""
AttentionAI — Real-time Classroom Attention Analytics
======================================================
Libraries: OpenCV · MediaPipe FaceMesh · NumPy · Pillow · Tkinter · csv/os
Python 3.12 | mediapipe 0.10.14 | opencv-python 4.13 | pillow 12

Run:  python attention_ai.py
"""

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import time
import random
import math
import csv
import os
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
from PIL import Image, ImageTk

# ─────────────────────────────────────────────────────────────────
#  THEME COLOURS (identical to sample GUI)
# ─────────────────────────────────────────────────────────────────
BG_DARK      = "#0F1117"
BG_PANEL     = "#1A1D26"
BG_SURFACE   = "#22263A"
BG_SURFACE2  = "#2A2F45"
ACCENT_GREEN = "#1D9E75"
ACCENT_BLUE  = "#378ADD"
ACCENT_AMBER = "#EF9F27"
ACCENT_RED   = "#E24B4A"
ACCENT_PINK  = "#D4537E"
TEXT_PRIMARY = "#EAEAF0"
TEXT_MUTED   = "#8A8DA8"
TEXT_GREEN   = "#5DCAA5"
BORDER       = "#2E3250"

STATE_COLORS = {
    "Focused":      ACCENT_GREEN,
    "Distracted":   ACCENT_AMBER,
    "Looking Away": ACCENT_RED,
    "Taking Notes": ACCENT_BLUE,
    "Drowsy":       ACCENT_PINK,
}

# Person colours for multi-person bounding boxes
PERSON_COLORS_HEX = [
    "#378ADD", "#1D9E75", "#EF9F27", "#E24B4A", "#D4537E",
    "#AFA9EC", "#5DCAA5", "#F28B30", "#A8D8A8", "#FF6B6B",
]
def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

PERSON_COLORS_BGR = [hex_to_bgr(c) for c in PERSON_COLORS_HEX]

# ─────────────────────────────────────────────────────────────────
#  DETECTION ENGINE
# ─────────────────────────────────────────────────────────────────

# MediaPipe landmark indices
# Left eye EAR: 33,160,158,133,153,144
# Right eye EAR: 362,385,387,263,373,380
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Iris centres
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
# Head pose points
NOSE_TIP   = 1
CHIN       = 152
LEFT_EAR   = 234
RIGHT_EAR  = 454
LEFT_MOUTH = 61
RIGHT_MOUTH= 291


def _dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def compute_ear(landmarks, eye_ids, w, h):
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye_ids]
    A = _dist(pts[1], pts[5])
    B = _dist(pts[2], pts[4])
    C = _dist(pts[0], pts[3])
    if C == 0:
        return 0.3
    return (A + B) / (2.0 * C)


def compute_head_pose(landmarks, w, h):
    """Returns (yaw_deg, pitch_deg) — simple ratio-based, no solvePnP."""
    def pt(i): return (landmarks[i].x * w, landmarks[i].y * h)
    nose   = pt(NOSE_TIP)
    left_e = pt(LEFT_EAR)
    right_e= pt(RIGHT_EAR)
    chin   = pt(CHIN)

    face_cx = (left_e[0] + right_e[0]) / 2
    face_cy = (left_e[1] + chin[1]) / 2
    face_w  = _dist(left_e, right_e)
    face_h  = _dist(left_e, chin)

    if face_w == 0 or face_h == 0:
        return 0.0, 0.0

    yaw   = (nose[0] - face_cx) / face_w * 90.0
    pitch = (nose[1] - face_cy) / face_h * 90.0
    return yaw, pitch


def compute_gaze(landmarks, w, h):
    """Returns (gx, gy) normalised iris offset in [-1, 1]."""
    def pt(i): return (landmarks[i].x * w, landmarks[i].y * h)

    # Left eye corners
    l_inner = pt(133)
    l_outer = pt(33)
    eye_w = _dist(l_inner, l_outer)
    if eye_w == 0:
        return 0.0, 0.0

    iris_pts = [pt(i) for i in LEFT_IRIS]
    iris_cx = np.mean([p[0] for p in iris_pts])
    iris_cy = np.mean([p[1] for p in iris_pts])
    eye_cx  = (l_inner[0] + l_outer[0]) / 2
    eye_cy  = (l_inner[1] + l_outer[1]) / 2

    gx = (iris_cx - eye_cx) / eye_w
    gy = (iris_cy - eye_cy) / eye_w
    return gx, gy


def classify_attention(ear, yaw, pitch, gx):
    """Rule-based attention state classifier."""
    EAR_THRESH    = 0.18
    YAW_THRESH    = 22.0
    PITCH_THRESH  = 18.0
    GAZE_THRESH   = 0.32
    PITCH_DOWN    = 20.0   # looking down → taking notes

    if ear < EAR_THRESH:
        return "Drowsy"
    if abs(yaw) > YAW_THRESH or abs(pitch) > PITCH_THRESH:
        return "Looking Away"
    if pitch > PITCH_DOWN:
        return "Taking Notes"
    if abs(gx) > GAZE_THRESH:
        return "Distracted"
    return "Focused"


def detect_faces(frame, face_mesh):
    """
    Run MediaPipe FaceMesh on a BGR frame.
    Returns list of dicts, one per face:
      {id, state, ear, yaw, pitch, gaze, confidence, bbox}
    """
    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    faces  = []

    if not result.multi_face_landmarks:
        return faces

    for idx, face_lm in enumerate(result.multi_face_landmarks):
        lm = face_lm.landmark

        ear   = (compute_ear(lm, LEFT_EYE, w, h) +
                 compute_ear(lm, RIGHT_EYE, w, h)) / 2.0
        yaw, pitch = compute_head_pose(lm, w, h)
        gx, gy     = compute_gaze(lm, w, h)
        state      = classify_attention(ear, yaw, pitch, gx)

        # Bounding box from all landmarks
        xs = [l.x * w for l in lm]
        ys = [l.y * h for l in lm]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        pad = 15
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        conf = int(min(99, max(60,
            100 - abs(yaw) * 0.8 - abs(pitch) * 0.5 +
            (ear - 0.18) * 50 + random.uniform(-3, 3)
        )))

        faces.append({
            "id":    idx + 1,
            "state": state,
            "ear":   round(ear, 3),
            "yaw":   round(yaw, 1),
            "pitch": round(pitch, 1),
            "gaze":  round(gx, 3),
            "conf":  conf,
            "bbox":  (x1, y1, x2, y2),
            "lm":    lm,
        })

    return faces


def draw_overlays(frame, faces):
    """Draw bounding boxes, landmarks, gaze dot on frame."""
    h, w = frame.shape[:2]
    for f in faces:
        pid   = f["id"] - 1
        col_p = PERSON_COLORS_BGR[pid % len(PERSON_COLORS_BGR)]
        state_col_map = {
            "Focused":      (29,158,117),
            "Distracted":   (39,154,239),
            "Looking Away": (74,75,226),
            "Taking Notes": (221,139,55),
            "Drowsy":       (126,83,212),
        }
        col_s = state_col_map.get(f["state"], (200,200,200))

        x1, y1, x2, y2 = f["bbox"]
        # Outer box (person colour)
        cv2.rectangle(frame, (x1,y1), (x2,y2), col_p, 2)
        # Inner box (state colour), inset 6px
        cv2.rectangle(frame, (x1+6,y1+6), (x2-6,y2-6), col_s, 1)

        # Corner ticks (outer)
        sz = 14
        for (ax, ay, dx, dy) in [
            (x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)
        ]:
            cv2.line(frame,(ax,ay),(ax+dx*sz,ay),col_p,3)
            cv2.line(frame,(ax,ay),(ax,ay+dy*sz),col_p,3)

        # Person label badge
        label = f"P{f['id']} {f['state']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+8,y1),col_p,-1)
        cv2.putText(frame, label, (x1+4, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # Landmark dots (subset)
        lm = f["lm"]
        key_ids = [1,33,61,133,152,234,291,362,454]
        for li in key_ids:
            px = int(lm[li].x * w)
            py = int(lm[li].y * h)
            cv2.circle(frame, (px,py), 2, (93,202,165), -1)

        # Gaze dot (iris centre, left eye)
        iris_pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in LEFT_IRIS]
        ix = int(np.mean([p[0] for p in iris_pts]))
        iy = int(np.mean([p[1] for p in iris_pts]))
        cv2.circle(frame, (ix,iy), 6, (39,159,239), -1)
        cv2.circle(frame, (ix,iy), 6, (255,255,255), 1)

    # LIVE badge
    cv2.rectangle(frame, (8,8), (54,26), (29,158,117), -1)
    cv2.putText(frame,"LIVE",(12,21),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w-80,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(93,202,165),1)
    return frame


# ─────────────────────────────────────────────────────────────────
#  MAIN GUI APPLICATION
# ─────────────────────────────────────────────────────────────────
class AttentionAI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AttentionAI  —  Real-time Attention Analytics")
        self.geometry("1400x860")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        # ── Session state ───────────────────
        self.session_start     = time.time()
        self.is_running        = False
        self.distraction_count = 0
        self.current_state     = "Focused"
        self.confidence        = 91
        self.attention_score   = 82
        self.focus_streak      = 0
        self.people_count      = 0
        self.alert_cooldown    = {}   # state → last alert time
        self.alert_limit       = 8
        self._alert_rows       = []

        # ── Video / capture ─────────────────
        self.cap              = None
        self.video_path       = None   # None = webcam
        self.current_frame_tk = None   # keep reference
        self._cam_img_id      = None

        # ── Session percentages ─────────────
        self.state_counts = {s: 0 for s in STATE_COLORS}
        self.total_frames  = 0

        # ── Timeline (80 slots) ─────────────
        self.timeline = ["Focused"] * 80

        # ── CSV export buffer ────────────────
        self.csv_rows = []

        # ── MediaPipe ───────────────────────
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._build_fonts()
        self._build_ui()

    # ────────────────────────────────────────
    #  FONTS
    # ────────────────────────────────────────
    def _build_fonts(self):
        self.f_title  = ("Segoe UI", 18, "bold")
        self.f_sub    = ("Segoe UI", 10)
        self.f_label  = ("Segoe UI", 9)
        self.f_metric = ("Segoe UI", 26, "bold")
        self.f_body   = ("Segoe UI", 11)
        self.f_small  = ("Segoe UI", 9)
        self.f_mono   = ("Courier New", 10)
        self.f_badge  = ("Segoe UI", 8, "bold")

    # ────────────────────────────────────────
    #  ROOT LAYOUT
    # ────────────────────────────────────────
    def _build_ui(self):
        self._topbar()
        content = tk.Frame(self, bg=BG_DARK)
        content.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self._metrics_row(content)
        mid = tk.Frame(content, bg=BG_DARK)
        mid.pack(fill="both", expand=True, pady=(10, 0))
        mid.columnconfigure(0, weight=55)
        mid.columnconfigure(1, weight=45)
        mid.rowconfigure(0, weight=1)
        left = tk.Frame(mid, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        right = tk.Frame(mid, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        self._camera_panel(left)
        self._right_column(right)
        self._bottom_row(content)

    # ────────────────────────────────────────
    #  TOP BAR
    # ────────────────────────────────────────
    def _topbar(self):
        bar = tk.Frame(self, bg=BG_DARK, height=56)
        bar.pack(fill="x", padx=14, pady=(12, 8))
        bar.pack_propagate(False)

        ic = tk.Canvas(bar, width=36, height=36, bg=BG_DARK, highlightthickness=0)
        ic.pack(side="left")
        ic.create_oval(0,0,36,36, fill=ACCENT_GREEN, outline="")
        ic.create_oval(10,10,26,26, fill=BG_DARK, outline="")
        ic.create_oval(14,14,22,22, fill=ACCENT_GREEN, outline="")

        name_f = tk.Frame(bar, bg=BG_DARK)
        name_f.pack(side="left", padx=(8,0))
        tk.Label(name_f, text="AttentionAI", font=self.f_title,
                 bg=BG_DARK, fg=TEXT_PRIMARY).pack(anchor="w")
        tk.Label(name_f, text="Real-time attention analytics",
                 font=self.f_small, bg=BG_DARK, fg=TEXT_MUTED).pack(anchor="w")

        pill = tk.Frame(bar, bg=ACCENT_GREEN, padx=10, pady=4)
        pill.pack(side="right", anchor="center")
        self.lbl_live_dot = tk.Label(pill, text="● LIVE", font=self.f_badge,
                                     bg=ACCENT_GREEN, fg="white")
        self.lbl_live_dot.pack(side="left")
        self.lbl_session = tk.Label(pill, text="00:00:00",
                                    font=self.f_badge, bg=ACCENT_GREEN, fg="white")
        self.lbl_session.pack(side="left", padx=(6,0))

    # ────────────────────────────────────────
    #  METRIC CARDS
    # ────────────────────────────────────────
    def _metrics_row(self, parent):
        row = tk.Frame(parent, bg=BG_DARK)
        row.pack(fill="x")
        self.metric_vars  = {}
        self.metric_labels = {}
        cards = [
            ("Attention score", "attn_score",    "—",  ACCENT_GREEN),
            ("Focus streak",    "focus_streak",   "0 min", TEXT_PRIMARY),
            ("Distractions",    "distractions",   "0",  ACCENT_AMBER),
            ("People detected", "people_count",   "0",  ACCENT_BLUE),
        ]
        for i, (label, key, init, col) in enumerate(cards):
            card = tk.Frame(row, bg=BG_PANEL, padx=16, pady=12)
            card.pack(side="left", fill="both", expand=True,
                      padx=(0 if i==0 else 8, 0))
            tk.Label(card, text=label, font=self.f_label,
                     bg=BG_PANEL, fg=TEXT_MUTED).pack(anchor="w")
            var = tk.StringVar(value=init)
            self.metric_vars[key] = var
            lbl = tk.Label(card, textvariable=var, font=self.f_metric,
                           bg=BG_PANEL, fg=col)
            lbl.pack(anchor="w")
            self.metric_labels[key] = lbl

    # ────────────────────────────────────────
    #  CAMERA PANEL  (left column)
    # ────────────────────────────────────────
    def _camera_panel(self, parent):
        panel = tk.Frame(parent, bg=BG_PANEL, padx=12, pady=12)
        panel.pack(fill="both", expand=True)

        # Title row with file open button
        hdr = tk.Frame(panel, bg=BG_PANEL)
        hdr.pack(fill="x", pady=(0,8))
        tk.Label(hdr, text="LIVE CAMERA FEED", font=self.f_badge,
                 bg=BG_PANEL, fg=TEXT_MUTED).pack(side="left")
        self._btn(hdr, "📁 Open Video", BG_SURFACE2,
                  self._on_open_video).pack(side="right")

        # Camera canvas
        self.cam_canvas = tk.Canvas(panel, bg="#0A0F14", height=310,
                                    highlightthickness=1,
                                    highlightbackground=BORDER)
        self.cam_canvas.pack(fill="x")
        self._draw_placeholder()

        # Per-person cards container
        self.persons_frame = tk.Frame(panel, bg=BG_PANEL)
        self.persons_frame.pack(fill="x", pady=(6,0))
        self._person_card_widgets = {}

        # Info row
        info = tk.Frame(panel, bg=BG_PANEL)
        info.pack(fill="x", pady=(8,0))
        self.info_vars = {}
        for label, key, init in [
            ("Head pose", "head_pose", "—"),
            ("Eye state", "eye_state", "—"),
            ("Posture",   "posture",   "—"),
        ]:
            box = tk.Frame(info, bg=BG_SURFACE, padx=10, pady=7)
            box.pack(side="left", fill="both", expand=True,
                     padx=(0,6) if key != "posture" else 0)
            tk.Label(box, text=label, font=self.f_small,
                     bg=BG_SURFACE, fg=TEXT_GREEN).pack(anchor="w")
            var = tk.StringVar(value=init)
            self.info_vars[key] = var
            tk.Label(box, textvariable=var, font=self.f_body,
                     bg=BG_SURFACE, fg=TEXT_PRIMARY).pack(anchor="w")

        # Buttons
        btn_row = tk.Frame(panel, bg=BG_PANEL)
        btn_row.pack(fill="x", pady=(10,0))
        self._btn(btn_row, "▶  Start Detection", ACCENT_GREEN,
                  self._on_start).pack(side="left", fill="x", expand=True)
        self._btn(btn_row, "⏸  Pause", BG_SURFACE2,
                  self._on_pause).pack(side="left", fill="x",
                                       expand=True, padx=(8,8))
        self._btn(btn_row, "📄  Export CSV", ACCENT_BLUE,
                  self._on_export).pack(side="left", fill="x", expand=True)

    def _draw_placeholder(self):
        c = self.cam_canvas
        c.delete("all")
        c.update_idletasks()
        W = c.winfo_reqwidth() or 700
        H = 310
        c.create_rectangle(0,0,W,H, fill="#0A0F14", outline="")
        c.create_text(W//2, H//2-16, text="📷",
                      font=("Segoe UI", 28), fill=TEXT_MUTED)
        c.create_text(W//2, H//2+20, text="Press ▶ Start Detection",
                      font=self.f_body, fill=TEXT_MUTED)

    # ────────────────────────────────────────
    #  RIGHT COLUMN
    # ────────────────────────────────────────
    def _right_column(self, parent):
        # Attention States panel
        sp = tk.Frame(parent, bg=BG_PANEL, padx=12, pady=12)
        sp.pack(fill="x")
        tk.Label(sp, text="ATTENTION STATES", font=self.f_badge,
                 bg=BG_PANEL, fg=TEXT_MUTED).pack(anchor="w")

        self.state_bars   = {}
        self.state_pct_vars = {}
        state_data = [
            ("Focused",      68, ACCENT_GREEN),
            ("Distracted",   14, ACCENT_AMBER),
            ("Looking Away", 10, ACCENT_RED),
            ("Taking Notes",  6, ACCENT_BLUE),
            ("Drowsy",        2, ACCENT_PINK),
        ]
        for state, pct, color in state_data:
            row = tk.Frame(sp, bg=BG_PANEL)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=state, font=self.f_small, width=13,
                     anchor="w", bg=BG_PANEL, fg=TEXT_PRIMARY).pack(side="left")
            bar_bg = tk.Frame(row, bg=BG_SURFACE2, height=8)
            bar_bg.pack(side="left", fill="x", expand=True)
            bar_bg.pack_propagate(False)
            bar = tk.Frame(bar_bg, bg=color, height=8)
            bar.place(x=0, y=0, relheight=1, relwidth=pct/100)
            self.state_bars[state] = bar
            var = tk.StringVar(value=f"{pct}%")
            self.state_pct_vars[state] = var
            tk.Label(row, textvariable=var, font=self.f_small, width=5,
                     anchor="e", bg=BG_PANEL, fg=TEXT_MUTED).pack(side="left")

        # Alerts panel
        ap = tk.Frame(parent, bg=BG_PANEL, padx=12, pady=12)
        ap.pack(fill="both", expand=True, pady=(10,0))
        tk.Label(ap, text="ALERTS", font=self.f_badge,
                 bg=BG_PANEL, fg=TEXT_MUTED).pack(anchor="w")
        self.alerts_frame = tk.Frame(ap, bg=BG_PANEL)
        self.alerts_frame.pack(fill="both", expand=True, pady=(6,0))

        self._add_alert("info", "System ready — press Start Detection")

    def _add_alert(self, kind, msg):
        cmap = {
            "warn":  (ACCENT_AMBER, "#2A2010"),
            "ok":    (ACCENT_GREEN, "#0D1F18"),
            "info":  (ACCENT_BLUE,  "#0D1828"),
            "error": (ACCENT_RED,   "#2A0F0F"),
        }
        dot_col, bg = cmap.get(kind, (TEXT_MUTED, BG_SURFACE))

        # Limit to 8 alerts
        if len(self._alert_rows) >= self.alert_limit:
            oldest = self._alert_rows.pop(0)
            oldest.destroy()

        row = tk.Frame(self.alerts_frame, bg=bg, padx=8, pady=5)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="●", fg=dot_col, bg=bg,
                 font=self.f_body).pack(side="left", padx=(0,6))
        tk.Label(row, text=msg, font=self.f_small,
                 bg=bg, fg=TEXT_PRIMARY, wraplength=260,
                 justify="left").pack(side="left", fill="x", expand=True)
        t = datetime.now().strftime("%H:%M:%S")
        tk.Label(row, text=t, font=self.f_small,
                 bg=bg, fg=TEXT_MUTED).pack(side="right")
        self._alert_rows.append(row)

    def _maybe_alert(self, state):
        """Fire alert with 5-sec cooldown per state."""
        now = time.time()
        last = self.alert_cooldown.get(state, 0)
        if now - last < 5:
            return
        self.alert_cooldown[state] = now
        if state == "Drowsy":
            self._add_alert("error", "Drowsiness detected! Consider a short break.")
        elif state == "Distracted":
            self._add_alert("warn", "Distraction detected — refocus reminder sent")
        elif state == "Looking Away":
            self._add_alert("warn", "Student looking away from screen")
        elif state == "Focused":
            self._add_alert("ok", "Focus restored")

    # ────────────────────────────────────────
    #  BOTTOM ROW
    # ────────────────────────────────────────
    def _bottom_row(self, parent):
        row = tk.Frame(parent, bg=BG_DARK)
        row.pack(fill="x", pady=(10,0))

        # Timeline
        tl = tk.Frame(row, bg=BG_PANEL, padx=12, pady=12)
        tl.pack(side="left", fill="both", expand=True)
        tk.Label(tl, text="ATTENTION TIMELINE (LAST 80 FRAMES)",
                 font=self.f_badge, bg=BG_PANEL, fg=TEXT_MUTED).pack(anchor="w")
        self.tl_canvas = tk.Canvas(tl, bg=BG_PANEL, height=36, highlightthickness=0)
        self.tl_canvas.pack(fill="x", pady=(6,0))
        leg = tk.Frame(tl, bg=BG_PANEL)
        leg.pack(anchor="w", pady=(4,0))
        for state, col in STATE_COLORS.items():
            f = tk.Frame(leg, bg=BG_PANEL)
            f.pack(side="left", padx=(0,12))
            tk.Label(f, text="■", fg=col, bg=BG_PANEL, font=self.f_small).pack(side="left")
            tk.Label(f, text=state, font=self.f_small,
                     bg=BG_PANEL, fg=TEXT_MUTED).pack(side="left")

        # Libraries
        lib_panel = tk.Frame(row, bg=BG_PANEL, padx=12, pady=12)
        lib_panel.pack(side="left", fill="both", padx=(10,0))
        tk.Label(lib_panel, text="LIBRARIES", font=self.f_badge,
                 bg=BG_PANEL, fg=TEXT_MUTED).pack(anchor="w")
        libs = [
            ("OpenCV",       ACCENT_GREEN), ("MediaPipe",   ACCENT_GREEN),
            ("NumPy",        ACCENT_AMBER), ("Pillow",      ACCENT_AMBER),
            ("Tkinter",      "#AFA9EC"),    ("csv/os",      ACCENT_AMBER),
            ("threading",    ACCENT_BLUE),  ("FaceMesh",    ACCENT_GREEN),
            ("EAR",          ACCENT_PINK),  ("GazeTrack",   ACCENT_BLUE),
            ("HeadPose",     ACCENT_RED),   ("Classifier",  ACCENT_GREEN),
        ]
        cf = tk.Frame(lib_panel, bg=BG_PANEL)
        cf.pack(fill="x", pady=(6,0))
        for i, (lib, col) in enumerate(libs):
            tk.Label(cf, text=lib, font=self.f_small, bg=BG_SURFACE2,
                     fg=col, padx=8, pady=4, relief="flat"
                    ).grid(row=i//4, column=i%4, padx=3, pady=3, sticky="ew")

        self._draw_timeline()

    def _draw_timeline(self):
        c = self.tl_canvas
        c.delete("all")
        c.update_idletasks()
        W  = c.winfo_width() or 600
        n  = len(self.timeline)
        sw = W / n
        for i, state in enumerate(self.timeline):
            col = STATE_COLORS.get(state, TEXT_MUTED)
            c.create_rectangle(i*sw, 0, i*sw+sw-1, 36, fill=col, outline="")

    # ────────────────────────────────────────
    #  BUTTON HELPER
    # ────────────────────────────────────────
    def _btn(self, parent, text, bg, command):
        return tk.Button(parent, text=text, bg=bg, fg="white",
                         font=self.f_body, relief="flat",
                         activebackground=bg, activeforeground="white",
                         cursor="hand2", command=command, padx=10, pady=7)

    # ────────────────────────────────────────
    #  BUTTON ACTIONS
    # ────────────────────────────────────────
    def _on_start(self):
        if self.is_running:
            return
        self.is_running    = True
        self.session_start = time.time()
        self._add_alert("ok", "Detection started — camera / video active")
        t = threading.Thread(target=self._detection_loop, daemon=True)
        t.start()
        self._tick_session_timer()

    def _on_pause(self):
        self.is_running = not self.is_running
        status = "resumed" if self.is_running else "paused"
        self._add_alert("info", f"Session {status}")

    def _on_open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv *.webm *.flv"),
                       ("All","*.*")])
        if path:
            self.video_path = path
            name = os.path.basename(path)
            self._add_alert("info", f"Video loaded: {name}")

    def _on_export(self):
        if not self.csv_rows:
            self._add_alert("warn", "No data to export yet — start detection first")
            return
        fname = f"attention_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(fname, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Timestamp","PersonID","State","Conf%","EAR","Yaw°","Pitch°"])
            writer.writerows(self.csv_rows)
        self._add_alert("ok", f"Exported {len(self.csv_rows)} rows → {fname}")

    # ────────────────────────────────────────
    #  SESSION TIMER TICK (main thread)
    # ────────────────────────────────────────
    def _tick_session_timer(self):
        if not self.is_running:
            self.after(1000, self._tick_session_timer)
            return
        elapsed = int(time.time() - self.session_start)
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)
        self.lbl_session.config(text=f"{h:02}:{m:02}:{s:02}")
        self.after(1000, self._tick_session_timer)

    # ────────────────────────────────────────
    #  DETECTION LOOP (background thread)
    # ────────────────────────────────────────
    def _detection_loop(self):
        # Open capture
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.after(0, self._add_alert, "error",
                       "Cannot open camera / video file")
            return

        FRAME_TIME = 1 / 30  # ~30 fps

        while self.is_running:
            t0 = time.time()
            ret, frame = self.cap.read()

            # Loop video if it ends
            if not ret:
                if self.video_path:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # Resize for processing speed
            h_orig, w_orig = frame.shape[:2]
            scale = min(1.0, 720 / max(w_orig, h_orig))
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Detect faces
            faces = detect_faces(frame, self.face_mesh)

            # Draw overlays
            frame_out = draw_overlays(frame.copy(), faces)

            # Aggregate session stats
            self.total_frames += 1
            if faces:
                primary = faces[0]
                st = primary["state"]
                self.state_counts[st] += 1
                if st in ("Distracted","Looking Away","Drowsy"):
                    self.distraction_count += 1

                # Focus streak from timeline
                streak_frames = 0
                for s in reversed(self.timeline):
                    if s == "Focused":
                        streak_frames += 1
                    else:
                        break
                streak_min = round(streak_frames / 30, 1)

                # Compute attention score (focused%)
                attn = int(self.state_counts["Focused"] /
                           max(1, self.total_frames) * 100)

                # CSV row
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                for f in faces:
                    self.csv_rows.append([
                        ts, f"P{f['id']}", f["state"], f["conf"],
                        f["ear"], f["yaw"], f["pitch"],
                    ])

                # Timeline
                self.timeline.append(st)
                if len(self.timeline) > 80:
                    self.timeline.pop(0)

                # Head pose label
                yaw   = primary["yaw"]
                pitch = primary["pitch"]
                if abs(yaw) < 5 and abs(pitch) < 5:
                    pose_label = "Front-facing"
                elif yaw < -10:
                    pose_label = "Slightly left"
                elif yaw > 10:
                    pose_label = "Slightly right"
                elif pitch < -10:
                    pose_label = "Head up"
                else:
                    pose_label = "Head down"

                ear = primary["ear"]
                if ear < 0.15:
                    eye_label = "Closed"
                elif ear < 0.22:
                    eye_label = "Narrowed"
                else:
                    eye_label = "Open"

                posture_label = "Upright" if abs(pitch) < 15 else "Leaning"

                self.after(0, self._update_ui,
                           frame_out, faces, st, primary["conf"],
                           attn, streak_min, pose_label,
                           eye_label, posture_label)
            else:
                # No face
                self.after(0, self._update_frame_only, frame_out)

            # Cap at 30 fps
            elapsed = time.time() - t0
            sleep_t = max(0, FRAME_TIME - elapsed)
            time.sleep(sleep_t)

        if self.cap:
            self.cap.release()
            self.cap = None

    # ────────────────────────────────────────
    #  UI UPDATE (main thread)
    # ────────────────────────────────────────
    def _update_frame_only(self, frame):
        self._render_frame(frame)

    def _update_ui(self, frame, faces, state, conf,
                   attn, streak, pose, eye, posture):
        col = STATE_COLORS.get(state, TEXT_PRIMARY)

        # Metric cards
        attn_col = (ACCENT_GREEN if attn >= 75 else
                    ACCENT_AMBER if attn >= 50 else ACCENT_RED)
        self.metric_vars["attn_score"].set(f"{attn}%")
        self.metric_labels["attn_score"].config(fg=attn_col)
        self.metric_vars["focus_streak"].set(f"{streak} min")
        self.metric_vars["distractions"].set(str(self.distraction_count))
        self.metric_vars["people_count"].set(str(len(faces)))

        # Info boxes
        self.info_vars["head_pose"].set(pose)
        self.info_vars["eye_state"].set(eye)
        self.info_vars["posture"].set(posture)

        # State bars
        tot = max(1, self.total_frames)
        for s, bar in self.state_bars.items():
            pct = self.state_counts[s] / tot
            bar.place_configure(relwidth=min(1.0, pct))
            self.state_pct_vars[s].set(f"{int(pct*100)}%")

        # Per-person cards
        self._update_person_cards(faces)

        # Alert
        self._maybe_alert(state)

        # Timeline
        self._draw_timeline()

        # Frame
        self._render_frame(frame)

    def _render_frame(self, frame):
        """Convert OpenCV frame to Tkinter image and display on canvas."""
        c = self.cam_canvas
        c.update_idletasks()
        cw = c.winfo_width()  or 700
        ch = c.winfo_height() or 310
        if cw < 10 or ch < 10:
            return

        # Resize frame to fit canvas
        h, w = frame.shape[:2]
        scale = min(cw/w, ch/h)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(frame, (nw, nh))

        img_rgb  = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(img_rgb)
        tk_img   = ImageTk.PhotoImage(pil_img)
        self.current_frame_tk = tk_img   # keep reference!

        c.delete("all")
        c.create_rectangle(0,0,cw,ch, fill="#0A0F14", outline="")
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        c.create_image(ox, oy, anchor="nw", image=tk_img)

    def _update_person_cards(self, faces):
        """Show/hide per-person cards dynamically."""
        # Destroy cards for people no longer in frame
        current_ids = {f["id"] for f in faces}
        for pid in list(self._person_card_widgets.keys()):
            if pid not in current_ids:
                self._person_card_widgets[pid].destroy()
                del self._person_card_widgets[pid]

        # Create / update cards
        for f in faces:
            pid   = f["id"]
            col   = PERSON_COLORS_HEX[(pid-1) % len(PERSON_COLORS_HEX)]
            s_col = STATE_COLORS.get(f["state"], TEXT_PRIMARY)
            label = (f"P{pid}  {f['state']}  conf:{f['conf']}%  "
                     f"EAR:{f['ear']}  yaw:{f['yaw']}°")
            if pid not in self._person_card_widgets:
                card = tk.Label(self.persons_frame, text=label,
                                font=self.f_small, bg=BG_SURFACE2, fg=col,
                                padx=8, pady=4, anchor="w", justify="left")
                card.pack(fill="x", pady=2)
                self._person_card_widgets[pid] = card
            else:
                self._person_card_widgets[pid].config(text=label, fg=s_col)


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = AttentionAI()
    app.mainloop()