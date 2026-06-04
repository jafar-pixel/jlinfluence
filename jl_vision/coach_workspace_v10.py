#!/usr/bin/env python3
"""
JL Vision Coach Workspace v10
PyQt5 video player + TensorFlow MoveNet pose + YOLO athlete detection

Environment : jl-tf-env  (NOT jl-vision-env)
Install     : pip install PyQt5 tensorflow tensorflow-hub ultralytics opencv-python numpy
Usage       : python coach_workspace_v10.py videos/test_video.mp4
"""

import re
import sys
import math
import csv
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QHBoxLayout, QVBoxLayout, QGroupBox, QTextEdit,
    QFileDialog, QMessageBox, QInputDialog, QSizePolicy, QFrame,
    QStackedWidget,
)
from PyQt5.QtCore import Qt, QTimer, QByteArray, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
try:
    from PyQt5.QtSvg import QSvgWidget
    _HAS_SVG = True
except ImportError:
    _HAS_SVG = False


# ------------------------------------------------------------------------------
# MoveNet
# ------------------------------------------------------------------------------

MOVENET_URLS = {
    'lightning': 'https://tfhub.dev/google/movenet/singlepose/lightning/4',
    'thunder':   'https://tfhub.dev/google/movenet/singlepose/thunder/4',
}
INPUT_SIZES = {'lightning': 192, 'thunder': 256}

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

SKELETON_EDGES = [
    ('left_shoulder',  'right_shoulder'),
    ('left_shoulder',  'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow',     'left_wrist'),
    ('right_elbow',    'right_wrist'),
    ('left_shoulder',  'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip',       'right_hip'),
    ('left_hip',       'left_knee'),
    ('right_hip',      'right_knee'),
    ('left_knee',      'left_ankle'),
    ('right_knee',     'right_ankle'),
]

# ── Muscle heat-map ───────────────────────────────────────────────────────────
# (threshold 0-1, hex color)  — REST → LOW → MED → HIGH → PEAK → MAX
HEAT_LEVELS = [
    (0.00, '#0033FF'), (0.15, '#00C8FF'), (0.35, '#00E676'),
    (0.55, '#FFB800'), (0.75, '#FF5500'), (0.90, '#FF0000'),
]

# Maps internal muscle key → SVG element id in body_map.svg
SVG_MUSCLE_IDS = {
    'glute_r':      'glute-r',      'glute_l':      'glute-l',
    'quad_r':       'quad-r',       'quad_l':       'quad-l',
    'hamstring_r':  'hamstring-r',  'hamstring_l':  'hamstring-l',
    'calf_r':       'calf-r',       'calf_l':       'calf-l',
    'hip_flexor_r': 'hip-flexor-r', 'hip_flexor_l': 'hip-flexor-l',
    'erector':      'erector',      'core':         'core',
    'shoulder_r':   'shoulder-r',   'shoulder_l':   'shoulder-l',
}

# CNS co-activation chains: grouped muscles that fire together
CNS_CHAINS = {
    'Posterior': ['glute_r', 'glute_l', 'hamstring_r', 'hamstring_l', 'calf_r', 'calf_l'],
    'Anterior':  ['quad_r', 'quad_l', 'hip_flexor_r', 'hip_flexor_l'],
    'Core':      ['core', 'erector'],
    'Lateral':   ['shoulder_r', 'shoulder_l'],
}

# EMS / injury risk: (muscle_key, display_label, peak_threshold)
EMS_ZONES = [
    ('glute_r',      'Glute Max R',     0.80),
    ('hamstring_r',  'Hamstring R',     0.75),
    ('calf_r',       'Gastrocnemius R', 0.85),
    ('hip_flexor_r', 'Hip Flexor R',    0.70),
]


class MoveNet:
    def __init__(self, variant='lightning'):
        print(f'Loading MoveNet {variant} ...')
        self._size = INPUT_SIZES[variant]
        module = hub.load(MOVENET_URLS[variant])
        self._infer = module.signatures['serving_default']
        print('MoveNet ready.')

    def estimate(self, bgr, box=None):
        """Return (17,3) float32 array of [y_norm, x_norm, conf] in full-frame coords,
        or None if the crop is empty."""
        h, w = bgr.shape[:2]
        if box is not None:
            x1, y1, x2, y2 = box
            pad = 30
            x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
            x2, y2 = min(x2 + pad, w), min(y2 + pad, h)
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            ch, cw = crop.shape[:2]
            ox, oy = x1, y1
        else:
            crop, cw, ch, ox, oy = bgr, w, h, 0, 0

        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._size, self._size))
        tensor = tf.cast(tf.expand_dims(img, 0), tf.int32)
        raw = self._infer(input=tensor)['output_0'].numpy()[0, 0]  # (17,3) crop-relative

        result = np.empty((17, 3), np.float32)
        for i in range(17):
            ky, kx, conf = raw[i]
            result[i] = [(oy + ky * ch) / h, (ox + kx * cw) / w, conf]
        return result


# ------------------------------------------------------------------------------
# YOLO helpers
# ------------------------------------------------------------------------------

def detect_people(yolo, bgr, conf=0.35):
    boxes = []
    for r in yolo(bgr, verbose=False, conf=conf):
        for b in r.boxes:
            if int(b.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append({'box': (x1, y1, x2, y2), 'conf': float(b.conf[0])})
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (union + 1e-9)


# ------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------

def angle_at(a, vertex, c):
    """Angle in degrees at 'vertex' between rays vertex->a and vertex->c."""
    ax, ay = a[0] - vertex[0], a[1] - vertex[1]
    cx, cy = c[0] - vertex[0], c[1] - vertex[1]
    dot = ax * cx + ay * cy
    mag = math.hypot(ax, ay) * math.hypot(cx, cy) + 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def kp_xy(kps, name, w, h, thresh=0.15):
    """Return (x, y) pixel or None."""
    y_n, x_n, conf = kps[KP[name]]
    if conf < thresh:
        return None
    return (int(x_n * w), int(y_n * h))


# ------------------------------------------------------------------------------
# Stride detector
# ------------------------------------------------------------------------------

class StrideDetector:
    """Detects ground-contact events from ankle keypoints and computes
    step frequency and stride length."""

    DEBOUNCE_SEC = 0.18   # min seconds between contacts on same foot
    WINDOW       = 9      # frames used for local-max detection

    def __init__(self, fps):
        self.fps = fps
        self._y   = {'left': deque(maxlen=30), 'right': deque(maxlen=30)}
        self._x   = {'left': deque(maxlen=30), 'right': deque(maxlen=30)}
        self._fr  = {'left': deque(maxlen=30), 'right': deque(maxlen=30)}
        self._contacts = {'left': [], 'right': []}   # (frame, x_px)
        # Published outputs
        self.stride_length_m  = None
        self.step_freq_hz     = None

    def update(self, frame_idx, kps, w, h, pixels_per_meter=None):
        for side in ('left', 'right'):
            pt = kp_xy(kps, f'{side}_ankle', w, h, thresh=0.2)
            if pt is None:
                continue
            self._x[side].append(pt[0])
            self._y[side].append(pt[1])
            self._fr[side].append(frame_idx)
        self._detect(pixels_per_meter)

    def _detect(self, ppm):
        deb = self.DEBOUNCE_SEC * self.fps
        half = self.WINDOW // 2
        for side in ('left', 'right'):
            ys = list(self._y[side])
            xs = list(self._x[side])
            fs = list(self._fr[side])
            if len(ys) < self.WINDOW:
                continue
            mid = len(ys) - half - 1
            if mid < half:
                continue
            y_mid = ys[mid]
            # local max in y (screen: larger y = closer to ground)
            if not all(y_mid >= ys[i] - 2 for i in range(len(ys))):
                continue
            f_mid = fs[mid]
            x_mid = xs[mid]
            last = self._contacts[side]
            if last and (f_mid - last[-1][0]) < deb:
                continue
            self._contacts[side].append((f_mid, x_mid))
            # keep only last 6 contacts per side
            self._contacts[side] = self._contacts[side][-6:]

        # step frequency
        all_c = sorted(
            self._contacts['left'][-3:] + self._contacts['right'][-3:],
            key=lambda c: c[0]
        )
        if len(all_c) >= 3:
            span = all_c[-1][0] - all_c[0][0]
            if span > 0:
                self.step_freq_hz = (len(all_c) - 1) / (span / self.fps)

        # stride length (same-side consecutive contacts)
        for side in ('left', 'right'):
            if len(self._contacts[side]) >= 2 and ppm:
                dx = abs(self._contacts[side][-1][1] -
                         self._contacts[side][-2][1])
                self.stride_length_m = dx / ppm
                break

    def reset(self):
        for side in ('left', 'right'):
            self._y[side].clear(); self._x[side].clear()
            self._fr[side].clear()
        self._contacts = {'left': [], 'right': []}
        self.stride_length_m = None
        self.step_freq_hz = None


# ------------------------------------------------------------------------------
# Muscle activation engine
# ------------------------------------------------------------------------------

class MuscleActivation:
    """Maps MoveNet keypoints + joint angles to per-muscle activation (0–1)."""

    def __init__(self):
        self._prev_ankle = {'left': None, 'right': None}

    def compute(self, kps, angles, w, h):
        hip_r  = angles.get('HIP-R',  180)
        hip_l  = angles.get('HIP-L',  180)
        knee_r = angles.get('KNEE-R', 180)
        knee_l = angles.get('KNEE-L', 180)

        act = {}
        # Glutes — hip extension (large angle)
        act['glute_r'] = self._norm(hip_r,  90, 175)
        act['glute_l'] = self._norm(hip_l,  90, 175)
        # Hip flexors — hip flexion (small angle → high activation)
        act['hip_flexor_r'] = self._norm(175 - hip_r,  0, 85)
        act['hip_flexor_l'] = self._norm(175 - hip_l,  0, 85)
        # Quads — knee extension
        act['quad_r'] = self._norm(knee_r, 90, 175)
        act['quad_l'] = self._norm(knee_l, 90, 175)
        # Hamstrings — knee flexion
        act['hamstring_r'] = self._norm(175 - knee_r, 0, 85)
        act['hamstring_l'] = self._norm(175 - knee_l, 0, 85)
        # Calves — ankle vertical velocity
        act['calf_r'] = self._ankle_act(kps, 'right', w, h)
        act['calf_l'] = self._ankle_act(kps, 'left',  w, h)
        # Core / erector — torso lean magnitude
        torso = angles.get('TORSO_LEAN', 0.0)
        act['core']    = min(1.0, torso / 35.0)
        act['erector'] = min(1.0, torso / 25.0)
        # Shoulders — keypoint confidence proxy
        for side, key in (('right', 'shoulder_r'), ('left', 'shoulder_l')):
            _, _, conf = kps[KP[f'{side}_shoulder']]
            act[key] = float(conf) if conf > 0.4 else 0.0
        return act

    @staticmethod
    def _norm(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-9)))

    def _ankle_act(self, kps, side, w, h):
        pt = kp_xy(kps, f'{side}_ankle', w, h, thresh=0.2)
        prev = self._prev_ankle[side]
        self._prev_ankle[side] = pt
        if pt and prev:
            return min(1.0, abs(pt[1] - prev[1]) / 20.0)
        return 0.0


def heat_color(level):
    """Return hex color for activation level 0-1."""
    for threshold, color in reversed(HEAT_LEVELS):
        if level >= threshold:
            return color
    return HEAT_LEVELS[0][1]


# ------------------------------------------------------------------------------
# Body Map widget
# ------------------------------------------------------------------------------

class BodyMapWidget(QWidget):
    """SVG body map with heat-mapped muscle fills driven by POSE data."""

    SVG_PATH = Path(__file__).parent / 'body_map.svg'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('background: #0d0f14;')
        self._activations = {}
        self._svg_raw = None
        self._ems_labels = {}

        col = QVBoxLayout(self)
        col.setContentsMargins(6, 6, 6, 6)
        col.setSpacing(6)

        # Header
        hdr = QLabel('LIVE MUSCLE ACTIVATION')
        hdr.setStyleSheet('color: #00ffd0; font-size: 11px; font-weight: bold; letter-spacing: 2px;')
        hdr.setAlignment(Qt.AlignCenter)
        col.addWidget(hdr)

        # SVG viewer (or placeholder)
        if _HAS_SVG:
            self._svg_view = QSvgWidget()
            self._svg_view.setMinimumHeight(220)
            self._svg_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            col.addWidget(self._svg_view, 3)
        else:
            self._svg_view = None

        self._placeholder = QLabel(
            'Add body_map.svg to jl_vision/ folder\n'
            'to enable live muscle heat-mapping.\n\n'
            'SVG muscle path IDs:\n'
            'glute-r  quad-r  hamstring-r  calf-r\n'
            'glute-l  quad-l  hamstring-l  calf-l\n'
            'hip-flexor-r  hip-flexor-l\n'
            'erector  core  shoulder-r  shoulder-l'
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            'color: #445; font-size: 11px; background: #131820; '
            'border: 1px dashed #00ffd025; border-radius: 6px; padding: 12px;')
        self._placeholder.setWordWrap(True)
        col.addWidget(self._placeholder, 3)

        # CNS chain bars
        col.addWidget(self._build_cns_section())

        # EMS / injury zones
        col.addWidget(self._build_ems_section())

        # Load SVG
        if self.SVG_PATH.exists():
            with open(self.SVG_PATH, 'r', encoding='utf-8') as f:
                self._svg_raw = f.read()
            self._placeholder.hide()
        elif self._svg_view:
            self._svg_view.hide()

    def _build_cns_section(self):
        grp = QGroupBox('CNS Activation Chains')
        grp.setStyleSheet(
            'QGroupBox { color: #00ffd0; font-size: 11px; font-weight: bold; '
            'border: 1px solid #1a2035; border-radius: 6px; margin-top: 8px; padding-top: 8px; }'
        )
        v = QVBoxLayout(grp)
        v.setSpacing(5)
        self._chain_bars = {}
        for chain_name in CNS_CHAINS:
            row = QHBoxLayout()
            lbl = QLabel(chain_name)
            lbl.setStyleSheet('color: #7a8a9a; font-size: 11px;')
            lbl.setFixedWidth(62)
            bar = QLabel()
            bar.setFixedHeight(8)
            bar.setStyleSheet('background: #1a1f2e; border-radius: 4px;')
            pct_lbl = QLabel('0%')
            pct_lbl.setStyleSheet('color: #556; font-size: 11px;')
            pct_lbl.setFixedWidth(30)
            pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(lbl)
            row.addWidget(bar, 1)
            row.addWidget(pct_lbl)
            v.addLayout(row)
            self._chain_bars[chain_name] = (bar, pct_lbl)
        return grp

    def _build_ems_section(self):
        grp = QGroupBox('EMS / Injury Risk Zones')
        grp.setStyleSheet(
            'QGroupBox { color: #FFB800; font-size: 11px; font-weight: bold; '
            'border: 1px solid #1a2035; border-radius: 6px; margin-top: 8px; padding-top: 8px; }'
        )
        v = QVBoxLayout(grp)
        v.setSpacing(4)
        self._ems_labels = {}
        for muscle_key, label, threshold in EMS_ZONES:
            row = QHBoxLayout()
            name_lbl = QLabel(label)
            name_lbl.setStyleSheet('color: #7a8a9a; font-size: 11px;')
            name_lbl.setFixedWidth(110)
            status_lbl = QLabel('OPTIMAL')
            status_lbl.setStyleSheet('color: #00E676; font-size: 11px; font-weight: bold;')
            status_lbl.setFixedWidth(70)
            row.addWidget(name_lbl)
            row.addWidget(status_lbl)
            row.addStretch()
            ems_btn = QPushButton('EMS')
            ems_btn.setFixedSize(42, 22)
            ems_btn.setStyleSheet(
                'QPushButton { font-size: 10px; font-weight: bold; '
                'background: #1a1020; border: 1px solid #FF3D3D40; '
                'border-radius: 4px; color: #FF3D3D; padding: 0; }'
                'QPushButton:hover { background: #FF3D3D; color: #0d0f14; }'
            )
            row.addWidget(ems_btn)
            v.addLayout(row)
            self._ems_labels[muscle_key] = status_lbl
        return grp

    def update_activations(self, activations):
        self._activations = activations
        self._refresh_svg()
        self._refresh_chains()
        self._refresh_ems()

    def _refresh_svg(self):
        if not self._svg_raw or not self._svg_view:
            return
        rules = [
            f'#{svg_id}{{fill:{heat_color(self._activations.get(k, 0.0))};'
            f'fill-opacity:{0.35 + self._activations.get(k, 0.0) * 0.60:.2f};}}'
            for k, svg_id in SVG_MUSCLE_IDS.items()
        ]
        style = '<style>' + ''.join(rules) + '</style>'
        modified = re.sub(r'(<svg[^>]*>)', r'\1' + style, self._svg_raw, count=1)
        self._svg_view.load(QByteArray(modified.encode('utf-8')))

    def _refresh_chains(self):
        for chain_name, muscles in CNS_CHAINS.items():
            vals = [self._activations.get(m, 0.0) for m in muscles]
            avg = sum(vals) / len(vals) if vals else 0.0
            pct = int(avg * 100)
            color = heat_color(avg)
            bar, lbl = self._chain_bars[chain_name]
            stop = f'{avg:.3f}'
            stop2 = f'{min(1.0, avg + 0.001):.3f}'
            bar.setStyleSheet(
                f'background: qlineargradient(x1:0,y1:0,x2:1,y2:0,'
                f'stop:0 {color},stop:{stop} {color},'
                f'stop:{stop2} #1a1f2e,stop:1 #1a1f2e);'
                f'border-radius:4px;'
            )
            lbl.setText(f'{pct}%')

    def _refresh_ems(self):
        for muscle_key, _, threshold in EMS_ZONES:
            level = self._activations.get(muscle_key, 0.0)
            lbl = self._ems_labels.get(muscle_key)
            if not lbl:
                continue
            if level >= 0.90:
                lbl.setText('MAX'); lbl.setStyleSheet('color:#FF0000;font-size:11px;font-weight:bold;')
            elif level >= threshold:
                lbl.setText('AT RISK'); lbl.setStyleSheet('color:#FF3D3D;font-size:11px;font-weight:bold;')
            elif level >= threshold * 0.85:
                lbl.setText('MONITOR'); lbl.setStyleSheet('color:#FFB800;font-size:11px;font-weight:bold;')
            elif level >= 0.60:
                lbl.setText('PEAK'); lbl.setStyleSheet('color:#FF5500;font-size:11px;font-weight:bold;')
            else:
                lbl.setText('OPTIMAL'); lbl.setStyleSheet('color:#00E676;font-size:11px;font-weight:bold;')


# ------------------------------------------------------------------------------
# Qt stylesheet
# ------------------------------------------------------------------------------

STYLE = """
* { font-family: "Segoe UI", Arial, sans-serif; }
QMainWindow, QWidget { background: #0d0f14; color: #dde; }
QLabel { color: #ccd; font-size: 13px; }
QPushButton {
    background: #131820; color: #00ffd0;
    border: 1px solid #00ffd0; border-radius: 5px;
    padding: 5px 12px; font-size: 13px; font-weight: bold;
}
QPushButton:hover  { background: #00ffd0; color: #0d0f14; }
QPushButton:pressed { background: #00b89a; }
QPushButton:checked { background: #00ffd0; color: #0d0f14; }
QPushButton:disabled { border-color: #333; color: #444; }
QSlider::groove:horizontal { background: #1a1f2e; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #00ffd0; width: 16px; height: 16px;
    margin: -5px 0; border-radius: 8px;
}
QSlider::sub-page:horizontal { background: #00ffd0; border-radius: 3px; }
QGroupBox {
    border: 1px solid #222840; border-radius: 8px;
    margin-top: 10px; padding-top: 10px;
    font-size: 13px; color: #00ffd0; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QTextEdit {
    background: #0a0c12; color: #aab; border: 1px solid #222840;
    border-radius: 4px; padding: 6px; font-size: 13px;
}
"""


# ------------------------------------------------------------------------------
# Metric label
# ------------------------------------------------------------------------------

class MetricRow(QWidget):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 4, 6, 4)
        self._lbl = QLabel(label)
        self._lbl.setStyleSheet('color: #7a8a9a; font-size: 14px;')
        self._lbl.setFixedWidth(140)
        self._val = QLabel('--')
        self._val.setStyleSheet('color: #dde; font-size: 15px; font-weight: bold;')
        self._val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._val.setWordWrap(False)
        self._val.setMaximumWidth(180)
        row.addWidget(self._lbl)
        row.addWidget(self._val, 1)

    def set(self, text, color='#dde'):
        self._val.setText(str(text))
        self._val.setStyleSheet(f'color: {color}; font-size: 16px; font-weight: bold;')

    def pending(self):
        self.set('pending', '#f5a623')


# ------------------------------------------------------------------------------
# Video canvas  (QLabel that draws overlay directly from OpenCV frame)
# ------------------------------------------------------------------------------

class VideoCanvas(QLabel):
    athlete_clicked = pyqtSignal(int)   # emits detected-box index
    cal_point_clicked = pyqtSignal(int, int)  # emits frame x,y

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('background: #000;')
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._frame_w = 1
        self._frame_h = 1
        self._disp_w = 1
        self._disp_h = 1
        self._off_x = 0
        self._off_y = 0
        self._boxes = []
        self._select_mode = True
        self._cal_mode = False
        self._last_overlay = None   # stored so resizeEvent can re-scale

    # -- coordinate mapping --------------------------------------------------

    def _to_frame(self, wx, wy):
        fx = (wx - self._off_x) * self._frame_w / self._disp_w
        fy = (wy - self._off_y) * self._frame_h / self._disp_h
        return int(fx), int(fy)

    # -- display -------------------------------------------------------------

    def show_frame(self, bgr):
        if bgr is None:
            return
        self._last_overlay = bgr
        self._paint(bgr)

    def _paint(self, bgr):
        lw, lh = self.width(), self.height()
        if lw < 2 or lh < 2:
            return
        h, w = bgr.shape[:2]
        self._frame_w, self._frame_h = w, h
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pm = QPixmap.fromImage(qi)
        scaled = pm.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._disp_w = scaled.width()
        self._disp_h = scaled.height()
        self._off_x = (lw - self._disp_w) // 2
        self._off_y = (lh - self._disp_h) // 2
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_overlay is not None:
            self._paint(self._last_overlay)

    # -- mouse ---------------------------------------------------------------

    def mousePressEvent(self, event):
        fx, fy = self._to_frame(event.x(), event.y())
        if self._cal_mode:
            self.cal_point_clicked.emit(fx, fy)
            return
        if not self._select_mode:
            return
        for i, item in enumerate(self._boxes):
            x1, y1, x2, y2 = item['box']
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                self.athlete_clicked.emit(i)
                return


# ------------------------------------------------------------------------------
# Main window
# ------------------------------------------------------------------------------

class JLVisionV10(QMainWindow):

    def __init__(self, video_path):
        super().__init__()
        self.video_path = str(video_path)
        self.setWindowTitle('JL Vision  .  Coach Workspace  v10')
        self.resize(1440, 900)
        self.setStyleSheet(STYLE)
        self.setWindowState(Qt.WindowMaximized)

        # models
        self.yolo = YOLO('yolov8n.pt')
        self.movenet = MoveNet('lightning')

        # video state
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open: {self.video_path}')
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_frame = 0
        self.playing = False
        self.speed = 1.0
        self._current_bgr = None

        # athlete state
        self.detected_boxes = []
        self.selected_box = None
        self.keypoints = None
        self.inference_on = True

        # speed tracking
        self.center_history = deque(maxlen=20)
        self.pixels_per_meter = None

        # stride detection
        self.stride_detector = StrideDetector(self.fps)
        self._cal_prompted = False   # auto-prompt calibration once on first lock

        # calibration
        self._cal_pts = []

        # session log (list of dicts, one per analyzed frame)
        self._session_log = []

        # body map
        self.muscle_act = MuscleActivation()
        self._last_angles = {}

        self._build_ui()
        self._connect()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

        # Delay initial render until after the window is shown and laid out,
        # so the canvas reports its true pixel size when scaling the first frame.
        QTimer.singleShot(150, self._initial_load)

    def _initial_load(self):
        self._seek(0)
        self._run_detection()

    # --------------------------------------------------------------------------
    # UI construction
    # --------------------------------------------------------------------------

    def _build_ui(self):
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root = QHBoxLayout(root_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._left_toolbar())

        center = QVBoxLayout()
        center.setContentsMargins(4, 4, 4, 4)
        center.setSpacing(4)
        center.addWidget(self._header())

        self.canvas = VideoCanvas()
        self.canvas.athlete_clicked.connect(self._athlete_selected)
        self.canvas.cal_point_clicked.connect(self._cal_click)
        center.addWidget(self.canvas, 1)

        center.addWidget(self._controls())

        center_w = QWidget()
        center_w.setLayout(center)
        root.addWidget(center_w, 1)

        root.addWidget(self._metrics_panel())

    # -- header ---------------------------------------------------------------

    def _header(self):
        w = QWidget()
        w.setFixedHeight(58)
        w.setStyleSheet('background: #080a10; border-bottom: 2px solid #00ffd020;')
        row = QHBoxLayout(w)
        row.setContentsMargins(20, 0, 20, 0)

        title = QLabel('JL VISION   Coach Workspace  v10')
        title.setStyleSheet('color: #00ffd0; font-size: 18px; font-weight: bold; letter-spacing: 2px;')
        row.addWidget(title)
        row.addStretch()

        self.hdr_frame = QLabel('Frame: -')
        self.hdr_speed = QLabel('Speed: 1x')
        self.hdr_pose  = QLabel('Pose: -')
        for lbl in [self.hdr_frame, self.hdr_speed, self.hdr_pose]:
            lbl.setStyleSheet('color: #889; font-size: 14px;')
            row.addWidget(lbl)
            sep = QLabel('  |  ')
            sep.setStyleSheet('color: #334; font-size: 14px;')
            row.addWidget(sep)

        return w

    # -- left annotation toolbar ----------------------------------------------

    def _left_toolbar(self):
        w = QWidget()
        w.setFixedWidth(160)
        w.setStyleSheet('background: #080a10; border-right: 2px solid #00ffd030;')
        col = QVBoxLayout(w)
        col.setContentsMargins(10, 18, 10, 18)
        col.setSpacing(8)
        col.setAlignment(Qt.AlignTop)

        # Section label
        lbl = QLabel('TOOLS')
        lbl.setStyleSheet('color: #556; font-size: 12px; font-weight: bold; letter-spacing: 3px;')
        lbl.setAlignment(Qt.AlignCenter)
        col.addWidget(lbl)
        col.addSpacing(6)

        self._tool_btns = {}
        for name in ['Select', 'Line', 'Angle', 'Circle', 'Dot', 'Arrow', 'Text', 'Draw', 'Erase']:
            btn = QPushButton(name)
            btn.setFixedSize(140, 44)
            btn.setCheckable(True)
            btn.setStyleSheet(
                'QPushButton { font-size: 14px; font-weight: bold; text-align: left; padding-left: 14px; }'
                'QPushButton:checked { background: #00ffd0; color: #0d0f14; }'
            )
            col.addWidget(btn)
            self._tool_btns[name] = btn

        self._tool_btns['Select'].setChecked(True)

        col.addStretch()

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet('color: #1a2035;')
        col.addWidget(line)
        col.addSpacing(6)

        self.btn_cal = QPushButton('Calibrate')
        self.btn_cal.setFixedSize(140, 44)
        self.btn_cal.setStyleSheet(
            'QPushButton { font-size: 14px; font-weight: bold; '
            'background: #0d1a16; border: 1px solid #00ffd0; border-radius: 5px; color: #00ffd0; }'
            'QPushButton:hover { background: #00ffd0; color: #0d0f14; }'
        )
        self.btn_cal.setToolTip('Click 2 points with known real-world distance')
        col.addWidget(self.btn_cal)

        return w

    # -- playback controls ----------------------------------------------------

    def _controls(self):
        w = QWidget()
        w.setFixedHeight(120)
        w.setStyleSheet('background: #080a10; border-top: 2px solid #00ffd020;')
        col = QVBoxLayout(w)
        col.setContentsMargins(16, 8, 16, 8)
        col.setSpacing(8)

        # scrubber row
        scr_row = QHBoxLayout()
        self._lbl_pos = QLabel('0')
        self._lbl_pos.setStyleSheet('color: #667; font-size: 13px; min-width: 40px;')
        self._lbl_pos.setAlignment(Qt.AlignRight)
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, max(self.total_frames - 1, 1))
        self._lbl_total = QLabel(str(self.total_frames))
        self._lbl_total.setStyleSheet('color: #667; font-size: 13px; min-width: 48px;')
        scr_row.addWidget(self._lbl_pos)
        scr_row.addWidget(self.scrubber, 1)
        scr_row.addWidget(self._lbl_total)
        col.addLayout(scr_row)

        # --- Row 1: transport ---
        transport = QHBoxLayout()
        transport.setSpacing(8)

        self.btn_p10  = QPushButton('<< -10')
        self.btn_p1   = QPushButton('< -1')
        self.btn_play = QPushButton('> Play')
        self.btn_play.setCheckable(True)
        self.btn_play.setMinimumWidth(100)
        self.btn_n1   = QPushButton('+1 >')
        self.btn_n10  = QPushButton('+10 >>')
        for b in [self.btn_p10, self.btn_p1, self.btn_play, self.btn_n1, self.btn_n10]:
            b.setFixedHeight(38)

        transport.addWidget(self.btn_p10)
        transport.addWidget(self.btn_p1)
        transport.addWidget(self.btn_play)
        transport.addWidget(self.btn_n1)
        transport.addWidget(self.btn_n10)
        transport.addStretch()

        spd_lbl = QLabel('Speed:')
        spd_lbl.setStyleSheet('color: #778; font-size: 13px;')
        transport.addWidget(spd_lbl)
        self._spd_btns = {}
        for label, val in [('1x', 1.0), ('1/2x', 0.5), ('1/4x', 0.25), ('1/8x', 0.125)]:
            b = QPushButton(label)
            b.setFixedSize(52, 38)
            b.setCheckable(True)
            b.setChecked(val == 1.0)
            transport.addWidget(b)
            self._spd_btns[val] = b

        col.addLayout(transport)

        # --- Row 2: tools ---
        tools = QHBoxLayout()
        tools.setSpacing(8)

        self.btn_inference = QPushButton('Pose ON')
        self.btn_inference.setCheckable(True)
        self.btn_inference.setChecked(True)
        self.btn_inference.setFixedHeight(34)

        self.btn_detect   = QPushButton('Detect Athletes')
        self.btn_detect.setFixedHeight(34)

        self.btn_reselect = QPushButton('Reselect')
        self.btn_reselect.setFixedHeight(34)

        tools.addWidget(self.btn_inference)
        tools.addWidget(self.btn_detect)
        tools.addWidget(self.btn_reselect)
        tools.addStretch()

        hint = QLabel('Space play/pause  |  < > step  |  1 2 4 8 speed  |  wheel = frame  |  Esc quit')
        hint.setStyleSheet('color: #445; font-size: 11px;')
        tools.addWidget(hint)

        col.addLayout(tools)

        return w

    # -- metrics panel --------------------------------------------------------

    def _metrics_panel(self):
        w = QWidget()
        w.setFixedWidth(380)
        w.setStyleSheet('background: #0a0c12; border-left: 2px solid #00ffd030;')
        col = QVBoxLayout(w)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        # ── Tab bar ──────────────────────────────────────────────────────────
        tab_bar = QWidget()
        tab_bar.setFixedHeight(44)
        tab_bar.setStyleSheet('background: #080a10; border-bottom: 1px solid #00ffd020;')
        tab_row = QHBoxLayout(tab_bar)
        tab_row.setContentsMargins(0, 0, 0, 0)
        tab_row.setSpacing(0)

        tab_style = (
            'QPushButton { background: transparent; color: #445; font-size: 12px; '
            'font-weight: bold; border: none; letter-spacing: 1px; '
            'border-bottom: 2px solid transparent; }'
            'QPushButton:checked { color: #00ffd0; border-bottom: 2px solid #00ffd0; }'
            'QPushButton:hover { color: #8af; }'
        )
        self._tab_joint = QPushButton('JOINT METRICS')
        self._tab_body  = QPushButton('BODY MAP')
        for btn in (self._tab_joint, self._tab_body):
            btn.setCheckable(True)
            btn.setFixedHeight(44)
            btn.setStyleSheet(tab_style)
            tab_row.addWidget(btn)
        self._tab_joint.setChecked(True)
        col.addWidget(tab_bar)

        # ── Stacked pages ────────────────────────────────────────────────────
        self._panel_stack = QStackedWidget()

        # Page 0: Joint Metrics
        page_metrics = QWidget()
        pm = QVBoxLayout(page_metrics)
        pm.setContentsMargins(14, 14, 14, 14)
        pm.setSpacing(10)
        self._build_metrics_page(pm)

        # Page 1: Body Map
        page_body = QWidget()
        pb = QVBoxLayout(page_body)
        pb.setContentsMargins(6, 6, 6, 6)
        pb.setSpacing(0)
        self.body_map = BodyMapWidget()
        pb.addWidget(self.body_map)

        self._panel_stack.addWidget(page_metrics)
        self._panel_stack.addWidget(page_body)
        col.addWidget(self._panel_stack, 1)

        self._tab_joint.clicked.connect(lambda: self._switch_tab(0))
        self._tab_body.clicked.connect(lambda: self._switch_tab(1))

        return w

    def _switch_tab(self, idx):
        self._panel_stack.setCurrentIndex(idx)
        self._tab_joint.setChecked(idx == 0)
        self._tab_body.setChecked(idx == 1)

    def _build_metrics_page(self, col):
        def grp(title, rows):
            g = QGroupBox(title)
            v = QVBoxLayout(g)
            v.setSpacing(0)
            metrics = {}
            for lbl in rows:
                m = MetricRow(lbl)
                v.addWidget(m)
                metrics[lbl] = m
            col.addWidget(g)
            return metrics

        self.m_athlete = grp('Target Athlete', [
            'Status', 'Frame', 'Box W x H', 'Center',
        ])

        self.m_pose = grp('Pose Layer — MoveNet', [
            'Pose', 'Hip R', 'Knee R', 'Hip L', 'Knee L', 'Torso lean',
        ])

        self.m_sprint = grp('Sprint Metrics', [
            'Speed', 'Calibration',
        ])
        self.m_sprint['Speed'].pending()
        self.m_sprint['Calibration'].set('not set', '#f5a623')

        self.m_pending = grp('Stride / GCT', [
            'Stride length', 'Ground contact', 'Step frequency',
        ])
        for m in self.m_pending.values():
            m.pending()

        # Coach notes
        ng = QGroupBox('Coach Notes')
        nv = QVBoxLayout(ng)
        self.notes = QTextEdit()
        self.notes.setPlaceholderText('Type coaching observations...')
        self.notes.setFixedHeight(68)
        nv.addWidget(self.notes)
        nr = QHBoxLayout()
        self.btn_export = QPushButton('Export CSV')
        self.btn_export.setFixedHeight(26)
        nr.addWidget(self.btn_export)
        nv.addLayout(nr)
        col.addWidget(ng)

        col.addStretch()

        # JL Pulse placeholder
        pp = QGroupBox('JL Pulse — Not Connected')
        pp.setStyleSheet('QGroupBox { color: #f5a623; }')
        pv = QVBoxLayout(pp)
        for lbl in ['Fatigue signal', 'Readiness', 'HRV']:
            m = MetricRow(lbl)
            m.pending()
            pv.addWidget(m)
        col.addWidget(pp)

    # --------------------------------------------------------------------------
    # Signal wiring
    # --------------------------------------------------------------------------

    def _connect(self):
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_p10.clicked.connect(lambda: self._step(-10))
        self.btn_p1.clicked.connect(lambda: self._step(-1))
        self.btn_n1.clicked.connect(lambda: self._step(1))
        self.btn_n10.clicked.connect(lambda: self._step(10))
        self.scrubber.sliderMoved.connect(self._scrub)
        self.scrubber.sliderPressed.connect(self._pause)

        for val, btn in self._spd_btns.items():
            btn.clicked.connect(lambda _, v=val: self._set_speed(v))

        self.btn_detect.clicked.connect(self._run_detection)
        self.btn_reselect.clicked.connect(self._reselect)
        self.btn_inference.clicked.connect(self._toggle_inference)
        self.btn_cal.clicked.connect(self._start_cal)
        self.btn_export.clicked.connect(self._export_csv)

        for name, btn in self._tool_btns.items():
            btn.clicked.connect(lambda _, n=name: self._pick_tool(n))

    # --------------------------------------------------------------------------
    # Playback
    # --------------------------------------------------------------------------

    def _toggle_play(self, on):
        if on:
            self._play()
        else:
            self._pause()

    def _play(self):
        self.playing = True
        self.btn_play.setChecked(True)
        self.btn_play.setText('|| Pause')
        self.timer.start(max(int(1000 / (self.fps * self.speed)), 16))

    def _pause(self):
        self.playing = False
        self.btn_play.setChecked(False)
        self.btn_play.setText('> Play')
        self.timer.stop()

    def _tick(self):
        if self.current_frame >= self.total_frames - 1:
            self._pause()
            return
        self._advance(1)

    def _step(self, delta):
        self._pause()
        self._advance(delta)

    def _advance(self, delta):
        target = max(0, min(self.current_frame + delta, self.total_frames - 1))
        if target != self.current_frame:
            self._seek(target)

    def _seek(self, idx):
        self.current_frame = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self._current_bgr = frame
        self._render(frame)
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(idx)
        self.scrubber.blockSignals(False)
        self._lbl_pos.setText(str(idx))
        self.hdr_frame.setText(f'Frame: {idx + 1} / {self.total_frames}')

    def _scrub(self, val):
        self._pause()
        self._seek(val)

    def _set_speed(self, v):
        self.speed = v
        self.hdr_speed.setText(f'Speed: {v}x')
        for sv, btn in self._spd_btns.items():
            btn.setChecked(sv == v)
        if self.playing:
            self.timer.setInterval(max(int(1000 / (self.fps * v)), 16))

    def _toggle_inference(self, on):
        self.inference_on = on
        self.btn_inference.setText('Pose ON' if on else 'Pose OFF')

    # --------------------------------------------------------------------------
    # Athlete detection & selection
    # --------------------------------------------------------------------------

    def _run_detection(self):
        if self._current_bgr is None:
            return
        self.detected_boxes = detect_people(self.yolo, self._current_bgr)
        self.selected_box = None
        self.keypoints = None
        self.canvas._boxes = self.detected_boxes
        self.canvas._select_mode = True
        self.canvas._cal_mode = False
        self._render(self._current_bgr)
        self.m_athlete['Status'].set('Select an athlete', '#f5a623')

    def _athlete_selected(self, idx):
        if idx >= len(self.detected_boxes):
            return
        self.selected_box = self.detected_boxes[idx]['box']
        self.canvas._select_mode = False
        self.center_history.clear()
        self.stride_detector.reset()
        self._render(self._current_bgr)
        self.m_athlete['Status'].set('Target Locked', '#00ffd0')
        # Auto-prompt calibration the first time an athlete is locked
        if not self._cal_prompted and self.pixels_per_meter is None:
            self._cal_prompted = True
            QMessageBox.information(
                self, 'Calibration needed',
                'Athlete locked.\n\n'
                'To enable Speed and Stride Length:\n'
                '  1. Click  Cal  in the left toolbar\n'
                '  2. Click two points with a known distance\n'
                '     (e.g. lane width = 1.22 m, 10 m cone gap)\n'
                '  3. Enter the real-world distance in metres\n\n'
                'You can skip this and calibrate later.'
            )

    def _reselect(self):
        self._pause()
        self._run_detection()

    # --------------------------------------------------------------------------
    # Render (overlay + display)
    # --------------------------------------------------------------------------

    def _render(self, bgr):
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        overlay = bgr.copy()

        if self.selected_box is None:
            # -- selection mode ----------------------------------------------
            for i, item in enumerate(self.detected_boxes):
                x1, y1, x2, y2 = item['box']
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 200), 2)
                # Short label "A1" centered above box — avoids overlap in crowds
                tag = f'A{i + 1}'
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cx_box = (x1 + x2) // 2
                tx = max(4, min(cx_box - tw // 2, overlay.shape[1] - tw - 4))
                ty = max(th + 6, y1 - 6)
                cv2.rectangle(overlay, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4),
                              (0, 20, 16), -1)
                cv2.putText(overlay, tag, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2, cv2.LINE_AA)

            self._glass(overlay, 30, 22, w - 60, 72)
            cv2.putText(overlay, 'JL VISION  |  SELECT TARGET ATHLETE', (52, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(overlay, 'Click on the athlete you want to analyze', (52, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        else:
            # -- review mode -------------------------------------------------

            # Re-track athlete via IoU (every 10 frames to keep playback smooth)
            if self.current_frame % 10 == 0:
                current = detect_people(self.yolo, bgr, conf=0.30)
                if current:
                    best = max(current, key=lambda b: iou(b['box'], self.selected_box))
                    if iou(best['box'], self.selected_box) > 0.25:
                        self.selected_box = best['box']

            x1, y1, x2, y2 = self.selected_box
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Track center for speed
            self.center_history.append((cx, cy, self.current_frame))

            # Draw box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 200), 3)
            cv2.line(overlay, (cx, y1), (cx, y2), (0, 200, 160), 1)

            # Badge above athlete
            badge_y = max(y1 - 44, 120)
            self._glass(overlay, x1, badge_y, 210, 36, 0.75)
            cv2.putText(overlay, 'TARGET ATHLETE', (x1 + 10, badge_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

            # Pose inference
            if self.inference_on:
                kps = self.movenet.estimate(bgr, box=self.selected_box)
                self.keypoints = kps
                if kps is not None:
                    self._draw_skeleton(overlay, kps, w, h)
                    angles = self._draw_callouts(overlay, kps, w, h)
                    self._update_pose_metrics(kps, angles, w, h)
                    self.stride_detector.update(
                        self.current_frame, kps, w, h, self.pixels_per_meter)
                    self._update_stride_metrics()
                    self.hdr_pose.setText('Pose: Active')
                else:
                    self.hdr_pose.setText('Pose: Not found')
                    self.m_pose['Pose'].set('Not found', '#f5a623')
            else:
                self.hdr_pose.setText('Pose: OFF')

            # Metrics panel
            self.m_athlete['Frame'].set(f'{self.current_frame + 1} / {self.total_frames}')
            self.m_athlete['Box W x H'].set(f'{bw} x {bh} px')
            self.m_athlete['Center'].set(f'{cx}, {cy}')

            self._update_speed()
            self._log_frame(cx, cy)

        self.canvas.show_frame(overlay)

    def _glass(self, frame, x, y, w, h, alpha=0.55):
        ov = frame.copy()
        cv2.rectangle(ov, (x, y), (x + w, y + h), (12, 15, 22), -1)
        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 255, 200), 1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    def _draw_skeleton(self, frame, kps, w, h):
        for a_name, b_name in SKELETON_EDGES:
            pa = kp_xy(kps, a_name, w, h)
            pb = kp_xy(kps, b_name, w, h)
            if pa and pb:
                cv2.line(frame, pa, pb, (0, 200, 255), 3, cv2.LINE_AA)
        # Draw all 17 keypoints regardless of edge connections
        for name in KEYPOINT_NAMES:
            pt = kp_xy(kps, name, w, h)
            if pt:
                color = (0, 255, 200) if name in (
                    'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                ) else (180, 220, 255)
                cv2.circle(frame, pt, 7, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 10, (0, 200, 255), 1, cv2.LINE_AA)

    def _draw_callouts(self, frame, kps, w, h):
        """Draw angle callout tags in two fixed columns flanking the bounding box.
        Right side tags go right of box x2; left side tags go left of box x1.
        Rows are spaced vertically so they never overlap each other."""
        angles = {}
        TAG_W, TAG_H = 190, 38

        if self.selected_box:
            bx1, by1, bx2, by2 = self.selected_box
            bh = by2 - by1
        else:
            bx1, by1, bx2, by2 = w // 2, 0, w // 2, h
            bh = h

        # Fixed column positions: right of box and left of box
        r_col = min(bx2 + 18, w - TAG_W - 4)
        l_col = max(bx1 - TAG_W - 18, 4)

        # Vertical slots: hip at ~35% down box, knee at ~62% down box
        slots = {
            'HIP-R':  (r_col, max(by1 + int(bh * 0.32), 42)),
            'KNEE-R': (r_col, max(by1 + int(bh * 0.62), 42 + TAG_H + 8)),
            'HIP-L':  (l_col, max(by1 + int(bh * 0.32), 42)),
            'KNEE-L': (l_col, max(by1 + int(bh * 0.62), 42 + TAG_H + 8)),
        }
        # Clamp rows to frame height
        for k in slots:
            tx, ty = slots[k]
            slots[k] = (tx, min(ty, h - TAG_H - 4))

        specs = [
            ('HIP-R',  'right_shoulder', 'right_hip',  'right_knee'),
            ('KNEE-R', 'right_hip',      'right_knee', 'right_ankle'),
            ('HIP-L',  'left_shoulder',  'left_hip',   'left_knee'),
            ('KNEE-L', 'left_hip',       'left_knee',  'left_ankle'),
        ]
        for label, a_kp, b_kp, c_kp in specs:
            pa = kp_xy(kps, a_kp, w, h)
            pb = kp_xy(kps, b_kp, w, h)
            pc = kp_xy(kps, c_kp, w, h)
            if not (pa and pb and pc):
                continue
            ang = angle_at(pa, pb, pc)
            angles[label] = ang
            tx, ty = slots[label]
            # leader line from joint to tag center-left
            anchor = (tx + TAG_W // 2, ty - TAG_H // 2)
            cv2.line(frame, pb, anchor, (0, 255, 200), 2, cv2.LINE_AA)
            # filled background
            cv2.rectangle(frame, (tx, ty - TAG_H), (tx + TAG_W, ty),
                          (10, 13, 20), -1)
            cv2.rectangle(frame, (tx, ty - TAG_H), (tx + TAG_W, ty),
                          (0, 255, 200), 2)
            cv2.putText(frame, f'{label}  {ang:.1f}deg',
                        (tx + 8, ty - 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 200), 2, cv2.LINE_AA)
        return angles

    def _update_pose_metrics(self, kps, angles, w, h):
        self.m_pose['Pose'].set('Active', '#00ffd0')
        self.m_pose['Hip R'].set(f"{angles['HIP-R']:.1f} deg"
                                 if 'HIP-R' in angles else '--', '#00ffd0')
        self.m_pose['Knee R'].set(f"{angles['KNEE-R']:.1f} deg"
                                  if 'KNEE-R' in angles else '--', '#00ffd0')
        self.m_pose['Hip L'].set(f"{angles['HIP-L']:.1f} deg"
                                 if 'HIP-L' in angles else '--', '#00ffd0')
        self.m_pose['Knee L'].set(f"{angles['KNEE-L']:.1f} deg"
                                  if 'KNEE-L' in angles else '--', '#00ffd0')

        # Torso lean
        r_sh = kp_xy(kps, 'right_shoulder', w, h)
        l_sh = kp_xy(kps, 'left_shoulder', w, h)
        r_hp = kp_xy(kps, 'right_hip', w, h)
        l_hp = kp_xy(kps, 'left_hip', w, h)
        lean = 0.0
        if all([r_sh, l_sh, r_hp, l_hp]):
            sh = ((r_sh[0] + l_sh[0]) // 2, (r_sh[1] + l_sh[1]) // 2)
            hp = ((r_hp[0] + l_hp[0]) // 2, (r_hp[1] + l_hp[1]) // 2)
            dx, dy = sh[0] - hp[0], sh[1] - hp[1]
            lean = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))
            direction = 'fwd' if dx > 0 else 'back'
            self.m_pose['Torso lean'].set(f'{lean:.1f} deg {direction}', '#00ffd0')

        # Push to body map (runs whether or not body map tab is active)
        angles['TORSO_LEAN'] = lean
        activations = self.muscle_act.compute(kps, angles, w, h)
        self.body_map.update_activations(activations)

        # Store angles for CSV export
        self._last_angles = angles

    def _update_speed(self):
        if self.pixels_per_meter is None:
            return
        hist = list(self.center_history)
        if len(hist) < 4:
            return
        window = hist[-8:] if len(hist) >= 8 else hist
        x0, y0, f0 = window[0]
        x1, y1, f1 = window[-1]
        df = f1 - f0
        if df <= 0:
            return
        dist_m = math.hypot(x1 - x0, y1 - y0) / self.pixels_per_meter
        time_s = df / self.fps
        speed_ms  = dist_m / time_s
        speed_mph = speed_ms * 2.237
        self.m_sprint['Speed'].set(f'{speed_ms:.1f} m/s  {speed_mph:.1f} mph', '#00ffd0')

    def _update_stride_metrics(self):
        sd = self.stride_detector
        if sd.stride_length_m is not None:
            self.m_pending['Stride length'].set(
                f'{sd.stride_length_m:.2f} m', '#00ffd0')
        if sd.step_freq_hz is not None:
            self.m_pending['Step frequency'].set(
                f'{sd.step_freq_hz:.1f} steps/s', '#00ffd0')
            # GCT estimate: sprinting duty factor ~0.30
            stride_time = 1.0 / max(sd.step_freq_hz / 2.0, 0.1)
            gct_ms = stride_time * 0.30 * 1000
            self.m_pending['Ground contact'].set(f'{gct_ms:.0f} ms', '#00ffd0')

    def _log_frame(self, cx, cy):
        a = self._last_angles
        entry = {
            'frame':    self.current_frame,
            'cx': cx,   'cy': cy,
            'hip_r':    f"{a['HIP-R']:.1f}"  if 'HIP-R'  in a else '',
            'knee_r':   f"{a['KNEE-R']:.1f}" if 'KNEE-R' in a else '',
            'hip_l':    f"{a['HIP-L']:.1f}"  if 'HIP-L'  in a else '',
            'knee_l':   f"{a['KNEE-L']:.1f}" if 'KNEE-L' in a else '',
            'torso_lean': f"{a['TORSO_LEAN']:.1f}" if 'TORSO_LEAN' in a else '',
            'speed_ms': '',
        }
        self._session_log.append(entry)

    # --------------------------------------------------------------------------
    # Calibration
    # --------------------------------------------------------------------------

    def _start_cal(self):
        self._pause()
        QMessageBox.information(
            self, 'Calibrate',
            'Click two points on the video that represent a known distance.\n'
            'Example: two lane lines (1.22 m), two cones (10 m).'
        )
        self._cal_pts = []
        self.canvas._cal_mode = True

    def _cal_click(self, fx, fy):
        self._cal_pts.append((fx, fy))
        if len(self._cal_pts) == 2:
            self.canvas._cal_mode = False
            p1, p2 = self._cal_pts
            px_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            meters, ok = QInputDialog.getDouble(
                self, 'Calibration',
                'Distance between the two points (meters):',
                value=10.0, min=0.01, max=1000.0, decimals=2,
            )
            if ok and meters > 0:
                self.pixels_per_meter = px_dist / meters
                self.m_sprint['Calibration'].set(
                    f'{self.pixels_per_meter:.1f} px/m', '#00ffd0')
                self.m_sprint['Speed'].set('Tracking...', '#aaa')

    # --------------------------------------------------------------------------
    # CSV export
    # --------------------------------------------------------------------------

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Session CSV', 'jl_vision_session.csv', 'CSV (*.csv)')
        if not path:
            return
        fields = ['frame', 'cx', 'cy', 'hip_r', 'knee_r', 'hip_l', 'knee_l', 'torso_lean', 'speed_ms']
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self._session_log)
        QMessageBox.information(self, 'Exported', f'Saved to:\n{path}')

    # --------------------------------------------------------------------------
    # Tool selection
    # --------------------------------------------------------------------------

    def _pick_tool(self, name):
        for n, btn in self._tool_btns.items():
            btn.setChecked(n == name)

    # --------------------------------------------------------------------------
    # Keyboard + wheel
    # --------------------------------------------------------------------------

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Space:
            self._toggle_play(not self.playing)
        elif k in (Qt.Key_Right, Qt.Key_Period):
            self._step(1)
        elif k in (Qt.Key_Left, Qt.Key_Comma):
            self._step(-1)
        elif k == Qt.Key_1:
            self._set_speed(1.0)
        elif k == Qt.Key_2:
            self._set_speed(0.5)
        elif k == Qt.Key_4:
            self._set_speed(0.25)
        elif k == Qt.Key_8:
            self._set_speed(0.125)
        elif k in (Qt.Key_Q, Qt.Key_Escape):
            self.close()

    def wheelEvent(self, event):
        self._step(1 if event.angleDelta().y() > 0 else -1)

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python coach_workspace_v10.py <video_path>')
        sys.exit(1)
    video = Path(sys.argv[1])
    if not video.exists():
        print(f'Video not found: {video}')
        sys.exit(1)

    # High-DPI support — must be set before QApplication is created
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    # Boost base font so everything scales up proportionally
    base_font = app.font()
    base_font.setPointSize(11)
    base_font.setFamily('Segoe UI')
    app.setFont(base_font)
    win = JLVisionV10(video)
    win.show()
    sys.exit(app.exec_())
