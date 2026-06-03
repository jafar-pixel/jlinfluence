#!/usr/bin/env python3
"""
JL Vision Coach Workspace v10
PyQt5 video player + TensorFlow MoveNet pose + YOLO athlete detection

Environment : jl-tf-env  (NOT jl-vision-env)
Install     : pip install PyQt5 tensorflow tensorflow-hub ultralytics opencv-python numpy
Usage       : python coach_workspace_v10.py videos/test_video.mp4
"""

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
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont


# ──────────────────────────────────────────────────────────────────────────────
# MoveNet
# ──────────────────────────────────────────────────────────────────────────────

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


class MoveNet:
    def __init__(self, variant='lightning'):
        print(f'Loading MoveNet {variant} …')
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


# ──────────────────────────────────────────────────────────────────────────────
# YOLO helpers
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────────

def angle_at(a, vertex, c):
    """Angle in degrees at 'vertex' between rays vertex→a and vertex→c."""
    ax, ay = a[0] - vertex[0], a[1] - vertex[1]
    cx, cy = c[0] - vertex[0], c[1] - vertex[1]
    dot = ax * cx + ay * cy
    mag = math.hypot(ax, ay) * math.hypot(cx, cy) + 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def kp_xy(kps, name, w, h, thresh=0.3):
    """Return (x, y) pixel or None."""
    y_n, x_n, conf = kps[KP[name]]
    if conf < thresh:
        return None
    return (int(x_n * w), int(y_n * h))


# ──────────────────────────────────────────────────────────────────────────────
# Qt stylesheet
# ──────────────────────────────────────────────────────────────────────────────

STYLE = """
* { font-family: "Segoe UI", Arial, sans-serif; }
QMainWindow, QWidget { background: #0d0f14; color: #dde; }
QLabel { color: #ccd; }
QPushButton {
    background: #131820; color: #00ffd0;
    border: 1px solid #00ffd0; border-radius: 4px;
    padding: 4px 10px; font-size: 12px;
}
QPushButton:hover  { background: #00ffd0; color: #0d0f14; }
QPushButton:pressed { background: #00b89a; }
QPushButton:checked { background: #00ffd0; color: #0d0f14; }
QPushButton:disabled { border-color: #333; color: #444; }
QSlider::groove:horizontal { background: #1a1f2e; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #00ffd0; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #00ffd0; border-radius: 2px; }
QGroupBox {
    border: 1px solid #222840; border-radius: 6px;
    margin-top: 8px; padding-top: 8px;
    font-size: 11px; color: #00ffd0; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QTextEdit {
    background: #0a0c12; color: #aab; border: 1px solid #222840;
    border-radius: 4px; padding: 4px; font-size: 12px;
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Metric label
# ──────────────────────────────────────────────────────────────────────────────

class MetricRow(QWidget):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 1, 4, 1)
        self._lbl = QLabel(label)
        self._lbl.setStyleSheet('color: #6a7a8a; font-size: 11px;')
        self._lbl.setFixedWidth(110)
        self._val = QLabel('—')
        self._val.setStyleSheet('color: #dde; font-size: 12px; font-weight: bold;')
        self._val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self._lbl)
        row.addWidget(self._val, 1)

    def set(self, text, color='#dde'):
        self._val.setText(str(text))
        self._val.setStyleSheet(f'color: {color}; font-size: 12px; font-weight: bold;')

    def pending(self):
        self.set('pending', '#f5a623')


# ──────────────────────────────────────────────────────────────────────────────
# Video canvas  (QLabel that draws overlay directly from OpenCV frame)
# ──────────────────────────────────────────────────────────────────────────────

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

    # ── coordinate mapping ──────────────────────────────────────────────────

    def _to_frame(self, wx, wy):
        fx = (wx - self._off_x) * self._frame_w / self._disp_w
        fy = (wy - self._off_y) * self._frame_h / self._disp_h
        return int(fx), int(fy)

    # ── display ─────────────────────────────────────────────────────────────

    def show_frame(self, bgr):
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        self._frame_w, self._frame_h = w, h
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pm = QPixmap.fromImage(qi)
        lw, lh = self.width(), self.height()
        scaled = pm.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._disp_w = scaled.width()
        self._disp_h = scaled.height()
        self._off_x = (lw - self._disp_w) // 2
        self._off_y = (lh - self._disp_h) // 2
        self.setPixmap(scaled)

    # ── mouse ───────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────

class JLVisionV10(QMainWindow):

    def __init__(self, video_path):
        super().__init__()
        self.video_path = str(video_path)
        self.setWindowTitle('JL Vision  ·  Coach Workspace  v10')
        self.resize(1440, 900)
        self.setStyleSheet(STYLE)

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

        # calibration
        self._cal_pts = []

        # session log (list of dicts, one per analyzed frame)
        self._session_log = []

        self._build_ui()
        self._connect()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

        self._seek(0)
        self._run_detection()

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────

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

    # ── header ───────────────────────────────────────────────────────────────

    def _header(self):
        w = QWidget()
        w.setFixedHeight(46)
        w.setStyleSheet('background: #080a10; border-bottom: 1px solid #181e2e;')
        row = QHBoxLayout(w)
        row.setContentsMargins(16, 0, 16, 0)

        title = QLabel('JL VISION   Coach Workspace  v10')
        title.setStyleSheet('color: #00ffd0; font-size: 14px; font-weight: bold; letter-spacing: 1px;')
        row.addWidget(title)
        row.addStretch()

        self.hdr_frame = QLabel('Frame: —')
        self.hdr_speed = QLabel('Speed: 1x')
        self.hdr_pose  = QLabel('Pose: —')
        for lbl in [self.hdr_frame, self.hdr_speed, self.hdr_pose]:
            lbl.setStyleSheet('color: #778; font-size: 12px;')
            row.addWidget(lbl)
            sep = QLabel('  |  ')
            sep.setStyleSheet('color: #333;')
            row.addWidget(sep)

        return w

    # ── left annotation toolbar ──────────────────────────────────────────────

    def _left_toolbar(self):
        w = QWidget()
        w.setFixedWidth(52)
        w.setStyleSheet('background: #080a10; border-right: 1px solid #181e2e;')
        col = QVBoxLayout(w)
        col.setContentsMargins(6, 12, 6, 12)
        col.setSpacing(6)
        col.setAlignment(Qt.AlignTop)

        self._tool_btns = {}
        for icon, name in [('↖','Select'),('╱','Line'),('∠','Angle'),
                            ('○','Circle'),('•','Dot'),('→','Arrow'),
                            ('T','Text'),('✎','Draw'),('⌫','Erase')]:
            btn = QPushButton(icon)
            btn.setFixedSize(38, 38)
            btn.setCheckable(True)
            btn.setToolTip(name)
            btn.setStyleSheet('QPushButton{font-size:15px;padding:0}'
                              'QPushButton:checked{background:#00ffd0;color:#0d0f14}')
            col.addWidget(btn)
            self._tool_btns[name] = btn

        self._tool_btns['Select'].setChecked(True)

        col.addStretch()

        self.btn_cal = QPushButton('⚖')
        self.btn_cal.setFixedSize(38, 38)
        self.btn_cal.setToolTip('Calibrate distance (2 clicks)')
        col.addWidget(self.btn_cal)

        return w

    # ── playback controls ────────────────────────────────────────────────────

    def _controls(self):
        w = QWidget()
        w.setFixedHeight(104)
        w.setStyleSheet('background: #080a10; border-top: 1px solid #181e2e;')
        col = QVBoxLayout(w)
        col.setContentsMargins(14, 6, 14, 6)
        col.setSpacing(6)

        # scrubber row
        scr_row = QHBoxLayout()
        self._lbl_pos = QLabel('0')
        self._lbl_pos.setStyleSheet('color: #555; font-size: 11px; min-width: 32px;')
        self._lbl_pos.setAlignment(Qt.AlignRight)
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, max(self.total_frames - 1, 1))
        self._lbl_total = QLabel(str(self.total_frames))
        self._lbl_total.setStyleSheet('color: #555; font-size: 11px; min-width: 40px;')
        scr_row.addWidget(self._lbl_pos)
        scr_row.addWidget(self.scrubber, 1)
        scr_row.addWidget(self._lbl_total)
        col.addLayout(scr_row)

        # button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_play  = QPushButton('▶  Play')
        self.btn_play.setCheckable(True)
        self.btn_play.setFixedHeight(32)

        self.btn_p10   = QPushButton('◀◀ -10')
        self.btn_p1    = QPushButton('◀ -1')
        self.btn_n1    = QPushButton('+1 ▶')
        self.btn_n10   = QPushButton('+10 ▶▶')
        for b in [self.btn_p10, self.btn_p1, self.btn_n1, self.btn_n10]:
            b.setFixedHeight(32)

        btn_row.addWidget(self.btn_p10)
        btn_row.addWidget(self.btn_p1)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_n1)
        btn_row.addWidget(self.btn_n10)
        btn_row.addStretch()

        # speed buttons
        spd_lbl = QLabel('Speed:')
        spd_lbl.setStyleSheet('color: #666; font-size: 12px;')
        btn_row.addWidget(spd_lbl)
        self._spd_btns = {}
        for label, val in [('1x', 1.0), ('½x', 0.5), ('¼x', 0.25), ('⅛x', 0.125)]:
            b = QPushButton(label)
            b.setFixedSize(40, 32)
            b.setCheckable(True)
            b.setChecked(val == 1.0)
            btn_row.addWidget(b)
            self._spd_btns[val] = b

        btn_row.addStretch()

        # inference toggle
        self.btn_inference = QPushButton('Pose ON')
        self.btn_inference.setCheckable(True)
        self.btn_inference.setChecked(True)
        self.btn_inference.setFixedHeight(32)
        btn_row.addWidget(self.btn_inference)

        # reselect
        self.btn_reselect = QPushButton('⟳ Reselect')
        self.btn_reselect.setFixedHeight(32)
        btn_row.addWidget(self.btn_reselect)

        col.addLayout(btn_row)

        # keyboard hint strip
        hint = QLabel(
            'Keyboard:  Space play/pause  ·  ← → or , . step frame  ·  '
            '1 2 4 8 speed  ·  Mouse wheel = frame step  ·  Esc quit'
        )
        hint.setStyleSheet('color: #445; font-size: 10px;')
        col.addWidget(hint)

        return w

    # ── metrics panel ────────────────────────────────────────────────────────

    def _metrics_panel(self):
        w = QWidget()
        w.setFixedWidth(270)
        w.setStyleSheet('background: #0a0c12; border-left: 1px solid #181e2e;')
        col = QVBoxLayout(w)
        col.setContentsMargins(10, 10, 10, 10)
        col.setSpacing(8)

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
            'Status', 'Frame', 'Box W × H', 'Center',
        ])

        self.m_pose = grp('Pose Layer — MoveNet', [
            'Pose', 'Hip R', 'Knee R', 'Hip L', 'Knee L', 'Torso lean',
        ])

        self.m_sprint = grp('Sprint Metrics', [
            'Speed', 'Calibration',
        ])
        self.m_sprint['Speed'].pending()
        self.m_sprint['Calibration'].set('not set', '#f5a623')

        self.m_pending = grp('Pending — v11', [
            'Stride length', 'Ground contact', 'Step frequency',
        ])
        for m in self.m_pending.values():
            m.pending()

        # coach notes
        ng = QGroupBox('Coach Notes')
        nv = QVBoxLayout(ng)
        self.notes = QTextEdit()
        self.notes.setPlaceholderText('Type coaching observations…')
        self.notes.setFixedHeight(72)
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
        for lbl in ['Body map', 'Fatigue signal', 'Readiness']:
            m = MetricRow(lbl)
            m.pending()
            pv.addWidget(m)
        col.addWidget(pp)

        return w

    # ──────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────────

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

        self.btn_reselect.clicked.connect(self._reselect)
        self.btn_inference.clicked.connect(self._toggle_inference)
        self.btn_cal.clicked.connect(self._start_cal)
        self.btn_export.clicked.connect(self._export_csv)

        for name, btn in self._tool_btns.items():
            btn.clicked.connect(lambda _, n=name: self._pick_tool(n))

    # ──────────────────────────────────────────────────────────────────────────
    # Playback
    # ──────────────────────────────────────────────────────────────────────────

    def _toggle_play(self, on):
        if on:
            self._play()
        else:
            self._pause()

    def _play(self):
        self.playing = True
        self.btn_play.setChecked(True)
        self.btn_play.setText('⏸  Pause')
        self.timer.start(max(int(1000 / (self.fps * self.speed)), 16))

    def _pause(self):
        self.playing = False
        self.btn_play.setChecked(False)
        self.btn_play.setText('▶  Play')
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

    # ──────────────────────────────────────────────────────────────────────────
    # Athlete detection & selection
    # ──────────────────────────────────────────────────────────────────────────

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
        self._render(self._current_bgr)
        self.m_athlete['Status'].set('Target Locked', '#00ffd0')

    def _reselect(self):
        self._pause()
        self._run_detection()

    # ──────────────────────────────────────────────────────────────────────────
    # Render (overlay + display)
    # ──────────────────────────────────────────────────────────────────────────

    def _render(self, bgr):
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        overlay = bgr.copy()

        if self.selected_box is None:
            # ── selection mode ──────────────────────────────────────────────
            for i, item in enumerate(self.detected_boxes):
                x1, y1, x2, y2 = item['box']
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 200), 2)
                label_y = max(y1 - 8, 24)
                cv2.rectangle(overlay, (x1, label_y - 22), (x1 + 130, label_y + 4),
                              (13, 16, 22), -1)
                cv2.putText(overlay, f'Athlete {i + 1}', (x1 + 6, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA)

            self._glass(overlay, 30, 22, w - 60, 72)
            cv2.putText(overlay, 'JL VISION  |  SELECT TARGET ATHLETE', (52, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(overlay, 'Click on the athlete you want to analyze', (52, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

        else:
            # ── review mode ─────────────────────────────────────────────────

            # Re-track athlete via IoU
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
            badge_y = max(y1 - 36, 115)
            self._glass(overlay, x1, badge_y, 170, 28, 0.75)
            cv2.putText(overlay, 'TARGET ATHLETE', (x1 + 8, badge_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 2, cv2.LINE_AA)

            # Pose inference
            if self.inference_on:
                kps = self.movenet.estimate(bgr, box=self.selected_box)
                self.keypoints = kps
                if kps is not None:
                    self._draw_skeleton(overlay, kps, w, h)
                    angles = self._draw_callouts(overlay, kps, w, h)
                    self._update_pose_metrics(kps, angles, w, h)
                    self.hdr_pose.setText('Pose: Active')
                else:
                    self.hdr_pose.setText('Pose: Not found')
                    self.m_pose['Pose'].set('Not found', '#f5a623')
            else:
                self.hdr_pose.setText('Pose: OFF')

            # Metrics panel
            self.m_athlete['Frame'].set(f'{self.current_frame + 1} / {self.total_frames}')
            self.m_athlete['Box W × H'].set(f'{bw} × {bh} px')
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
                cv2.line(frame, pa, pb, (0, 200, 255), 2, cv2.LINE_AA)
        for name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                     'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
            pt = kp_xy(kps, name, w, h)
            if pt:
                cv2.circle(frame, pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 9, (0, 200, 255), 1, cv2.LINE_AA)

    def _draw_callouts(self, frame, kps, w, h):
        """Draw offset angle tags; return dict of computed angles."""
        angles = {}

        def callout(joint, a_kp, b_kp, c_kp, label_prefix, off_x, off_y):
            pa = kp_xy(kps, a_kp, w, h)
            pb = kp_xy(kps, b_kp, w, h)
            pc = kp_xy(kps, c_kp, w, h)
            if pa and pb and pc:
                ang = angle_at(pa, pb, pc)
                angles[label_prefix] = ang
                tag_x = max(4, min(pb[0] + off_x, w - 148))
                tag_y = max(26, min(pb[1] + off_y, h - 8))
                cv2.line(frame, pb, (tag_x + 66, tag_y - 6),
                         (0, 255, 200), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (tag_x, tag_y - 22), (tag_x + 140, tag_y + 4),
                              (12, 15, 22), -1)
                cv2.rectangle(frame, (tag_x, tag_y - 22), (tag_x + 140, tag_y + 4),
                              (0, 255, 200), 1)
                cv2.putText(frame, f'{label_prefix}  {ang:.1f}°', (tag_x + 6, tag_y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 1, cv2.LINE_AA)

        callout('hip_r',  'right_shoulder', 'right_hip',  'right_knee',  'HIP-R',  60, -40)
        callout('knee_r', 'right_hip',      'right_knee', 'right_ankle', 'KNEE-R', 60,  40)
        callout('hip_l',  'left_shoulder',  'left_hip',   'left_knee',   'HIP-L', -160, -40)
        callout('knee_l', 'left_hip',       'left_knee',  'left_ankle',  'KNEE-L', -160, 40)
        return angles

    def _update_pose_metrics(self, kps, angles, w, h):
        self.m_pose['Pose'].set('Active', '#00ffd0')
        self.m_pose['Hip R'].set(f"{angles.get('HIP-R', '—'):.1f}°"
                                 if 'HIP-R' in angles else '—', '#00ffd0')
        self.m_pose['Knee R'].set(f"{angles.get('KNEE-R', '—'):.1f}°"
                                  if 'KNEE-R' in angles else '—', '#00ffd0')
        self.m_pose['Hip L'].set(f"{angles.get('HIP-L', '—'):.1f}°"
                                 if 'HIP-L' in angles else '—', '#00ffd0')
        self.m_pose['Knee L'].set(f"{angles.get('KNEE-L', '—'):.1f}°"
                                  if 'KNEE-L' in angles else '—', '#00ffd0')

        # Torso lean
        r_sh = kp_xy(kps, 'right_shoulder', w, h)
        l_sh = kp_xy(kps, 'left_shoulder', w, h)
        r_hp = kp_xy(kps, 'right_hip', w, h)
        l_hp = kp_xy(kps, 'left_hip', w, h)
        if all([r_sh, l_sh, r_hp, l_hp]):
            sh = ((r_sh[0] + l_sh[0]) // 2, (r_sh[1] + l_sh[1]) // 2)
            hp = ((r_hp[0] + l_hp[0]) // 2, (r_hp[1] + l_hp[1]) // 2)
            dx, dy = sh[0] - hp[0], sh[1] - hp[1]
            lean = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))
            direction = 'fwd' if dx > 0 else 'back'
            self.m_pose['Torso lean'].set(f'{lean:.1f}° {direction}', '#00ffd0')

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

    def _log_frame(self, cx, cy):
        entry = {
            'frame': self.current_frame,
            'cx': cx, 'cy': cy,
            'hip_r': '', 'knee_r': '', 'hip_l': '', 'knee_l': '',
            'speed_ms': '',
        }
        self._session_log.append(entry)

    # ──────────────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────────────

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
                self.m_sprint['Speed'].set('Tracking…', '#aaa')

    # ──────────────────────────────────────────────────────────────────────────
    # CSV export
    # ──────────────────────────────────────────────────────────────────────────

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Session CSV', 'jl_vision_session.csv', 'CSV (*.csv)')
        if not path:
            return
        fields = ['frame', 'cx', 'cy', 'hip_r', 'knee_r', 'hip_l', 'knee_l', 'speed_ms']
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self._session_log)
        QMessageBox.information(self, 'Exported', f'Saved to:\n{path}')

    # ──────────────────────────────────────────────────────────────────────────
    # Tool selection
    # ──────────────────────────────────────────────────────────────────────────

    def _pick_tool(self, name):
        for n, btn in self._tool_btns.items():
            btn.setChecked(n == name)

    # ──────────────────────────────────────────────────────────────────────────
    # Keyboard + wheel
    # ──────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python coach_workspace_v10.py <video_path>')
        sys.exit(1)
    video = Path(sys.argv[1])
    if not video.exists():
        print(f'Video not found: {video}')
        sys.exit(1)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = JLVisionV10(video)
    win.show()
    sys.exit(app.exec_())
