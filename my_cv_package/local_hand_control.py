#!/usr/bin/env python3
"""
============================================================================
 Local Hand Control — runs on your LAPTOP (Windows / no ROS needed)
============================================================================

 Captures the webcam, runs MediaPipe Pose Landmarker (new Tasks API) to
 track your hand, maps the hand barycenter to a 3×3 grid, and sends
 velocity commands over TCP to the remote ROS 2 node (hand_teleop).

 This script acts as a TCP SERVER — the remote node connects to it.

         ┌──────────────┬──────────────┬──────────────┐
         │ UPPER_LEFT   │      UP      │ UPPER_RIGHT  │
         │ fwd + L turn │   forward    │ fwd + R turn │
         ├──────────────┼──────────────┼──────────────┤
         │     LEFT     │   NEUTRAL    │    RIGHT     │
         │  turn left   │     stop     │  turn right  │
         ├──────────────┼──────────────┼──────────────┤
         │ LOWER_LEFT   │     DOWN     │ LOWER_RIGHT  │
         │ rev + L turn │   backward   │ rev + R turn │
         └──────────────┴──────────────┴──────────────┘

 Safety: no hand detected → stop (send zero).

 Keys: [s] enable sending  ·  [q] quit

 INSTALL (on your Windows laptop)
 ---------------------------------
   pip install mediapipe opencv-python numpy

 The script auto-downloads the pose model on first run.

 RUN
 ---
   python local_hand_control.py
   (then on the remote PC: ros2 run my_cv_package hand_teleop)
============================================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import socket
import json
import argparse
import time
import sys
import os
import urllib.request
import threading

# ─── Pose landmark indices (MediaPipe 33-point model) ────────────────────
LEFT_WRIST  = 15
LEFT_INDEX  = 19
LEFT_THUMB  = 21
LEFT_PINKY  = 17
RIGHT_WRIST = 16
RIGHT_INDEX = 20
RIGHT_THUMB = 22
RIGHT_PINKY = 18

# ─── Zone names ──────────────────────────────────────────────────────────
NEUTRAL      = 'NEUTRAL'
UP           = 'UP'
DOWN         = 'DOWN'
LEFT         = 'LEFT'
RIGHT        = 'RIGHT'
UPPER_LEFT   = 'UPPER_LEFT'
UPPER_RIGHT  = 'UPPER_RIGHT'
LOWER_LEFT   = 'LOWER_LEFT'
LOWER_RIGHT  = 'LOWER_RIGHT'

ZONE_CMD = {
    NEUTRAL:     (0.0,  0.0),
    UP:          (+1.0, 0.0),
    DOWN:        (-1.0, 0.0),
    LEFT:        (0.0, +1.0),
    RIGHT:       (0.0, -1.0),
    UPPER_LEFT:  (+1.0, +1.0),
    UPPER_RIGHT: (+1.0, -1.0),
    LOWER_LEFT:  (-1.0, +1.0),
    LOWER_RIGHT: (-1.0, -1.0),
}

ZONE_GRID = [
    [UPPER_LEFT, UP,      UPPER_RIGHT],
    [LEFT,       NEUTRAL, RIGHT      ],
    [LOWER_LEFT, DOWN,    LOWER_RIGHT],
]


def zone_from_normalized(nx, ny):
    col = 0 if nx < 1/3 else (2 if nx > 2/3 else 1)
    row = 0 if ny < 1/3 else (2 if ny > 2/3 else 1)
    return ZONE_GRID[row][col], row, col


def hand_barycenter(landmarks, use_left):
    """Average of wrist + index + thumb + pinky landmarks (normalized)."""
    if use_left:
        ids = [LEFT_WRIST, LEFT_INDEX, LEFT_THUMB, LEFT_PINKY]
    else:
        ids = [RIGHT_WRIST, RIGHT_INDEX, RIGHT_THUMB, RIGHT_PINKY]

    pts = []
    for i in ids:
        lm = landmarks[i]
        vis = getattr(lm, 'visibility', 1.0)
        if vis is None:
            vis = 1.0
        if vis > 0.4:
            pts.append(lm)

    if len(pts) < 2:
        return None
    x = sum(p.x for p in pts) / len(pts)
    y = sum(p.y for p in pts) / len(pts)
    return x, y


def draw_overlay(frame, landmarks, hand_xy, zone, row, col, v, w, running, connected):
    h, fw, _ = frame.shape

    if landmarks is not None:
        for lm in landmarks:
            cx, cy = int(lm.x * fw), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (180, 180, 180), -1)
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
            (25, 27), (26, 28),
        ]
        for a, b in connections:
            if a < len(landmarks) and b < len(landmarks):
                pt1 = (int(landmarks[a].x * fw), int(landmarks[a].y * h))
                pt2 = (int(landmarks[b].x * fw), int(landmarks[b].y * h))
                cv2.line(frame, pt1, pt2, (120, 120, 120), 1)

    TOP = 52
    for i in (1, 2):
        x = i * fw // 3
        cv2.line(frame, (x, TOP), (x, h), (140, 140, 140), 1)
        y = TOP + i * (h - TOP) // 3
        cv2.line(frame, (0, y), (fw, y), (140, 140, 140), 1)

    x0, x1 = col * fw // 3, (col + 1) * fw // 3
    cell_h = (h - TOP) // 3
    y0, y1 = TOP + row * cell_h, TOP + (row + 1) * cell_h
    cell_color = (0, 200, 0) if zone == NEUTRAL else (0, 165, 255)
    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), cell_color, -1)
    cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), cell_color, 2)

    if hand_xy is not None:
        cv2.drawMarker(frame, hand_xy, (0, 220, 255),
                       cv2.MARKER_CROSS, 24, 3)
        cv2.circle(frame, hand_xy, 14, (0, 220, 255), 2)

    cv2.rectangle(frame, (0, 0), (fw, TOP), (30, 30, 30), -1)
    run_str = '[RUNNING]' if running else '[STOPPED]'
    conn_str = 'CONNECTED' if connected else 'WAITING FOR REMOTE...'
    status = zone if hand_xy is not None else 'NO HAND'
    cv2.putText(frame,
                f'{run_str}  {status:>12}  v={v:+.2f}  w={w:+.2f}',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    conn_color = (0, 220, 0) if connected else (0, 0, 255)
    cv2.putText(frame, conn_str,
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, conn_color, 1)
    cv2.putText(frame, '[s]start  [q]quit',
                (fw - 190, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 220, 0), 1)

    cv2.imshow('Hand Teleop (Local)', frame)


def download_model(model_path):
    """Download the pose landmarker model if not present."""
    url = ('https://storage.googleapis.com/mediapipe-models/'
           'pose_landmarker/pose_landmarker_lite/float16/1/'
           'pose_landmarker_lite.task')
    print(f'[INFO] Downloading pose model to {model_path} ...')
    urllib.request.urlretrieve(url, model_path)
    print(f'[OK] Model downloaded ({os.path.getsize(model_path) / 1e6:.1f} MB)')


def main():
    parser = argparse.ArgumentParser(description='Local hand control (TCP server)')
    parser.add_argument('--port', type=int, default=9090,
                        help='TCP port to listen on (default: 9090)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--max-linear', type=float, default=0.18,
                        help='Max linear speed m/s (default: 0.18)')
    parser.add_argument('--max-angular', type=float, default=0.8,
                        help='Max angular speed rad/s (default: 0.8)')
    parser.add_argument('--left-hand', action='store_true',
                        help='Track left hand instead of right')
    parser.add_argument('--model', type=str, default='pose_landmarker_lite.task',
                        help='Path to the pose landmarker .task model file')
    args = parser.parse_args()

    # ── Download model if needed ──
    model_path = args.model
    if not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, os.path.basename(model_path))
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            download_model(model_path)

    # ── MediaPipe Pose Landmarker (VIDEO mode) ──
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # ── Camera ──
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open camera {args.camera}')
        sys.exit(1)
    print(f'[OK] Camera {args.camera} opened')

    # ── TCP SERVER — accept connection from remote node ──
    client_sock = None
    connected = False
    lock = threading.Lock()

    def accept_loop():
        nonlocal client_sock, connected
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('0.0.0.0', args.port))
        srv.listen(1)
        print(f'[OK] TCP server listening on 0.0.0.0:{args.port}')
        print(f'[INFO] On the remote PC run:')
        print(f'       ros2 run my_cv_package hand_teleop')
        while True:
            try:
                conn, addr = srv.accept()
                print(f'[OK] Remote node connected from {addr}')
                with lock:
                    # close old client if any
                    if client_sock:
                        try:
                            client_sock.close()
                        except Exception:
                            pass
                    client_sock = conn
                    connected = True
            except Exception as e:
                print(f'[WARN] Accept error: {e}')

    t = threading.Thread(target=accept_loop, daemon=True)
    t.start()

    def send_cmd(v, w):
        nonlocal client_sock, connected
        with lock:
            if not connected or client_sock is None:
                return
            msg = json.dumps({'v': round(v, 4), 'w': round(w, 4)}) + '\n'
            try:
                client_sock.sendall(msg.encode())
            except (BrokenPipeError, ConnectionResetError, OSError):
                print('[WARN] Remote disconnected, waiting for reconnect...')
                connected = False
                try:
                    client_sock.close()
                except Exception:
                    pass
                client_sock = None

    running = False
    frame_ts = 0

    print(f'[INFO] Tracking {"left" if args.left_hand else "right"} hand')
    print(f'[INFO] Press [s] in the webcam window to start sending commands')

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w_px, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_ts += 33
        result = landmarker.detect_for_video(mp_image, frame_ts)

        v, ang = 0.0, 0.0
        zone, row, col = NEUTRAL, 1, 1
        hand_xy = None
        landmarks_list = None

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks_list = result.pose_landmarks[0]
            bc = hand_barycenter(landmarks_list, args.left_hand)
            if bc is not None:
                nx, ny = bc
                hand_xy = (int(nx * w_px), int(ny * h))
                zone, row, col = zone_from_normalized(nx, ny)
                lin_s, ang_s = ZONE_CMD[zone]
                v   = lin_s * args.max_linear
                ang = ang_s * args.max_angular

        if running:
            send_cmd(v, ang)

        with lock:
            is_connected = connected

        draw_overlay(frame, landmarks_list, hand_xy, zone, row, col,
                     v, ang, running, is_connected)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            running = True
            print('[▶] Sending commands')
        elif key == ord('q'):
            running = False
            send_cmd(0.0, 0.0)
            print('[■] Stopped — quitting')
            break

    send_cmd(0.0, 0.0)
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    with lock:
        if client_sock:
            client_sock.close()


if __name__ == '__main__':
    main()
