#!/usr/bin/env python3
"""
hsv_tuner_node.py — cv2 trackbar tuner for /line_follower
==========================================================
Two windows:
  Camera   — top: camera + colored overlay, bottom: binary masks on black
  Controls — all trackbar sliders

  Pushes slider values to /line_follower via ros2 param set.
ROS2 spins in a background thread; cv2 owns the main thread.

Run:  ros2 run my_cv_package hsv_tuner_node
"""
import threading
import subprocess
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


def _set(name, value):
    """Push a parameter to /line_follower in a background thread."""
    threading.Thread(
        target=lambda: subprocess.run(
            ['ros2', 'param', 'set', '/line_follower', name, str(value)],
            capture_output=True),
        daemon=True).start()


class HsvTunerNode(Node):
    def __init__(self):
        super().__init__('hsv_tuner')
        self._bridge = CvBridge()
        self._frame  = None
        self._lock   = threading.Lock()

        sq = qos_profile_sensor_data
        self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self._cb_comp, sq)
        self.create_subscription(Image, '/image_raw', self._cb_raw, sq)
        self.create_subscription(Image, '/camera/image_raw', self._cb_raw, sq)

    def _cb_comp(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        f   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if f is not None:
            with self._lock:
                self._frame = f

    def _cb_raw(self, msg):
        try:
            with self._lock:
                self._frame = self._bridge.imgmsg_to_cv2(
                    msg, desired_encoding='bgr8')
        except Exception:
            pass

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


def main(args=None):
    rclpy.init(args=args)
    node = HsvTunerNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    WIN_IMG = 'Camera'
    WIN_CTL = 'Controls'
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kern    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Create windows and pump Qt
    cv2.namedWindow(WIN_IMG, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_IMG, 640, 600)
    cv2.namedWindow(WIN_CTL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTL, 500, 100)
    blank = np.zeros((100, 500, 3), dtype=np.uint8)
    cv2.imshow(WIN_CTL, blank)
    cv2.imshow(WIN_IMG, np.zeros((480, 640, 3), dtype=np.uint8))
    for _ in range(10):
        cv2.waitKey(50)

    # Slider state
    st = dict(
        v=40, s=40,
        gh_lo=35, gh_hi=90,
        rh_lo=0, rh_hi=12,
        rh2_lo=155, rh2_hi=180,
        offset=256, speed=0,
        kp=5, ki=1, kd=2,
        hyst=15, min_area=300, sat=14,
    )

    # Callbacks
    def on_v(v):
        st['v'] = v; _set('green_v_low', v); _set('red_v_low', v)

    def on_s(v):
        st['s'] = v; _set('green_s_low', v); _set('red_s_low', v)

    def on_gh_lo(v):  st['gh_lo']  = v; _set('green_h_low',  v)
    def on_gh_hi(v):  st['gh_hi']  = v; _set('green_h_high', v)
    def on_rh_lo(v):  st['rh_lo']  = v; _set('red_h_low1',   v)
    def on_rh_hi(v):  st['rh_hi']  = v; _set('red_h_high1',  v)
    def on_rh2_lo(v): st['rh2_lo'] = v; _set('red_h_low2',   v)
    def on_rh2_hi(v): st['rh2_hi'] = v; _set('red_h_high2',  v)
    def on_offset(v): st['offset'] = v; _set('offset_pixels', v)
    def on_speed(v):  st['speed']  = v; _set('linear_speed',  round(v * 0.01, 3))
    def on_kp(v):     st['kp']     = v; _set('kp',            round(v * 0.001, 4))
    def on_ki(v):     st['ki']     = v; _set('ki',            round(v * 0.0001, 5))
    def on_kd(v):     st['kd']     = v; _set('kd',            round(v * 0.001, 4))
    def on_hyst(v):   st['hyst']   = v; _set('switch_hysteresis', round(v * 0.1, 1))
    def on_area(v):   st['min_area'] = max(v, 10); _set('min_line_area', max(v, 10))
    def on_sat(v):    st['sat']    = v; _set('saturation_boost', round(v * 0.1, 2))

    def tb(lbl, key, maxv, cb):
        cv2.createTrackbar(lbl, WIN_CTL, st[key], maxv, cb)

    tb('V low',      'v',        255, on_v)
    tb('S low',      's',        255, on_s)
    tb('G Hue lo',   'gh_lo',     90, on_gh_lo)
    tb('G Hue hi',   'gh_hi',     90, on_gh_hi)
    tb('R Hue lo',   'rh_lo',     30, on_rh_lo)
    tb('R Hue hi',   'rh_hi',     30, on_rh_hi)
    tb('R2 Hue lo',  'rh2_lo',   180, on_rh2_lo)
    tb('R2 Hue hi',  'rh2_hi',   180, on_rh2_hi)
    tb('Offset px',  'offset',   400, on_offset)
    tb('Speed x100', 'speed',     30, on_speed)
    tb('Kp x1000',   'kp',        50, on_kp)
    tb('Ki x10000',  'ki',        50, on_ki)
    tb('Kd x1000',   'kd',        50, on_kd)
    tb('Hyst x10',   'hyst',      30, on_hyst)
    tb('Min area',   'min_area', 5000, on_area)
    tb('Sat x10',    'sat',       30, on_sat)

    # Helper — defined once outside the loop; closes over `st`
    def cx_of(m):
        Mv = cv2.moments(m)
        return (Mv['m10'] / Mv['m00']) if Mv['m00'] >= st['min_area'] else None

    # ── Main loop ──────────────────────────────────────────────────────
    while rclpy.ok():
        frame = node.get_frame()
        if frame is None:
            waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(waiting_frame, 'Waiting for camera...', (30, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 200), 2)
            cv2.imshow(WIN_IMG, waiting_frame)
            if cv2.waitKey(33) == ord('q'):
                break
            continue

        # Enhancement
        hsv_e = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_e[:, :, 1] = np.clip(hsv_e[:, :, 1] * (st['sat'] * 0.1), 0, 255)
        hsv_e[:, :, 2] = clahe.apply(
            hsv_e[:, :, 2].astype(np.uint8)).astype(np.float32)
        frame = cv2.cvtColor(hsv_e.astype(np.uint8), cv2.COLOR_HSV2BGR)

        H, W = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v, s = st['v'], st['s']

        # Masks
        g_mask = cv2.inRange(hsv,
            np.array([st['gh_lo'], s, v], dtype=np.uint8),
            np.array([st['gh_hi'], 255, 255], dtype=np.uint8))
        r_mask = cv2.bitwise_or(
            cv2.inRange(hsv,
                np.array([st['rh_lo'], s, v], dtype=np.uint8),
                np.array([st['rh_hi'], 255, 255], dtype=np.uint8)),
            cv2.inRange(hsv,
                np.array([st['rh2_lo'], s, v], dtype=np.uint8),
                np.array([st['rh2_hi'], 255, 255], dtype=np.uint8)))

        # Morphological cleanup — removes small noise blobs
        g_mask = cv2.morphologyEx(g_mask, cv2.MORPH_OPEN,  kern)
        g_mask = cv2.morphologyEx(g_mask, cv2.MORPH_CLOSE, kern)
        r_mask = cv2.morphologyEx(r_mask, cv2.MORPH_OPEN,  kern)
        r_mask = cv2.morphologyEx(r_mask, cv2.MORPH_CLOSE, kern)

        area_g = cv2.countNonZero(g_mask)
        area_r = cv2.countNonZero(r_mask)

        # ── TOP: camera + colored overlay ──────────────────────────────
        display = frame.copy()
        ov = display.copy()
        ov[g_mask > 0] = (0, 220, 0)
        ov[r_mask > 0] = (0, 0, 220)
        cv2.addWeighted(ov, 0.4, display, 0.6, 0, display)
        cv2.line(display, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)

        cx_g = cx_of(g_mask)
        cx_r = cx_of(r_mask)
        active    = 'red' if area_r >= area_g else 'green'
        active_cx = cx_r  if active == 'red'  else cx_g

        if cx_r is not None:
            cv2.circle(display, (int(cx_r), H // 2), 8, (0, 0, 255), -1)
            cv2.line(display, (int(cx_r - st['offset']), 0),
                     (int(cx_r - st['offset']), H), (0, 100, 255), 1)
        if cx_g is not None:
            cv2.circle(display, (int(cx_g), H // 2), 8, (0, 255, 0), -1)
            cv2.line(display, (int(cx_g + st['offset']), 0),
                     (int(cx_g + st['offset']), H), (100, 255, 0), 1)

        err = 0
        if active_cx is not None:
            tgt = (active_cx - st['offset']) if active == 'red' \
                  else (active_cx + st['offset'])
            cv2.line(display, (int(tgt), 0), (int(tgt), H),
                     (0, 255, 255), 2)
            err = int(tgt - W / 2)

        txt = (f'ACT:{active.upper()} | R={area_r} G={area_g} | '
               f'err={err:+d} | off={st["offset"]} | '
               f'spd={st["speed"] * 0.01:.2f}')
        cv2.rectangle(display, (0, 0), (W, 28), (0, 0, 0), -1)
        cv2.putText(display, txt, (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # ── BOTTOM: binary masks on black ──────────────────────────────
        mask_view = np.zeros((H, W, 3), dtype=np.uint8)
        mask_view[g_mask > 0] = (0, 255, 0)
        mask_view[r_mask > 0] = (0, 0, 255)
        cv2.putText(mask_view, f'GREEN px={area_g}', (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(mask_view, f'RED px={area_r}', (W - 150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Stack: camera on top, masks on bottom
        combined = np.vstack([display, mask_view])
        cv2.imshow(WIN_IMG, combined)

        if cv2.waitKey(33) == ord('q'):
            break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()