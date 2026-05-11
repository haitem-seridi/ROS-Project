#!/usr/bin/env python3
"""
Challenge 1 — Line Follower Node (v2)
======================================
Single-line following with pixel offset for lane centering.

Algorithm:
  1. Grab camera frame → enhance (CLAHE + saturation boost)
  2. Build HSV masks for red and green lines (FULL FRAME, no ROI crop)
  3. Count pixels: red vs green
  4. Follow whichever line has MORE pixels (tunable hysteresis)
  5. Find centroid (cx) of the chosen line
  6. Compute target position with uni-directional offset:
     - RED (right boundary): target = cx - offset_pixels  (aim LEFT of red)
     - GREEN (left boundary): target = cx + offset_pixels (aim RIGHT of green)
  7. Error = target_x - frame_center_x
  8. PID on error → angular velocity
  9. Publish /cmd_vel

Speed starts at 0 — tune HSV/masks/offset in the tuner, then increase speed.
Publishes /debug_image — view with: rqt_image_view /debug_image
"""

from typing import Optional
import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy,
    QoSDurabilityPolicy, QoSHistoryPolicy,
    qos_profile_sensor_data,
)
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist


def _desc(d: str) -> ParameterDescriptor:
    pd = ParameterDescriptor()
    pd.description = d
    return pd


class LineFollowerNode(Node):

    def __init__(self):
        super().__init__('line_follower')
        self._bridge = CvBridge()
        self._clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ── Enhancement ────────────────────────────────────────────────
        self.declare_parameter('use_enhancement', True,
                               _desc('CLAHE + saturation boost'))
        self.declare_parameter('saturation_boost', 1.4,
                               _desc('Saturation multiplier'))

        # ── Green HSV ──────────────────────────────────────────────────
        # Lowered S/V for bright room where colors appear faint
        self.declare_parameter('green_h_low',   35, _desc('Green hue lower'))
        self.declare_parameter('green_s_low',   40, _desc('Green saturation lower'))
        self.declare_parameter('green_v_low',   40, _desc('Green value lower'))
        self.declare_parameter('green_h_high',  90, _desc('Green hue upper'))

        # ── Red HSV (dual range — red wraps at 0/180) ─────────────────
        self.declare_parameter('red_h_low1',     0, _desc('Red range-1 hue lower'))
        self.declare_parameter('red_h_high1',   12, _desc('Red range-1 hue upper'))
        self.declare_parameter('red_h_low2',   155, _desc('Red range-2 hue lower'))
        self.declare_parameter('red_h_high2',  180, _desc('Red range-2 hue upper'))
        self.declare_parameter('red_s_low',     50, _desc('Red saturation lower'))
        self.declare_parameter('red_v_low',     40, _desc('Red value lower'))

        # ── Offset ──────────────────────────────────────────────────────
        # Should be roughly HALF the lane width in pixels.
        # Too large → target overshoots past center → robot oscillates.
        # Tune via tuner: watch the cyan TARGET line, it should sit at lane center.
        self.declare_parameter('offset_pixels', 100,
                               _desc('Pixel offset from tracked line into the lane. '
                                     'Set to ~half the lane width in pixels.'))

        # ── Line detection ─────────────────────────────────────────────
        self.declare_parameter('min_line_area', 300,
                               _desc('Min pixels to consider a line detected'))
        self.declare_parameter('switch_hysteresis', 1.5,
                               _desc('Need Nx more pixels to switch tracked line'))

        # ── PID ────────────────────────────────────────────────────────
        self.declare_parameter('kp',   0.002,  _desc('Proportional gain'))
        self.declare_parameter('ki',   0.0,    _desc('Integral gain — keep 0 until Kp is tuned'))
        self.declare_parameter('kd',   0.001,  _desc('Derivative gain'))

        # ── Motion ─────────────────────────────────────────────────────
        # SPEED STARTS AT ZERO so you can tune before the robot moves.
        self.declare_parameter('linear_speed',     0.0,
                               _desc('Forward speed [m/s]. Increase after tuning.'))
        self.declare_parameter('max_angular_speed', 2.0,
                               _desc('Angular clamp [rad/s]'))

        self.add_on_set_parameters_callback(self._on_param_change)

        # ── Internal state ─────────────────────────────────────────────
        self._prev_error:        float          = 0.0
        self._integral:          float          = 0.0
        self._last_stamp                        = None
        self._last_known_target: Optional[float] = None
        self._active_line:       str            = 'red'
        self._active_area:       int            = 0
        self._frame_width:       int            = 640
        self._last_frame_time:   float          = 0.0  # monotonic clock

        # ── Publishers / Subscribers ───────────────────────────────────
        # QoS = RELIABLE depth 10 — matches Zenoh bridge publisher QoS.
        cq = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed',
            self._compressed_cb, cq)
        self.create_subscription(
            Image, '/image_raw', self._image_cb, cq)
        self.create_subscription(
            Image, '/camera/image_raw', self._image_cb, cq)

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', cq)
        self._dbg_pub = self.create_publisher(Image,  '/debug_image', 1)

        # ── Watchdog: stop robot if camera feed dies ───────────────────
        import time as _time
        self._time = _time
        self._last_frame_time = _time.monotonic()
        self._watchdog_timer = self.create_timer(0.5, self._watchdog_cb)

        self.get_logger().info(
            'LineFollowerNode started — speed=0, tune params then increase.')

    # ── Param callback ─────────────────────────────────────────────────

    def _on_param_change(self, params) -> SetParametersResult:
        for p in params:
            self.get_logger().info(f'[param] {p.name} = {p.value}')
        return SetParametersResult(successful=True)

    # ── Watchdog — stop robot if camera feed dies ──────────────────────

    def _watchdog_cb(self):
        elapsed = self._time.monotonic() - self._last_frame_time
        if elapsed > 0.5 and self._last_frame_time > 0.0:
            self._cmd_pub.publish(Twist())  # STOP
            self._reset_pid()

    # ── Camera callbacks ───────────────────────────────────────────────

    def _compressed_cb(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, np.uint8)
        f   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if f is not None:
            self._process_frame(f)

    def _image_cb(self, msg: Image):
        self._process_frame(
            self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'))

    # ── Core processing ────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray):
        self._last_frame_time = self._time.monotonic()
        frame = self._enhance_frame(frame)
        H, W  = frame.shape[:2]
        self._frame_width = W

        # ── Build masks (full frame) ───────────────────────────────────
        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        g_mask = self._green_mask(hsv)
        r_mask = self._red_mask(hsv)

        # Morphological cleanup — remove small noise blobs
        kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        g_mask = cv2.morphologyEx(g_mask, cv2.MORPH_OPEN,  kern)
        g_mask = cv2.morphologyEx(g_mask, cv2.MORPH_CLOSE, kern)
        r_mask = cv2.morphologyEx(r_mask, cv2.MORPH_OPEN,  kern)
        r_mask = cv2.morphologyEx(r_mask, cv2.MORPH_CLOSE, kern)

        area_g = int(cv2.countNonZero(g_mask))
        area_r = int(cv2.countNonZero(r_mask))

        mina   = self.get_parameter('min_line_area').value
        hyst   = self.get_parameter('switch_hysteresis').value
        offset = self.get_parameter('offset_pixels').value

        # ── Dynamic line selection with hysteresis ─────────────────────
        prev_line = self._active_line
        if self._active_line == 'red':
            if area_g > area_r * hyst and area_g >= mina:
                self._active_line = 'green'
                self._active_area = area_g
                self.get_logger().info(
                    f'[LINE] red→green  (g={area_g} r={area_r})')
            else:
                self._active_area = area_r
        else:
            if area_r > area_g * hyst and area_r >= mina:
                self._active_line = 'red'
                self._active_area = area_r
                self.get_logger().info(
                    f'[LINE] green→red  (r={area_r} g={area_g})')
            else:
                self._active_area = area_g

        if prev_line != self._active_line:
            self._reset_pid()

        # ── Centroid of active line ────────────────────────────────────
        active_mask = r_mask if self._active_line == 'red' else g_mask
        cx = self._centroid_x(active_mask, mina)

        # ── Target position (uni-directional offset) ───────────────────
        if cx is not None:
            if self._active_line == 'red':
                target_x = cx - offset
            else:
                target_x = cx + offset
            self._last_known_target = target_x
        elif self._last_known_target is not None:
            target_x = self._last_known_target
        else:
            self._publish_debug(frame, g_mask, r_mask, None, None,
                                0.0, area_r, area_g)
            self._cmd_pub.publish(Twist())
            return

        # ── SAFETY: if speed=0, stop completely — no PID accumulation ──
        linear = self.get_parameter('linear_speed').value
        if linear <= 0.0:
            self._reset_pid()
            self._cmd_pub.publish(Twist())
            self._publish_debug(frame, g_mask, r_mask, cx, target_x,
                                0.0, area_r, area_g)
            return

        # ── Error = target - frame center ──────────────────────────────
        error = target_x - (W / 2.0)

        # ── PID → angular velocity ────────────────────────────────────
        omega   = self._pid(error)
        max_omg = self.get_parameter('max_angular_speed').value

        t = Twist()
        t.linear.x  = float(linear)
        t.angular.z = float(np.clip(omega, -max_omg, max_omg))
        self._cmd_pub.publish(t)

        # ── Debug image ────────────────────────────────────────────────
        self._publish_debug(frame, g_mask, r_mask, cx, target_x,
                            error, area_r, area_g)

    # ── Debug image ────────────────────────────────────────────────────

    def _publish_debug(self, frame, g_mask, r_mask, cx, target_x,
                       error, area_r, area_g):
        if self._dbg_pub.get_subscription_count() == 0:
            return

        H, W    = frame.shape[:2]
        display = frame.copy()

        # Mask overlays
        ov = display.copy()
        ov[g_mask > 0] = (0, 220, 0)
        ov[r_mask > 0] = (0, 0, 220)
        cv2.addWeighted(ov, 0.4, display, 0.6, 0, display)

        # Frame center line
        cv2.line(display, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)

        # Active-line centroid marker
        color = (0, 0, 255) if self._active_line == 'red' else (0, 255, 0)
        if cx is not None:
            cv2.circle(display, (int(cx), H // 2), 10, color, -1)
            cv2.putText(display, 'cx', (int(cx) + 12, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Target line
        if target_x is not None:
            cv2.line(display, (int(target_x), 0), (int(target_x), H),
                     (0, 255, 255), 2)

        # Status bar
        offset = self.get_parameter('offset_pixels').value
        speed  = self.get_parameter('linear_speed').value
        det    = 'OK' if cx else 'LOST'
        txt = (f'LINE:{self._active_line.upper()} {det} | '
               f'cx={int(cx) if cx else "?":>4} | '
               f'err={int(error):+d} | '
               f'off={offset} | spd={speed:.2f} | '
               f'R={area_r} G={area_g}')
        cv2.rectangle(display, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(display, txt, (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

        try:
            self._dbg_pub.publish(
                self._bridge.cv2_to_imgmsg(display, encoding='bgr8'))
        except Exception:
            pass

    # ── Helpers ────────────────────────────────────────────────────────

    def _centroid_x(self, mask, min_area):
        M = cv2.moments(mask)
        return (M['m10'] / M['m00']) if M['m00'] >= min_area else None

    def _enhance_frame(self, frame):
        if not self.get_parameter('use_enhancement').value:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] * self.get_parameter('saturation_boost').value,
            0, 255)
        hsv[:, :, 2] = self._clahe.apply(
            hsv[:, :, 2].astype(np.uint8)).astype(np.float32)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _green_mask(self, hsv):
        return cv2.inRange(
            hsv,
            np.array([self.get_parameter('green_h_low').value,
                      self.get_parameter('green_s_low').value,
                      self.get_parameter('green_v_low').value],
                     dtype=np.uint8),
            np.array([self.get_parameter('green_h_high').value,
                      255, 255], dtype=np.uint8))

    def _red_mask(self, hsv):
        s = self.get_parameter('red_s_low').value
        v = self.get_parameter('red_v_low').value
        m1 = cv2.inRange(
            hsv,
            np.array([self.get_parameter('red_h_low1').value, s, v],
                     dtype=np.uint8),
            np.array([self.get_parameter('red_h_high1').value, 255, 255],
                     dtype=np.uint8))
        m2 = cv2.inRange(
            hsv,
            np.array([self.get_parameter('red_h_low2').value, s, v],
                     dtype=np.uint8),
            np.array([self.get_parameter('red_h_high2').value, 255, 255],
                     dtype=np.uint8))
        return cv2.bitwise_or(m1, m2)

    def _pid(self, error):
        now = self.get_clock().now()
        dt  = 0.033 if self._last_stamp is None else \
              (now - self._last_stamp).nanoseconds * 1e-9
        if dt <= 0.0 or dt > 0.5:
            dt = 0.033
        self._last_stamp = now
        self._integral   = float(np.clip(
            self._integral + error * dt, -300.0, 300.0))
        deriv            = (error - self._prev_error) / dt
        self._prev_error = error
        return -(self.get_parameter('kp').value * error +
                 self.get_parameter('ki').value * self._integral +
                 self.get_parameter('kd').value * deriv)

    def _reset_pid(self):
        self._prev_error    = 0.0
        self._integral      = 0.0
        self._last_stamp    = None
        self._last_known_target = None


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cmd_pub.publish(Twist())  # stop robot
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()