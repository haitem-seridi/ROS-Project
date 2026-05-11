#!/usr/bin/env python3
"""
============================================================================
 Challenge 2 — Line following with yellow obstacle avoidance + GUI
============================================================================

 ┌─────────────────────────────────────────────────────────────────────────┐
 │ INSTALLATION                                                            │
 │                                                                         │
 │   1. Drop this file at:                                                 │
 │        ~/ros2_ws/src/projet/projet/obstacle_avoidance.py                │
 │      ── replace `projet` with your package name if it differs.          │
 │                                                                         │
 │   2. In setup.py, register the entry point:                             │
 │        entry_points={                                                   │
 │          'console_scripts': [                                           │
 │            'obstacle_avoidance = projet.obstacle_avoidance:main',       │
 │            #  └── node executable name    └── package    └── filename   │
 │          ],                                                             │
 │        }                                                                │
 │                                                                         │
 │   3. Build & source:                                                    │
 │        cd ~/ros2_ws && colcon build --packages-select projet            │
 │        source install/setup.bash                                        │
 │                                                                         │
 │   4. Run:                                                               │
 │        ros2 run projet obstacle_avoidance                               │
 │                                                                         │
 │   GUI: requires a display. On headless / SSH-without-X, launch with     │
 │        --ros-args -p enable_gui:=false                                  │
 └─────────────────────────────────────────────────────────────────────────┘

 STATE MACHINE
 -------------
   LINE_FOLLOWING --[yellow blob big enough + lidar confirms]--> AVOID_TURN
   AVOID_TURN     --[target blob exits FOV]--> AVOID_DRIVE   (record θ)
   AVOID_DRIVE    --[lidar shows obstacle abeam]--> AVOID_COUNTER
   AVOID_COUNTER  --[rotated 2·θ in opposite direction]--> AVOID_RETURN
   AVOID_RETURN   --[camera: lane centered]--> AVOID_REALIGN
   AVOID_REALIGN  --[rotated θ back to forward]--> LINE_FOLLOWING

 GUI
 ---
   Window  "Challenge2_Control" : trackbars (live tuning) + start/stop toggle
   Window  "Challenge2_POV"     : camera feed + state, metrics, error bar,
                                  yellow-blob bbox, key bindings
   Window  "Challenge2_Masks"   : combined view — yellow / red / green masks

   Keys (focus on any of the three windows):
     s        start
     q        stop
     +  /  =  speed up
     -        slow down
     d        toggle metrics overlay
     r        clear self-calibration (re-calibrates on next trigger)
     c        log current HSV ranges + calibration
     p        log full state snapshot
============================================================================
"""

import math
from enum import Enum, auto

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, LaserScan


class State(Enum):
    LINE_FOLLOWING = auto()
    AVOID_TURN = auto()
    AVOID_DRIVE = auto()
    AVOID_COUNTER = auto()
    AVOID_RETURN = auto()
    AVOID_REALIGN = auto()


# BGR colors per state for the POV banner
STATE_COLORS = {
    State.LINE_FOLLOWING: (0,   200, 0),     # green
    State.AVOID_TURN:     (0,   165, 255),   # orange
    State.AVOID_DRIVE:    (0,   220, 220),   # yellow
    State.AVOID_COUNTER:  (60,   80, 255),   # red-orange
    State.AVOID_RETURN:   (220, 220, 0),     # cyan
    State.AVOID_REALIGN:  (220, 0,   220),   # magenta
}


class ObstacleAvoidanceNode(Node):

    def __init__(self):
        # >>> EDIT: rename node here if you want a different name in `ros2 node list`.
        super().__init__('obstacle_avoidance')

        # ==================================================================
        # PARAMETERS — overridable via launch file. >>> TUNE = check on real
        # robot. >>> EDIT = topic / name overrides.
        # ==================================================================

        # >>> TUNE
        self.declare_parameter('linear_speed', 0.12)
        self.declare_parameter('angular_speed', 0.5)

        # >>> TUNE (most important): pixel-area floor for triggering avoidance.
        # Replaced by the calibrated value after the first real obstacle.
        self.declare_parameter('yellow_trigger_area', 3000)

        # Lidar confirmation distance (sanity check, not primary trigger)
        self.declare_parameter('lidar_confirm_distance', 1.20)

        self.declare_parameter('forward_arc_deg', 25.0)
        self.declare_parameter('side_arc_deg', 25.0)

        # >>> TUNE
        self.declare_parameter('lane_kp', 0.004)
        self.declare_parameter('lane_centered_px', 30)

        self.declare_parameter('blob_lost_frames', 3)

        # >>> EDIT — verify with `ros2 topic list`
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_topic', '/cmd_vel')

        # >>> TUNE on real robot — record a rosbag in lab lighting and tune
        self.declare_parameter('yellow_hsv_lower', [20, 100, 80])
        self.declare_parameter('yellow_hsv_upper', [35, 255, 255])
        self.declare_parameter('red_hsv_lower',  [0, 100, 80])
        self.declare_parameter('red_hsv_upper',  [10, 255, 255])
        self.declare_parameter('green_hsv_lower', [40, 80, 60])
        self.declare_parameter('green_hsv_upper', [85, 255, 255])

        # >>> EDIT: set false on a headless box / SSH without X-forwarding
        self.declare_parameter('enable_gui', True)

        g = lambda n: self.get_parameter(n).value
        self.v_lin = g('linear_speed')
        self.v_ang = g('angular_speed')
        self.trigger_area = g('yellow_trigger_area')
        self.lidar_confirm = g('lidar_confirm_distance')
        self.fwd_arc = math.radians(g('forward_arc_deg'))
        self.side_arc = math.radians(g('side_arc_deg'))
        self.lane_kp = g('lane_kp')
        self.lane_tol_px = g('lane_centered_px')
        self.blob_lost_n = g('blob_lost_frames')
        self.yellow_lo = np.array(g('yellow_hsv_lower'))
        self.yellow_hi = np.array(g('yellow_hsv_upper'))
        self.red_lo    = np.array(g('red_hsv_lower'))
        self.red_hi    = np.array(g('red_hsv_upper'))
        self.green_lo  = np.array(g('green_hsv_lower'))
        self.green_hi  = np.array(g('green_hsv_upper'))
        self._gui_enabled = g('enable_gui')

        # ==================================================================
        # RUNTIME STATE
        # ==================================================================
        self.state = State.LINE_FOLLOWING
        self.bridge = CvBridge()

        self.latest_scan: LaserScan | None = None
        self.current_yaw = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.have_odom = False

        self.theta = 0.0
        self.phase_start_yaw = 0.0
        self.turn_dir = 0          # +1 left, -1 right
        self.target_side = None    # 'left' / 'right'

        # Obstacle pinned in odom frame at trigger time. Used by AVOID_DRIVE
        # to compute the obstacle's bearing in the robot's *current* frame and
        # decide when it's truly behind us.
        self.obstacle_odom_x = 0.0
        self.obstacle_odom_y = 0.0
        self.have_obstacle_pose = False

        # Self-calibrated trigger snapshot — set on the first real obstacle,
        # reused for every subsequent obstacle.
        self.calib_blob_area = None
        self.calib_lidar_dist = None

        # Camera-derived
        self.lane_error = 0.0
        self.lane_visible = False
        self.yellow_in_fov = False
        self.yellow_centroid_x = None
        self.yellow_centroid_y_in_roi = None
        self.yellow_bbox = None    # (x, y, w, h) in ROI coords
        self.yellow_area = 0
        self.image_w = 0
        self.image_h = 0
        self.blob_lost_count = 0

        # GUI state
        self._running = False                  # robot motion gated by this
        self._debug_overlay = True
        self._latest_frame = None              # full BGR frame for display
        self._latest_yellow_mask = None
        self._latest_red_mask = None
        self._latest_green_mask = None

        # ==================================================================
        # I/O
        # ==================================================================
        # Sensor topics use BEST_EFFORT QoS. The default `10` would be RELIABLE
        # and would silently fail to connect to the camera publisher — the
        # cause of the "no signal from camera" symptom in the previous version.
        self.create_subscription(LaserScan, g('scan_topic'),
                                 self.cb_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, g('odom_topic'),
                                 self.cb_odom, qos_profile_sensor_data)

        # >>> EDIT: list of camera topics to try. Subscribing to all of them
        # is cheap; whichever the driver actually publishes will deliver
        # messages, the others stay silent. Print of the first frame received
        # will tell you which one fired.
        self._tried_topics = []
        raw_topics = [
            g('image_topic'),
            '/camera/image_raw',
            '/image_raw',
            '/camera/image',
            '/camera/color/image_raw',
        ]
        compressed_topics = [
            '/camera/image_raw/compressed',
            '/image_raw/compressed',
            '/camera/image/compressed',
            '/camera/color/image_raw/compressed',
        ]
        for t in dict.fromkeys(raw_topics):  # de-dup, preserve order
            self.create_subscription(Image, t, self.cb_image_raw,
                                     qos_profile_sensor_data)
            self._tried_topics.append(t)
        for t in dict.fromkeys(compressed_topics):
            self.create_subscription(CompressedImage, t,
                                     self.cb_image_compressed,
                                     qos_profile_sensor_data)
            self._tried_topics.append(t)

        self.cmd_pub = self.create_publisher(Twist, g('cmd_topic'), 10)
        self._got_camera = False
        self._active_camera_topic = None

        self.create_timer(0.05, self.control_loop)         # 20 Hz control
        if self._gui_enabled:
            self._init_gui()
            self.create_timer(0.033, self._gui_spin)       # ~30 Hz GUI

        self.get_logger().info(
            'Challenge 2 node up. State: LINE_FOLLOWING. Press [s] to start.')

    # ============================================================ callbacks
    def cb_scan(self, msg: LaserScan):
        self.latest_scan = msg

    def cb_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.have_odom = True

    def cb_image_raw(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge raw: {e}', throttle_duration_sec=5.0)
            return
        self._note_camera('raw')
        self._process_frame(frame)

    def cb_image_compressed(self, msg: CompressedImage):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().warn(f'cv_bridge compressed: {e}',
                                   throttle_duration_sec=5.0)
            return
        if frame is None:
            return
        self._note_camera('compressed')
        self._process_frame(frame)

    def _note_camera(self, kind):
        if not self._got_camera:
            self._got_camera = True
            self._active_camera_topic = kind
            self.get_logger().info(f'First camera frame received ({kind}).')

    def _process_frame(self, frame):
        h, w, _ = frame.shape
        self.image_w = w
        self.image_h = h
        self._latest_frame = frame

        roi = frame[h // 2:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self._process_yellow(hsv, w)
        self._process_lane(hsv, w)

    # =============================================================== vision
    def _process_yellow(self, hsv, w):
        mask = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        self._latest_yellow_mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # While turning AWAY from the target obstacle, only consider yellow
        # in the half of the image it's being pushed toward — rejects the
        # second obstacle (opposite side per assumption) from confounding
        # the "blob is gone" trigger.
        if self.state == State.AVOID_TURN and self.target_side is not None:
            half = w // 2
            keep = []
            for c in contours:
                M = cv2.moments(c)
                if M['m00'] <= 0:
                    continue
                cx = M['m10'] / M['m00']
                if (self.target_side == 'right' and cx >= half) or \
                   (self.target_side == 'left'  and cx <  half):
                    keep.append(c)
            contours = keep

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 200:
                M = cv2.moments(largest)
                self.yellow_centroid_x = int(M['m10'] / M['m00'])
                self.yellow_centroid_y_in_roi = int(M['m01'] / M['m00'])
                self.yellow_bbox = cv2.boundingRect(largest)
                self.yellow_area = int(area)
                self.yellow_in_fov = True
                self.blob_lost_count = 0
                return

        self.yellow_centroid_x = None
        self.yellow_centroid_y_in_roi = None
        self.yellow_bbox = None
        self.yellow_area = 0
        self.blob_lost_count += 1
        if self.blob_lost_count >= self.blob_lost_n:
            self.yellow_in_fov = False

    def _process_lane(self, hsv, w):
        rmask = cv2.inRange(hsv, self.red_lo,   self.red_hi)
        gmask = cv2.inRange(hsv, self.green_lo, self.green_hi)
        self._latest_red_mask = rmask
        self._latest_green_mask = gmask
        rM = cv2.moments(rmask)
        gM = cv2.moments(gmask)
        if rM['m00'] > 200 and gM['m00'] > 200:
            r_cx = rM['m10'] / rM['m00']
            g_cx = gM['m10'] / gM['m00']
            self.lane_error = 0.5 * (r_cx + g_cx) - (w / 2.0)
            self.lane_visible = True
        else:
            self.lane_visible = False

    # ============================================================= geometry
    @staticmethod
    def _ang_diff(a, b):
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))

    def _min_dist_in_arc(self, center_rad, half_width_rad):
        if self.latest_scan is None:
            return float('inf')
        s = self.latest_scan
        out = float('inf')
        for i, r in enumerate(s.ranges):
            if not math.isfinite(r) or r <= s.range_min:
                continue
            a = s.angle_min + i * s.angle_increment
            if abs(self._ang_diff(a, center_rad)) <= half_width_rad:
                if r < out:
                    out = r
        return out

    def _closest_forward_bearing(self):
        """
        Return (bearing, distance) of the closest lidar return inside the
        forward arc. Used at trigger time to pin the obstacle's position
        in odom frame, since the camera only gives us a side ('left'/'right')
        but the lidar gives us an angle.
        """
        if self.latest_scan is None:
            return None, None
        s = self.latest_scan
        best_d = float('inf')
        best_a = None
        for i, r in enumerate(s.ranges):
            if not math.isfinite(r) or r <= s.range_min:
                continue
            a = s.angle_min + i * s.angle_increment
            if abs(self._ang_diff(a, 0.0)) <= self.fwd_arc:
                if r < best_d:
                    best_d = r
                    best_a = a
        return best_a, best_d

    def _bearing_to_pinned_obstacle(self):
        """
        Bearing (in robot frame, [-π, π]) from the robot's current pose to
        the obstacle position recorded at trigger time. Returns None if no
        obstacle is currently pinned.

        |bearing| <  90°  → obstacle is ahead of us
        |bearing| =  90°  → obstacle is exactly abeam
        |bearing| >  90°  → obstacle is behind us
        """
        if not self.have_obstacle_pose:
            return None
        dx = self.obstacle_odom_x - self.current_x
        dy = self.obstacle_odom_y - self.current_y
        world_bearing = math.atan2(dy, dx)
        return self._ang_diff(world_bearing, self.current_yaw)

    # ============================================================= triggers
    def _should_trigger_avoidance(self):
        if not self.yellow_in_fov or self.yellow_centroid_x is None:
            return None
        area_threshold = (self.calib_blob_area
                          if self.calib_blob_area is not None
                          else self.trigger_area)
        if self.yellow_area < area_threshold:
            return None
        d_fwd = self._min_dist_in_arc(0.0, self.fwd_arc)
        confirm_dist = (self.calib_lidar_dist * 1.3
                        if self.calib_lidar_dist is not None
                        else self.lidar_confirm)
        if d_fwd > confirm_dist:
            return None
        return 'left' if self.yellow_centroid_x < self.image_w / 2 else 'right'

    # =============================================================== motion
    def _publish(self, lin, ang):
        t = Twist()
        t.linear.x = float(lin)
        t.angular.z = float(ang)
        self.cmd_pub.publish(t)

    def _stop(self):
        self._publish(0.0, 0.0)

    def _set_state(self, new_state, reason=''):
        if new_state != self.state:
            self.get_logger().info(
                f'{self.state.name} → {new_state.name}  {reason}'.rstrip())
            self.state = new_state

    # ============================================================ main loop
    def control_loop(self):
        # Robot motion is gated by the running flag (set by GUI).
        if not self._running:
            self._stop()
            return
        if not self.have_odom or self.latest_scan is None:
            return

        if self.state in (State.AVOID_RETURN, State.AVOID_REALIGN):
            side = self._should_trigger_avoidance()
            if side is not None:
                self._begin_avoidance(side, reason='(preempted)')
                return

        dispatch = {
            State.LINE_FOLLOWING: self._do_line_following,
            State.AVOID_TURN:     self._do_avoid_turn,
            State.AVOID_DRIVE:    self._do_avoid_drive,
            State.AVOID_COUNTER:  self._do_avoid_counter,
            State.AVOID_RETURN:   self._do_avoid_return,
            State.AVOID_REALIGN:  self._do_avoid_realign,
        }
        dispatch[self.state]()

    # ================================================================ states
    # >>> EDIT: this is the EMBEDDED, BASIC line follower. Not calling any
    # external file. Replace the body of this method with your own logic if
    # yours is better. The state machine doesn't care how it works as long
    # as it publishes Twist on /cmd_vel and respects the trigger check.
    def _do_line_following(self):
        side = self._should_trigger_avoidance()
        if side is not None:
            self._begin_avoidance(side)
            return
        ang = -self.lane_kp * self.lane_error if self.lane_visible else 0.0
        self._publish(self.v_lin, ang)

    def _begin_avoidance(self, side, reason=''):
        if self.calib_blob_area is None:
            self.calib_blob_area = self.yellow_area
            self.calib_lidar_dist = self._min_dist_in_arc(0.0, self.fwd_arc)
            self.get_logger().info(
                f'Calibrated trigger: blob={self.calib_blob_area} px², '
                f'dist={self.calib_lidar_dist:.2f} m')

        # Pin the obstacle's position in odom frame. From now until the end
        # of the maneuver, "is the obstacle behind us?" is answered by
        # comparing the robot's current pose to this fixed point — not by
        # peeking at lidar side-arcs which can fire on walls / ground / noise.
        bearing, dist = self._closest_forward_bearing()
        if bearing is not None and dist is not None and math.isfinite(dist):
            world_angle = self.current_yaw + bearing
            self.obstacle_odom_x = self.current_x + dist * math.cos(world_angle)
            self.obstacle_odom_y = self.current_y + dist * math.sin(world_angle)
            self.have_obstacle_pose = True
            self.get_logger().info(
                f'Pinned obstacle at odom ({self.obstacle_odom_x:.2f}, '
                f'{self.obstacle_odom_y:.2f}) — '
                f'lidar bearing {math.degrees(bearing):+.0f}°, '
                f'dist {dist:.2f} m')
        else:
            self.have_obstacle_pose = False
            self.get_logger().warn(
                'Could not pin obstacle pose — falling back to side-arc abeam check')

        self.target_side = side
        self.turn_dir = +1 if side == 'right' else -1
        self.phase_start_yaw = self.current_yaw
        self.blob_lost_count = 0
        self._set_state(State.AVOID_TURN,
                        f'(blob on {side}, turning '
                        f'{"left" if self.turn_dir > 0 else "right"}) {reason}')

    def _do_avoid_turn(self):
        self._publish(0.0, self.turn_dir * self.v_ang)
        if not self.yellow_in_fov:
            self.theta = abs(self._ang_diff(self.current_yaw,
                                            self.phase_start_yaw))
            self._set_state(State.AVOID_DRIVE,
                            f'(θ = {math.degrees(self.theta):.1f}°)')

    def _do_avoid_drive(self):
        """
        Drive forward at heading +θ until the obstacle is past abeam in
        WORLD terms. Side-arc lidar checks are unreliable here (they fire
        on the obstacle while it's still ahead-and-to-the-side, not abeam,
        and they fire on walls / ground noise too). Instead, use the
        obstacle's pinned odom-frame position — pure geometry.
        """
        self._publish(self.v_lin, 0.0)

        bearing = self._bearing_to_pinned_obstacle()
        if bearing is None:
            # Fallback: no pose was pinned. Use the old side-arc heuristic.
            side_angle = -math.pi / 2 if self.turn_dir > 0 else +math.pi / 2
            d_side = self._min_dist_in_arc(side_angle, self.side_arc)
            abeam_threshold = (self.calib_lidar_dist * 1.5
                               if self.calib_lidar_dist is not None else 0.6)
            if d_side < abeam_threshold:
                self.phase_start_yaw = self.current_yaw
                self._set_state(State.AVOID_COUNTER,
                                f'(side-arc fallback, d={d_side:.2f} m)')
            return

        # 95° rather than exactly 90° gives us 5° of margin past abeam, so
        # the obstacle is unambiguously behind before we counter-turn.
        if abs(bearing) >= math.radians(95.0):
            self.phase_start_yaw = self.current_yaw
            self._set_state(State.AVOID_COUNTER,
                            f'(obstacle bearing {math.degrees(bearing):+.0f}°)')

    def _do_avoid_counter(self):
        self._publish(0.0, -self.turn_dir * self.v_ang)
        rotated = abs(self._ang_diff(self.current_yaw, self.phase_start_yaw))
        if rotated >= 2.0 * self.theta:
            self._set_state(State.AVOID_RETURN,
                            f'(rotated {math.degrees(rotated):.1f}°)')

    def _do_avoid_return(self):
        self._publish(self.v_lin, 0.0)
        if self.lane_visible and abs(self.lane_error) < self.lane_tol_px:
            self.phase_start_yaw = self.current_yaw
            self._set_state(State.AVOID_REALIGN,
                            f'(lane error = {self.lane_error:.0f} px)')

    def _do_avoid_realign(self):
        self._publish(0.0, self.turn_dir * self.v_ang)
        rotated = abs(self._ang_diff(self.current_yaw, self.phase_start_yaw))
        if rotated >= self.theta:
            self.target_side = None
            self.have_obstacle_pose = False
            self._set_state(State.LINE_FOLLOWING, '(maneuver complete)')

    # ====================================================================
    # GUI — control window, POV with overlay, mask viewer, keyboard
    # ====================================================================
    def _set_running(self, running: bool):
        if running == self._running:
            return
        self._running = running
        if not running:
            self._stop()
        self.get_logger().info('▶ Started' if running else '■ Stopped')

    def _init_gui(self):
        cv2.namedWindow('Challenge2_Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Challenge2_Control', 460, 320)

        # >>> EDIT: tweak ranges here if your scale differs (e.g. higher
        # max linear speed). Trackbars are integer-only so we use scale factors.

        cv2.createTrackbar(
            'lin_speed x100', 'Challenge2_Control',
            int(self.v_lin * 100), 25,
            lambda v: setattr(self, 'v_lin', v / 100.0))

        cv2.createTrackbar(
            'ang_speed x10', 'Challenge2_Control',
            int(self.v_ang * 10), 15,
            lambda v: setattr(self, 'v_ang', v / 10.0))

        cv2.createTrackbar(
            'lane_Kp x1000', 'Challenge2_Control',
            int(self.lane_kp * 1000), 30,
            lambda v: setattr(self, 'lane_kp', v / 1000.0))

        # Trigger area in hundreds of pixels
        cv2.createTrackbar(
            'trigger_area /100', 'Challenge2_Control',
            int(self.trigger_area / 100), 100,
            lambda v: setattr(self, 'trigger_area', max(v, 1) * 100))

        # Lidar confirm distance in cm
        cv2.createTrackbar(
            'lidar_confirm cm', 'Challenge2_Control',
            int(self.lidar_confirm * 100), 250,
            lambda v: setattr(self, 'lidar_confirm', max(v, 10) / 100.0))

        # Yellow hue range — most likely to need lab tuning
        cv2.createTrackbar(
            'yellow_H_lo', 'Challenge2_Control', int(self.yellow_lo[0]), 60,
            lambda v: self.yellow_lo.__setitem__(0, v))
        cv2.createTrackbar(
            'yellow_H_hi', 'Challenge2_Control', int(self.yellow_hi[0]), 60,
            lambda v: self.yellow_hi.__setitem__(0, v))

        # >>> EDIT: add more HSV trackbars here if needed (S_lo, V_lo, etc.)

        cv2.createTrackbar(
            '0=STOP 1=START', 'Challenge2_Control', 0, 1,
            lambda v: self._set_running(v == 1))

    def _handle_keys(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 255 or key == 0xFF:
            return
        if key == ord('s'):
            self._set_running(True)
            cv2.setTrackbarPos('0=STOP 1=START', 'Challenge2_Control', 1)
        elif key == ord('q'):
            self._set_running(False)
            cv2.setTrackbarPos('0=STOP 1=START', 'Challenge2_Control', 0)
        elif key in (ord('+'), ord('=')):
            self.v_lin = min(self.v_lin + 0.01, 0.25)
            cv2.setTrackbarPos('lin_speed x100', 'Challenge2_Control',
                               int(self.v_lin * 100))
        elif key == ord('-'):
            self.v_lin = max(self.v_lin - 0.01, 0.02)
            cv2.setTrackbarPos('lin_speed x100', 'Challenge2_Control',
                               int(self.v_lin * 100))
        elif key == ord('d'):
            self._debug_overlay = not self._debug_overlay
        elif key == ord('r'):
            self.calib_blob_area = None
            self.calib_lidar_dist = None
            self.get_logger().info('Calibration cleared — will re-trigger.')
        elif key == ord('c'):
            self.get_logger().info(
                f'HSV yellow: lo={self.yellow_lo.tolist()} '
                f'hi={self.yellow_hi.tolist()} | '
                f'trigger_area={self.trigger_area} | '
                f'calib_area={self.calib_blob_area} '
                f'calib_dist={self.calib_lidar_dist}')
        elif key == ord('p'):
            d_fwd = self._min_dist_in_arc(0.0, self.fwd_arc)
            self.get_logger().info(
                f'state={self.state.name} lane_err={self.lane_error:.0f} '
                f'yellow_area={self.yellow_area} d_fwd={d_fwd:.2f} '
                f'θ={math.degrees(self.theta):.1f}°')

    def _draw_overlay(self):
        # No image yet — show a waiting screen listing every topic we tried.
        if self._latest_frame is None:
            canvas = np.zeros((max(260, 90 + 18 * len(self._tried_topics)),
                               560, 3), dtype=np.uint8)
            cv2.putText(canvas, 'WAITING FOR CAMERA', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(canvas, 'Topics subscribed:', (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            y = 100
            for t in self._tried_topics:
                cv2.putText(canvas, t, (42, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 200, 255), 1)
                y += 18
            cv2.putText(canvas, '[s] start  [q] stop', (30, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
            cv2.imshow('Challenge2_POV', canvas)
            return

        frame = self._latest_frame.copy()
        h, w, _ = frame.shape

        state_color = STATE_COLORS.get(self.state, (255, 255, 255))

        # --- top banner: state + running flag ---
        cv2.rectangle(frame, (0, 0), (w, 24), (30, 30, 30), -1)
        banner = self.state.name + ('  [RUNNING]' if self._running else '  [STOPPED]')
        cv2.putText(frame, banner, (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)

        # --- yellow blob bbox + crosshair (in full-frame coords) ---
        if self.yellow_bbox is not None:
            x, y, bw, bh = self.yellow_bbox
            y_off = h // 2  # ROI starts at h/2
            cv2.rectangle(frame, (x, y + y_off), (x + bw, y + bh + y_off),
                          (0, 220, 255), 2)
            cx = self.yellow_centroid_x
            cy = (self.yellow_centroid_y_in_roi or 0) + y_off
            cv2.drawMarker(frame, (cx, cy), (0, 220, 255),
                           cv2.MARKER_CROSS, 14, 1)
            cv2.putText(frame, f'{self.yellow_area}px2', (x, y + y_off - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

        # --- metrics block ---
        if self._debug_overlay:
            d_fwd = self._min_dist_in_arc(0.0, self.fwd_arc)
            calib_str = (f'{self.calib_blob_area}px2 / {self.calib_lidar_dist:.2f}m'
                         if self.calib_blob_area is not None else 'not yet')
            trigger_str = (f'{self.calib_blob_area}'
                           if self.calib_blob_area is not None
                           else f'{self.trigger_area}')
            bearing = self._bearing_to_pinned_obstacle()
            bearing_str = (f'{math.degrees(bearing):+.0f}deg'
                           if bearing is not None else 'unpinned')
            lines = [
                f'lane_err={self.lane_error:+.1f}px  yellow={self.yellow_area}px2',
                f'v={self.v_lin:.2f}m/s  w_max={self.v_ang:.2f}rad/s',
                f'd_fwd={d_fwd:.2f}m  theta={math.degrees(self.theta):.1f}deg',
                f'obstacle bearing: {bearing_str}  (trip @ |95|deg)',
                f'calib: {calib_str}',
                f'Kp={self.lane_kp:.4f}  trigger>={trigger_str}px2',
                f'[s]start [q]stop [+/-]speed [d]dbg [r]reset_cal [c]hsv [p]print',
            ]
            y0 = 42
            for i, txt in enumerate(lines):
                cv2.putText(frame, txt, (6, y0 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

        # --- lateral error bar (bottom) ---
        if self.lane_visible:
            bar_cx = w // 2
            bar_y = h - 14
            bar_len = 100
            cv2.line(frame, (bar_cx - bar_len, bar_y),
                            (bar_cx + bar_len, bar_y), (60, 60, 60), 4)
            err_px = int(np.clip(self.lane_error, -bar_len, bar_len))
            color = ((0, 200, 0)   if abs(self.lane_error) < self.lane_tol_px else
                     (0, 165, 255) if abs(self.lane_error) < 60 else
                     (0, 0, 255))
            cv2.circle(frame, (bar_cx + err_px, bar_y), 5, color, -1)
            cv2.putText(frame, 'lane error', (bar_cx - bar_len, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        cv2.imshow('Challenge2_POV', frame)

        # --- combined masks window ---
        if self._latest_yellow_mask is not None:
            my = self._latest_yellow_mask
            mr = self._latest_red_mask
            mg = self._latest_green_mask
            masks = np.zeros((*my.shape, 3), dtype=np.uint8)
            masks[my > 0] = (0, 220, 220)   # yellow blobs → yellow-ish
            if mr is not None:
                masks[mr > 0] = (0, 0, 220)
            if mg is not None:
                masks[mg > 0] = (0, 220, 0)
            cv2.imshow('Challenge2_Masks', masks)

    def _gui_spin(self):
        try:
            self._handle_keys()
            self._draw_overlay()
        except cv2.error as e:
            self.get_logger().warn(f'GUI: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        if node._gui_enabled:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
