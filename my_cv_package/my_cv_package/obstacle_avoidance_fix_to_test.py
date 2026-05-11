#!/usr/bin/env python3
"""
============================================================================
 Challenge 2 — Line following + yellow obstacle avoidance  (lidar-primary)
============================================================================

 ARCHITECTURE
 ------------
 Lidar is the PRIMARY trigger. Camera is confirmation + sidedness.
 Fixed thresholds throughout — no auto-calibration cascade.

 Trigger rule:
   lidar forward distance < TRIGGER_DIST           (e.g. 0.5 m)
   AND
   camera sees a yellow blob with area > YELLOW_MIN_AREA   (noise floor)
   → begin S-curve avoidance, dodge to the side opposite the blob centroid.

 State machine
 -------------
   LINE_FOLLOWING --[trigger]----------------------> AVOID_TURN
   AVOID_TURN     --[target blob exits FOV]-------> AVOID_DRIVE   (record θ)
   AVOID_DRIVE    --[obstacle past abeam in odom]-> AVOID_COUNTER
   AVOID_COUNTER  --[rotated 2·θ opposite]--------> AVOID_RETURN
   AVOID_RETURN   --[camera: lane centered]-------> AVOID_REALIGN
   AVOID_REALIGN  --[rotated θ back to forward]--> LINE_FOLLOWING

 KEY DESIGN POINTS
 -----------------
 1. AVOID_DRIVE uses the obstacle's ODOM-FRAME position (pinned at trigger
    time from lidar bearing+distance) to decide when it's truly behind us.
    Pure geometry — no fragile side-arc heuristics.

 2. Preemption: a fresh qualifying trigger during AVOID_RETURN/REALIGN
    restarts the maneuver from AVOID_TURN. Same trigger conditions, so the
    second obstacle is treated identically to the first.

 3. Embedded line follower is intentionally minimal — your dedicated line
    follower node will outperform it. The embedded one exists so this node
    is self-contained and testable in isolation.

 INSTALLATION
 ------------
   1. Place at: ~/ros2_ws/src/projet/projet/obstacle_avoidance.py
      (rename `projet` to your package name as needed).
   2. setup.py entry point:
        'obstacle_avoidance = projet.obstacle_avoidance:main'
   3. colcon build --packages-select projet
   4. ros2 run projet obstacle_avoidance
   5. Headless?  --ros-args -p enable_gui:=false
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


STATE_COLORS = {                       # BGR for the POV banner
    State.LINE_FOLLOWING: (0,   200, 0),
    State.AVOID_TURN:     (0,   165, 255),
    State.AVOID_DRIVE:    (0,   220, 220),
    State.AVOID_COUNTER:  (60,   80, 255),
    State.AVOID_RETURN:   (220, 220, 0),
    State.AVOID_REALIGN:  (220, 0,   220),
}


class ObstacleAvoidanceNode(Node):

    def __init__(self):
        # >>> EDIT: node name (visible in `ros2 node list`)
        super().__init__('obstacle_avoidance')

        # ==================================================================
        # PARAMETERS
        # ==================================================================
        # >>> TUNE: motion speeds
        self.declare_parameter('linear_speed', 0.12)
        self.declare_parameter('angular_speed', 0.5)

        # >>> TUNE (PRIMARY): lidar forward distance below which we trigger.
        # Single most important parameter. Fixed value — no auto-calibration.
        self.declare_parameter('trigger_distance', 0.50)        # metres

        # Noise floor on yellow blob area — just rejects HSV speckle, NOT a
        # primary trigger. Keep small. Larger blobs always pass.
        self.declare_parameter('yellow_min_area', 800)          # px²

        # Arcs (degrees, half-width) for forward / side lidar checks
        self.declare_parameter('forward_arc_deg', 25.0)
        self.declare_parameter('side_arc_deg', 25.0)

        # >>> TUNE: line follower P-gain and tolerance
        self.declare_parameter('lane_kp', 0.004)
        self.declare_parameter('lane_centered_px', 30)

        self.declare_parameter('blob_lost_frames', 3)           # debounce

        # >>> EDIT: topic names — verify with `ros2 topic list`
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_topic', '/cmd_vel')

        # >>> TUNE on real robot — lab lighting won't match Gazebo
        self.declare_parameter('yellow_hsv_lower', [20, 100, 80])
        self.declare_parameter('yellow_hsv_upper', [35, 255, 255])
        self.declare_parameter('red_hsv_lower',  [0, 100, 80])
        self.declare_parameter('red_hsv_upper',  [10, 255, 255])
        self.declare_parameter('green_hsv_lower', [40, 80, 60])
        self.declare_parameter('green_hsv_upper', [85, 255, 255])

        # >>> EDIT: set false on headless / SSH-without-X
        self.declare_parameter('enable_gui', True)

        g = lambda n: self.get_parameter(n).value
        self.v_lin = g('linear_speed')
        self.v_ang = g('angular_speed')
        self.trigger_dist = g('trigger_distance')
        self.yellow_min_area = g('yellow_min_area')
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

        # Obstacle pinned in odom frame at trigger time, used by AVOID_DRIVE.
        self.obstacle_odom_x = 0.0
        self.obstacle_odom_y = 0.0
        self.have_obstacle_pose = False

        # Camera-derived
        self.lane_error = 0.0
        self.lane_visible = False
        self.yellow_in_fov = False
        self.yellow_centroid_x = None
        self.yellow_centroid_y_in_roi = None
        self.yellow_bbox = None
        self.yellow_area = 0
        self.image_w = 0
        self.image_h = 0
        self.blob_lost_count = 0

        # GUI / runtime gating
        self._running = False
        self._debug_overlay = True
        self._latest_frame = None
        self._latest_yellow_mask = None
        self._latest_red_mask = None
        self._latest_green_mask = None
        self._last_v = 0.0
        self._last_w = 0.0
        self._got_camera = False
        self._tried_topics: list[str] = []

        # ==================================================================
        # I/O — all sensor subs use sensor_data QoS (BEST_EFFORT)
        # ==================================================================
        self.create_subscription(LaserScan, g('scan_topic'),
                                 self.cb_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, g('odom_topic'),
                                 self.cb_odom, qos_profile_sensor_data)

        # Cover common camera topic variations. Whichever the driver actually
        # publishes will deliver frames; the rest stay silent.
        raw_topics = dict.fromkeys([
            g('image_topic'), '/camera/image_raw', '/image_raw',
            '/camera/image', '/camera/color/image_raw',
        ])
        compressed_topics = dict.fromkeys([
            '/camera/image_raw/compressed', '/image_raw/compressed',
            '/camera/image/compressed', '/camera/color/image_raw/compressed',
        ])
        for t in raw_topics:
            self.create_subscription(Image, t, self.cb_image_raw,
                                     qos_profile_sensor_data)
            self._tried_topics.append(t)
        for t in compressed_topics:
            self.create_subscription(CompressedImage, t,
                                     self.cb_image_compressed,
                                     qos_profile_sensor_data)
            self._tried_topics.append(t)

        self.cmd_pub = self.create_publisher(Twist, g('cmd_topic'), 10)

        self.create_timer(0.05, self.control_loop)             # 20 Hz
        if self._gui_enabled:
            self._init_gui()
            self.create_timer(0.033, self._gui_spin)           # ~30 Hz

        self.get_logger().info(
            f'Challenge 2 node up. State: LINE_FOLLOWING. '
            f'Press [s] to start.  trigger_dist={self.trigger_dist:.2f} m')

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

        # While turning AWAY from the target obstacle, only count yellow
        # in the half of the image we're pushing it toward — rejects the
        # second obstacle (opposite side) from extending the trigger.
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
            if area > 200:                       # absolute noise floor (pre-trigger)
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
        """(bearing, distance) of the closest lidar return in the forward arc."""
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
        Bearing in robot frame to the obstacle pinned at trigger time.
        |bearing| < 90° → ahead.  |bearing| > 90° → behind.
        """
        if not self.have_obstacle_pose:
            return None
        dx = self.obstacle_odom_x - self.current_x
        dy = self.obstacle_odom_y - self.current_y
        return self._ang_diff(math.atan2(dy, dx), self.current_yaw)

    # ============================================================== triggers
    def _should_trigger_avoidance(self):
        """
        Lidar-primary. Camera confirms colour and tells us which side.

        Returns 'left' / 'right' if conditions met, else None.
        """
        # Gate 1 — lidar says something is genuinely close ahead.
        d_fwd = self._min_dist_in_arc(0.0, self.fwd_arc)
        if d_fwd >= self.trigger_dist:
            return None

        # Gate 2 — camera confirms it's yellow and big enough not to be noise.
        if not self.yellow_in_fov or self.yellow_centroid_x is None:
            return None
        if self.yellow_area < self.yellow_min_area:
            return None

        return 'left' if self.yellow_centroid_x < self.image_w / 2 else 'right'

    # ============================================================= motion
    def _publish(self, lin, ang):
        t = Twist()
        t.linear.x = float(lin)
        t.angular.z = float(ang)
        self._last_v = float(lin)
        self._last_w = float(ang)
        self.cmd_pub.publish(t)

    def _stop(self):
        self._publish(0.0, 0.0)

    def _shaped_linear_speed(self, ang_cmd):
        """
        Optional curvature- and proximity-based speed reduction.
        DISABLED by default (both floors at 1.0). Drop the floors below 1.0
        to enable.  See conversation notes for rationale.
        """
        CURVE_FLOOR = 1.0       # 1.0 disables curvature slowdown
        PROX_FLOOR  = 1.0       # 1.0 disables proximity slowdown
        SLOW_FROM   = 1.0
        speed = self.v_lin
        if CURVE_FLOOR < 1.0 and self.v_ang > 0:
            abs_ang = abs(ang_cmd)
            if abs_ang > 0.05:
                ratio = min(abs_ang / self.v_ang, 1.0)
                speed *= 1.0 - (1.0 - CURVE_FLOOR) * ratio
        if PROX_FLOOR < 1.0:
            d_fwd = self._min_dist_in_arc(0.0, self.fwd_arc)
            slow_at = self.trigger_dist * SLOW_FROM
            if math.isfinite(d_fwd) and d_fwd < slow_at:
                ratio = max(0.0, d_fwd / slow_at)
                speed *= PROX_FLOOR + (1.0 - PROX_FLOOR) * ratio
        return speed

    def _set_state(self, new_state, reason=''):
        if new_state != self.state:
            self.get_logger().info(
                f'{self.state.name} → {new_state.name}  {reason}'.rstrip())
            self.state = new_state

    # ============================================================ main loop
    def control_loop(self):
        if not self._running:
            self._stop()
            return
        if not self.have_odom or self.latest_scan is None:
            return

        # Preemption: a fresh qualifying trigger during recovery restarts.
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

    # ============================================================ states

    # >>> EDIT: minimal embedded line follower. Your dedicated line-follower
    # node will likely outperform this. Replace the body of this method
    # with calls into yours, or run two nodes and mux /cmd_vel.
    def _do_line_following(self):
        side = self._should_trigger_avoidance()
        if side is not None:
            self._begin_avoidance(side)
            return
        ang = -self.lane_kp * self.lane_error if self.lane_visible else 0.0
        lin = self._shaped_linear_speed(ang)
        self._publish(lin, ang)

    def _begin_avoidance(self, side, reason=''):
        # Pin the obstacle's odom-frame position from the lidar bearing+dist
        # we have RIGHT NOW. Used by AVOID_DRIVE to decide when it's behind.
        bearing, dist = self._closest_forward_bearing()
        if bearing is not None and dist is not None and math.isfinite(dist):
            world_angle = self.current_yaw + bearing
            self.obstacle_odom_x = self.current_x + dist * math.cos(world_angle)
            self.obstacle_odom_y = self.current_y + dist * math.sin(world_angle)
            self.have_obstacle_pose = True
            self.get_logger().info(
                f'Pinned obstacle at ({self.obstacle_odom_x:.2f}, '
                f'{self.obstacle_odom_y:.2f}) — '
                f'lidar bearing {math.degrees(bearing):+.0f}°, '
                f'dist {dist:.2f} m')
        else:
            self.have_obstacle_pose = False
            self.get_logger().warn(
                'No lidar return at trigger — AVOID_DRIVE will use side-arc fallback')

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
        WORLD terms (pinned odom position + current robot pose). Lidar
        side-arcs are unreliable here — they fire on walls, ground, and
        on the obstacle while it's still ahead-and-to-the-side.
        """
        self._publish(self.v_lin, 0.0)

        bearing = self._bearing_to_pinned_obstacle()
        if bearing is None:
            # Fallback: pose wasn't pinned. Use side-arc heuristic.
            side_angle = -math.pi / 2 if self.turn_dir > 0 else +math.pi / 2
            d_side = self._min_dist_in_arc(side_angle, self.side_arc)
            if d_side < self.trigger_dist * 1.5:
                self.phase_start_yaw = self.current_yaw
                self._set_state(State.AVOID_COUNTER,
                                f'(side-arc fallback d={d_side:.2f} m)')
            return

        # 95° gives 5° of margin past true abeam — obstacle unambiguously behind.
        if abs(bearing) >= math.radians(95.0):
            self.phase_start_yaw = self.current_yaw
            self._set_state(State.AVOID_COUNTER,
                            f'(bearing {math.degrees(bearing):+.0f}°)')

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

    # ======================================================================
    # GUI
    # ======================================================================
    def _set_running(self, running: bool):
        if running == self._running:
            return
        self._running = running
        if not running:
            self._stop()
        self.get_logger().info('▶ Started' if running else '■ Stopped')

    def _force_idle(self):
        """Hard-reset the FSM back to LINE_FOLLOWING — useful if it gets stuck."""
        self.have_obstacle_pose = False
        self.target_side = None
        self.theta = 0.0
        self._set_state(State.LINE_FOLLOWING, '(force reset)')

    def _init_gui(self):
        cv2.namedWindow('Challenge2_Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Challenge2_Control', 460, 320)

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

        # PRIMARY trigger threshold, in cm for trackbar resolution
        cv2.createTrackbar(
            'trigger_dist cm', 'Challenge2_Control',
            int(self.trigger_dist * 100), 200,
            lambda v: setattr(self, 'trigger_dist', max(v, 10) / 100.0))

        # Noise floor on blob area, in hundreds of pixels
        cv2.createTrackbar(
            'yellow_min /100', 'Challenge2_Control',
            int(self.yellow_min_area / 100), 50,
            lambda v: setattr(self, 'yellow_min_area', max(v, 1) * 100))

        cv2.createTrackbar(
            'yellow_H_lo', 'Challenge2_Control', int(self.yellow_lo[0]), 60,
            lambda v: self.yellow_lo.__setitem__(0, v))
        cv2.createTrackbar(
            'yellow_H_hi', 'Challenge2_Control', int(self.yellow_hi[0]), 60,
            lambda v: self.yellow_hi.__setitem__(0, v))

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
        elif key == ord('r'):                  # repurposed: force IDLE
            self._force_idle()
        elif key == ord('c'):
            self.get_logger().info(
                f'HSV yellow: lo={self.yellow_lo.tolist()} '
                f'hi={self.yellow_hi.tolist()} | '
                f'trigger_dist={self.trigger_dist:.2f} m | '
                f'yellow_min={self.yellow_min_area} px²')
        elif key == ord('p'):
            d_fwd = self._min_dist_in_arc(0.0, self.fwd_arc)
            bearing = self._bearing_to_pinned_obstacle()
            bs = f'{math.degrees(bearing):+.0f}°' if bearing else 'none'
            self.get_logger().info(
                f'state={self.state.name} lane_err={self.lane_error:.0f} '
                f'yellow_area={self.yellow_area} d_fwd={d_fwd:.2f} '
                f'bearing={bs} θ={math.degrees(self.theta):.1f}°')

    def _draw_overlay(self):
        # Waiting screen — also lists every topic subscribed
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
            cv2.putText(canvas, '[s] start  [q] stop  [r] reset',
                        (30, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
            cv2.imshow('Challenge2_POV', canvas)
            return

        frame = self._latest_frame.copy()
        h, w, _ = frame.shape
        state_color = STATE_COLORS.get(self.state, (255, 255, 255))

        # Top banner
        cv2.rectangle(frame, (0, 0), (w, 24), (30, 30, 30), -1)
        banner = self.state.name + ('  [RUNNING]' if self._running else '  [STOPPED]')
        cv2.putText(frame, banner, (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)

        # Yellow blob bbox + crosshair (translate ROI y to full-frame y)
        if self.yellow_bbox is not None:
            x, y, bw, bh = self.yellow_bbox
            y_off = h // 2
            cv2.rectangle(frame, (x, y + y_off), (x + bw, y + bh + y_off),
                          (0, 220, 255), 2)
            cx = self.yellow_centroid_x
            cy = (self.yellow_centroid_y_in_roi or 0) + y_off
            cv2.drawMarker(frame, (cx, cy), (0, 220, 255),
                           cv2.MARKER_CROSS, 14, 1)
            cv2.putText(frame, f'{self.yellow_area}px2', (x, y + y_off - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

        # Metrics block
        if self._debug_overlay:
            d_fwd = self._min_dist_in_arc(0.0, self.fwd_arc)
            bearing = self._bearing_to_pinned_obstacle()
            bearing_str = (f'{math.degrees(bearing):+.0f}deg'
                           if bearing is not None else 'unpinned')
            # Visual cue: does the current lidar/yellow pair satisfy the trigger?
            armed = d_fwd < self.trigger_dist
            yellow_ok = self.yellow_in_fov and self.yellow_area >= self.yellow_min_area
            gate_str = (f'lidar:{"OK" if armed else "--"}  '
                        f'yellow:{"OK" if yellow_ok else "--"}')
            lines = [
                f'lane_err={self.lane_error:+.1f}px  yellow={self.yellow_area}px2',
                f'cmd v={self._last_v:.2f}m/s  w={self._last_w:+.2f}rad/s '
                f'(v_max={self.v_lin:.2f})',
                f'd_fwd={d_fwd:.2f}m  trigger@<{self.trigger_dist:.2f}m',
                f'gates: {gate_str}',
                f'theta={math.degrees(self.theta):.1f}deg  '
                f'obstacle bearing: {bearing_str}',
                f'[s]start [q]stop [+/-]speed [d]dbg [r]reset [c]hsv [p]print',
            ]
            y0 = 42
            for i, txt in enumerate(lines):
                cv2.putText(frame, txt, (6, y0 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

        # Lateral error bar
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

        cv2.imshow('Challenge2_POV', frame)

        if self._latest_yellow_mask is not None:
            my = self._latest_yellow_mask
            mr = self._latest_red_mask
            mg = self._latest_green_mask
            masks = np.zeros((*my.shape, 3), dtype=np.uint8)
            masks[my > 0] = (0, 220, 220)
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
