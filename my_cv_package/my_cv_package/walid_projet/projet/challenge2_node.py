"""
challenge2_node.py — Challenge 2 : Line Following + Obstacle Avoidance
Pour TurtleBot3 RÉEL — Version ULTRA-ROBUSTE.

Architecture :
  - Timer de contrôle à 10 Hz (clavier marche TOUJOURS, même sans caméra)
  - Auto-souscription à PLUSIEURS topics caméra
  - Évitement LIDAR fort : le LIDAR prend le contrôle total pendant l'obstacle
  - Vision SANS BEV : traitement direct du bas de l'image

Commandes clavier (fenêtre OpenCV) :
  [s] Démarrer  [q] Arrêter  [l/r] Direction rond-point
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2
import time
from cv_bridge import CvBridge

from .vision        import LaneVision
from .fsm           import ChallengeFSM, State
from .controller    import LaneController
from .lidar_handler import LidarHandler


class Challenge2Node(Node):

    def __init__(self):
        super().__init__('challenge2')

        # ── Paramètres ROS2 ──────────────────────────────────────────────
        self.declare_parameter('roundabout_direction', 'right')
        self.declare_parameter('linear_speed',         0.04)
        self.declare_parameter('seuil_hsv',            30)
        self.declare_parameter('avoid_gain',           2.2)
        self.declare_parameter('lidar_front_index',    0)
        self.declare_parameter('camera_topic',         'auto')

        direction  = self.get_parameter('roundabout_direction').value
        lin_speed  = float(self.get_parameter('linear_speed').value)
        seuil_hsv  = int(self.get_parameter('seuil_hsv').value)
        avoid_gain = float(self.get_parameter('avoid_gain').value)
        front_idx  = int(self.get_parameter('lidar_front_index').value)
        cam_topic  = str(self.get_parameter('camera_topic').value)

        # ── Modules ──────────────────────────────────────────────────────
        self.vision = LaneVision(seuil_hsv=seuil_hsv)
        self.fsm    = ChallengeFSM(roundabout_direction=direction, challenge=2)
        self.fsm.OBSTACLE_AVOID_DIST = 0.42
        self.fsm.AVOID_MIN_DURATION = 0.55
        self.fsm.AVOID_MAX_DURATION = 2.4
        self.ctrl   = LaneController(linear_speed=lin_speed)
        self.lidar  = LidarHandler(
            stop_dist=0.25,
            avoid_dist=0.60,
            lidar_front_index=front_idx
        )
        self.bridge = CvBridge()

        # ── Évitement ────────────────────────────────────────────────────
        self._avoid_gain = avoid_gain
        self._avoid_side = 0          # +1 = left, -1 = right
        self._avoid_phase = "idle"    # steer, pass, recover
        self._avoid_phase_time = 0.0
        self._avoid_last_state = False

        # ── État ─────────────────────────────────────────────────────────
        self._last_frame    = None
        self._last_time     = time.time()
        self._last_img_time = 0.0        # 0 = jamais reçu
        self._log_timer     = 0.0
        self._cam_topics_tried = []

        # ── GUI OpenCV ───────────────────────────────────────────────────
        cv2.namedWindow("Challenge2")
        cv2.createTrackbar("Vitesse x100", "Challenge2",
                           int(lin_speed * 100), 20,
                           lambda v: setattr(self.ctrl, 'linear_speed', v / 100.0))
        cv2.createTrackbar("Seuil HSV", "Challenge2", seuil_hsv, 80,
                           lambda s: setattr(self.vision, 'seuil_hsv', s))
        cv2.createTrackbar("Avoid gain x10", "Challenge2",
                           int(avoid_gain * 10), 80,
                           lambda v: setattr(self, '_avoid_gain', v / 10.0))
        cv2.createTrackbar("0=STOP 1=START", "Challenge2", 0, 1,
                           lambda v: self._do_start() if v == 1 else self._do_stop())

        # ── Topics ROS2 ─────────────────────────────────────────────────
        # LIDAR
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # CAMÉRA : essayer plusieurs topics
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        if cam_topic != 'auto':
            # Topic spécifié par l'utilisateur
            self._subscribe_compressed(cam_topic, qos_sensor)
        else:
            # Auto-detection : essayer les topics les plus courants
            topics_compressed = [
                '/camera/image_raw/compressed',
                '/image_raw/compressed',
                '/camera/image/compressed',
                '/camera/color/image_raw/compressed',
            ]
            for topic in topics_compressed:
                self._subscribe_compressed(topic, qos_sensor)

            # Aussi essayer le topic RAW (non compressé)
            topics_raw = [
                '/camera/image_raw',
                '/image_raw',
                '/camera/image',
            ]
            for topic in topics_raw:
                self.create_subscription(
                    Image, topic, self._raw_image_callback, qos_sensor)
                self._cam_topics_tried.append(topic)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Timer principal 10 Hz ────────────────────────────────────────
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            f"Challenge2 PRÊT | v={lin_speed} avoid={avoid_gain} "
            f"front_idx={front_idx}")
        self.get_logger().info(
            f"Topics caméra essayés : {self._cam_topics_tried}")
        self.get_logger().info("[s]=start [q]=stop dans la fenêtre OpenCV")

    def _subscribe_compressed(self, topic, qos):
        self.create_subscription(
            CompressedImage, topic, self._compressed_callback, qos)
        self._cam_topics_tried.append(topic)

    # ─── Contrôles ────────────────────────────────────────────────────────────
    def _do_start(self):
        self.fsm.start()
        self.ctrl.reset()
        self.get_logger().info("▶ Challenge2 START")

    def _do_stop(self):
        self.fsm.stop()
        self.cmd_pub.publish(Twist())
        self.get_logger().info("■ Challenge2 STOP")

    # ─── Callbacks ────────────────────────────────────────────────────────────
    def scan_callback(self, msg):
        self.lidar.update(msg)

    def _compressed_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self._last_frame = frame
            if self._last_img_time == 0.0:
                self.get_logger().info(
                    f"✓ Caméra reçue ! Taille : {frame.shape[1]}x{frame.shape[0]}")
            self._last_img_time = time.time()

    def _raw_image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if frame is not None:
                self._last_frame = frame
                if self._last_img_time == 0.0:
                    self.get_logger().info(
                        f"✓ Caméra RAW reçue ! Taille : {frame.shape[1]}x{frame.shape[0]}")
                self._last_img_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"Erreur conversion image RAW : {e}",
                                   throttle_duration_sec=5.0)

    # ─── Boucle de contrôle (10 Hz) ──────────────────────────────────────────
    def control_loop(self):
        now = time.time()
        dt  = max(now - self._last_time, 0.001)
        dt  = min(dt, 0.2)
        self._last_time = now

        # ── Clavier (marche TOUJOURS, même sans caméra) ──────────────────
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('s'): self._do_start()
        elif key == ord('q'): self._do_stop()
        elif key == ord('l'): self.fsm.set_direction('left')
        elif key == ord('r'): self.fsm.set_direction('right')

        twist = Twist()
        frame = self._last_frame

        # ── Pas de caméra → afficher message d'attente ───────────────────
        if frame is None:
            self._show_waiting_screen()
            self.cmd_pub.publish(twist)
            return

        # ── Vision ───────────────────────────────────────────────────────
        result = self.vision.process(
            frame,
            in_roundabout        = self.fsm.in_roundabout,
            roundabout_direction = self.fsm.roundabout_direction
        )

        error = result['error']
        shift = 0.0

        # ── FSM ──────────────────────────────────────────────────────────
        self.fsm.update(result['curvature'], self.lidar.obstacle_dist, dt)

        if self.fsm.is_avoiding and not self._avoid_last_state:
            self._start_avoidance()
        elif not self.fsm.is_avoiding and self._avoid_last_state:
            self._avoid_side = 0
            self._avoid_phase = "idle"
            self._avoid_phase_time = 0.0
        self._avoid_last_state = self.fsm.is_avoiding

        # ── Commande ─────────────────────────────────────────────────────
        if self.fsm.is_running:
            linear, angular = self.ctrl.compute(
                error, result['curvature'], dt,
                in_roundabout=self.fsm.in_roundabout
            )

            # ── ÉVITEMENT LIDAR ──────────────────────────────────────────
            if self.fsm.is_avoiding:
                linear, angular, shift = self._compute_avoidance(
                    dt, result, linear, angular)

                self.get_logger().info(
                    f"AVOID {self._avoid_phase} | side={self._avoid_side:+d} "
                    f"F={self.lidar.front_min_dist:.2f} "
                    f"FL={self.lidar.front_left_min_dist:.2f} "
                    f"FR={self.lidar.front_right_min_dist:.2f} "
                    f"v={linear:.2f} w={angular:+.2f}",
                    throttle_duration_sec=0.3)

            twist.linear.x  = linear
            twist.angular.z = float(np.clip(angular, -1.2, 1.2))

        self.cmd_pub.publish(twist)

        # ── Log ──────────────────────────────────────────────────────────
        self._log_timer += dt
        if self._log_timer > 1.0:
            self._log_timer = 0.0
            self.get_logger().info(
                f"[{self.fsm.state.name:>15s}] "
                f"err={error:+6.1f} shift={shift:+.2f} "
                f"F={self.lidar.front_dist:.2f}m "
                f"v={twist.linear.x:.3f} w={twist.angular.z:+.3f}")

        # ── Debug visuel ─────────────────────────────────────────────────
        self._show_debug(result, error, shift, twist)

    def _start_avoidance(self):
        left_clear = min(self.lidar.left_dist, self.lidar.front_left_dist)
        right_clear = min(self.lidar.right_dist, self.lidar.front_right_dist)
        self._avoid_side = 1 if left_clear >= right_clear else -1
        self._avoid_phase = "steer"
        self._avoid_phase_time = 0.0
        self.ctrl.reset()
        self.get_logger().warn(
            f"AVOID START side={self._avoid_side:+d} "
            f"L={left_clear:.2f} R={right_clear:.2f}")

    def _compute_avoidance(self, dt, result, lane_linear, lane_angular):
        self._avoid_phase_time += dt

        lane_error = result['error']
        front = min(self.lidar.front_dist, self.lidar.front_min_dist)
        fl = min(self.lidar.front_left_dist, self.lidar.front_left_min_dist)
        fr = min(self.lidar.front_right_dist, self.lidar.front_right_min_dist)
        lane_width = float(result.get('lane_width', 0.0))
        lane_left = float(result.get('lane_left_x', 0.0))
        lane_right = float(result.get('lane_right_x', 0.0))
        image_center = float(result.get('image_width', 640)) * 0.5
        both_lines = bool(result.get('green_found', False) and
                          result.get('red_found', False) and
                          lane_width > 90.0)

        if self._avoid_side == 0:
            self._start_avoidance()

        # Emergency: do not push into the bottle. Rotate slowly, then let the
        # normal lane controller pull the robot back between the lines.
        if front < 0.20:
            return 0.0, 0.22 * self._avoid_side, 0.0

        if self._avoid_phase == "steer":
            if self._avoid_phase_time > 0.35 or front > 0.44:
                self._avoid_phase = "pass"
                self._avoid_phase_time = 0.0
            linear = min(lane_linear, 0.032)
            shift_limit = 0.16

        elif self._avoid_phase == "pass":
            blocked_on_side = fl if self._avoid_side > 0 else fr
            if self._avoid_phase_time > 0.35 and front > 0.50 and blocked_on_side > 0.32:
                self._avoid_phase = "recover"
                self._avoid_phase_time = 0.0
            linear = min(lane_linear, 0.035)
            shift_limit = 0.10

        else:
            linear = min(lane_linear, 0.035)
            shift_limit = -0.10

        line_pull = float(np.clip(lane_angular, -0.42, 0.42))

        if both_lines:
            desired_shift = -self._avoid_side * min(shift_limit * lane_width, 34.0)
            desired_target = image_center + lane_error + desired_shift

            # Keep the target in the safe central band of the red/green lane.
            safe_left = lane_left + 0.36 * lane_width
            safe_right = lane_right - 0.36 * lane_width
            desired_target = float(np.clip(desired_target, safe_left, safe_right))
            guarded_error = desired_target - image_center

            # Convert the bounded pixel translation into a small angular bias.
            # Negative pixel delta means target moves left, so angular must grow.
            pixel_delta = guarded_error - lane_error
            angular = line_pull - 0.0040 * pixel_delta
            shift = pixel_delta
        else:
            # Without both lane borders, obstacle avoidance is not allowed to
            # invent a large detour. Stay slow and let line following recover.
            linear = min(linear, 0.025)
            angular = line_pull
            shift = 0.0

        # Hard lane guard: if vision says we are already far from center,
        # suppress the obstacle bias and prioritize returning to the lane.
        if abs(lane_error) > 95:
            angular = line_pull
            linear = min(linear, 0.03)
            shift = 0.0

        angular = float(np.clip(angular, -0.48, 0.48))
        return linear, angular, shift

    # ─── Écran d'attente caméra ───────────────────────────────────────────────
    def _show_waiting_screen(self):
        canvas = np.zeros((300, 450, 3), dtype=np.uint8)
        cv2.putText(canvas, "EN ATTENTE DE CAMERA...", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(canvas, "Topics essayes :", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y = 145
        for t in self._cam_topics_tried[:6]:
            cv2.putText(canvas, t, (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
            y += 18
        cv2.putText(canvas, "Verifier : ros2 topic list | grep image", (30, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1)

        # LIDAR status
        cv2.putText(canvas, f"LIDAR: F={self.lidar.front_dist:.2f}m "
                    f"L={self.lidar.left_dist:.2f} R={self.lidar.right_dist:.2f}",
                    (30, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)

        status = "RUNNING" if self.fsm.is_running else "STOPPED"
        cv2.putText(canvas, f"FSM: {status}  [s]=start [q]=stop",
                    (30, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 220, 0) if self.fsm.is_running else (0, 0, 200), 1)

        cv2.imshow("Challenge2", canvas)

    # ─── Debug visuel ─────────────────────────────────────────────────────────
    def _show_debug(self, result, error, shift, twist):
        debug = result['debug_bev'].copy()
        h, w = debug.shape[:2]

        state_color = {
            State.LINE_FOLLOWING: (0, 220, 0),
            State.ROUNDABOUT:     (0, 165, 255),
            State.OBSTACLE_STOP:  (0, 0, 255),
            State.OBSTACLE_AVOID: (0, 200, 200),
        }.get(self.fsm.state, (128, 128, 128))

        # Barre d'état en haut
        cv2.rectangle(debug, (0, 0), (w, 22), (30, 30, 30), -1)
        cv2.putText(debug, f"{self.fsm.state.name}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)

        # LIDAR info en bas
        cv2.rectangle(debug, (0, h - 40), (w, h), (20, 20, 20), -1)
        cv2.putText(debug,
                    f"F={self.lidar.front_dist:.2f} "
                    f"L={self.lidar.left_dist:.2f} "
                    f"R={self.lidar.right_dist:.2f}",
                    (4, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
        cv2.putText(debug,
                    f"v={twist.linear.x:.3f} w={twist.angular.z:+.3f} "
                    f"shift={shift:+.2f}",
                    (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 220, 180), 1)

        # Indicateur AVOID
        if self.fsm.is_avoiding:
            cv2.rectangle(debug, (w - 100, 2), (w - 2, 20), (0, 200, 200), -1)
            cv2.putText(debug, "AVOIDING", (w - 96, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        cv2.imshow("Challenge2", debug)


def main(args=None):
    rclpy.init(args=args)
    node = Challenge2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
