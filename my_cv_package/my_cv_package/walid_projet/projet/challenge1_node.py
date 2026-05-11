"""
challenge1_node.py — Challenge 1 : Line Following + Roundabout + Obstacle Stop
Architecture : LaneVision (BEV+poly) + ChallengeFSM + PD controller + LIDAR

Commandes clavier dans la fenêtre OpenCV :
  [s] Démarrer
  [q] Arrêter
  [l] Direction rond-point = LEFT
  [r] Direction rond-point = RIGHT
  [+] Augmenter vitesse
  [-] Diminuer vitesse
  [c] Calibration BEV (affiche les points src)
  [d] Toggle debug détaillé
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import cv2
import time
from cv_bridge import CvBridge

# from .vision        import LaneVision
# from .fsm           import ChallengeFSM, State
# from .controller    import LaneController
# from .lidar_handler import LidarHandler
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision        import LaneVision
from fsm           import ChallengeFSM, State
from controller    import LaneController
from lidar_handler import LidarHandler

# ─── Couleurs de l'interface ──────────────────────────────────────────────────
COLOR = {
    State.IDLE:           (128, 128, 128),
    State.LINE_FOLLOWING: (0,   220,   0),
    State.ROUNDABOUT:     (0,   165, 255),
    State.OBSTACLE_STOP:  (0,     0, 255),
    State.OBSTACLE_AVOID: (0,   200, 200),
}


class Challenge1Node(Node):

    def __init__(self):
        super().__init__('challenge1')

        # ── Paramètres ROS ────────────────────────────────────────────────
        self.declare_parameter('roundabout_direction', 'right')
        self.declare_parameter('linear_speed',         0.08)
        self.declare_parameter('kp',                   0.005)
        self.declare_parameter('kd',                   0.002)
        self.declare_parameter('seuil_hsv',            30)
        self.declare_parameter('obstacle_dist',        0.35)
        self.declare_parameter('roundabout_duration',  5.0)
        self.declare_parameter('roundabout_min_time',  5.0)
        self.declare_parameter('challenge',            1)

        direction   = self.get_parameter('roundabout_direction').value
        lin_speed   = self.get_parameter('linear_speed').value
        kp          = self.get_parameter('kp').value
        kd          = self.get_parameter('kd').value
        seuil_hsv   = self.get_parameter('seuil_hsv').value
        obs_dist    = self.get_parameter('obstacle_dist').value
        ra_dur      = self.get_parameter('roundabout_duration').value
        ra_min_time = float(self.get_parameter('roundabout_min_time').value)
        challenge   = self.get_parameter('challenge').value

        self.get_logger().info(
            f"Challenge {challenge} | direction={direction} | "
            f"v={lin_speed} Kp={kp} Kd={kd} seuil={seuil_hsv}"
        )

        # ── Modules ───────────────────────────────────────────────────────
        self.vision  = LaneVision(seuil_hsv=seuil_hsv)
        self.fsm     = ChallengeFSM(roundabout_direction=direction, challenge=challenge)
        self.fsm.OBSTACLE_STOP_DIST  = obs_dist
        self.fsm.ROUNDABOUT_DURATION = ra_dur
        self.fsm.CURVATURE_THRESHOLD = 0.006
        self.fsm.CURVATURE_CONFIRM_N = 28

        self.ctrl  = LaneController(kp=kp, kd=kd, linear_speed=lin_speed)
        self.lidar = LidarHandler(stop_dist=obs_dist)
        self.bridge = CvBridge()

        # ── Timing ────────────────────────────────────────────────────────
        self._last_time = time.time()
        self._debug_mode = True
        self._run_start_time = None
        self._roundabout_min_time = ra_min_time
        self._got_camera = False
        self._camera_topics = []

        # ── Fenêtre de contrôle OpenCV ────────────────────────────────────
        self._init_gui()

        # ── ROS2 topics ───────────────────────────────────────────────────
        for topic in [
            '/camera/image_raw/compressed',
            '/image_raw/compressed',
            '/camera/image/compressed',
            '/camera/color/image_raw/compressed',
        ]:
            self.create_subscription(
                CompressedImage, topic,
                self.image_callback, qos_profile_sensor_data)
            self._camera_topics.append(topic)

        for topic in [
            '/camera/image_raw',
            '/image_raw',
            '/camera/image',
        ]:
            self.create_subscription(
                Image, topic,
                self.raw_image_callback, qos_profile_sensor_data)
            self._camera_topics.append(topic)

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan',
            self.scan_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer de sécurité : si pas d'image depuis 2s, stopper le robot
        self._last_img_time = time.time()
        self._safety_timer  = self.create_timer(0.5, self._safety_check)
        self._gui_timer     = self.create_timer(0.05, self._gui_spin)

        self.get_logger().info(
            "Challenge1 prêt │ [s]=start [q]=stop [l/r]=direction [+/-]=vitesse"
        )
        self.get_logger().info(f"Topics caméra essayés: {self._camera_topics}")

    # ─── GUI ──────────────────────────────────────────────────────────────────
    def _init_gui(self):
        cv2.namedWindow("Challenge1_Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Challenge1_Control", 420, 220)

        cv2.createTrackbar("Vitesse×100", "Challenge1_Control",
                           int(self.ctrl.linear_speed * 100), 25,
                           lambda v: setattr(self.ctrl, 'linear_speed', v / 100.0))

        cv2.createTrackbar("Kp×1000", "Challenge1_Control",
                           int(self.ctrl.kp * 1000), 30,
                           lambda v: setattr(self.ctrl, 'kp', v / 1000.0))

        cv2.createTrackbar("Kd×1000", "Challenge1_Control",
                           int(self.ctrl.kd * 1000), 20,
                           lambda v: setattr(self.ctrl, 'kd', v / 1000.0))

        cv2.createTrackbar("Seuil HSV", "Challenge1_Control",
                           self.vision.seuil_hsv, 80,
                           lambda s: setattr(self.vision, 'seuil_hsv', s))

        cv2.createTrackbar("Durée RA×10", "Challenge1_Control",
                           int(self.fsm.ROUNDABOUT_DURATION * 10), 100,
                           lambda v: setattr(self.fsm, 'ROUNDABOUT_DURATION', v / 10.0))

        cv2.createTrackbar("Seuil curv×10K", "Challenge1_Control",
                           int(self.fsm.CURVATURE_THRESHOLD * 10000), 100,
                           lambda v: setattr(self.fsm, 'CURVATURE_THRESHOLD', v / 10000.0))
        cv2.createTrackbar("0=STOP 1=START", "Challenge1_Control", 0, 1,
                           lambda v: self._start_robot() if v == 1 else self._stop_robot())

    def _start_robot(self):
        self.fsm.start()
        self.ctrl.reset()
        self.vision.reset_memory()
        self._run_start_time = time.time()

    def _stop_robot(self):
        self.fsm.stop()
        self._run_start_time = None

    # ─── Callbacks ────────────────────────────────────────────────────────────
    def scan_callback(self, msg):
        self.lidar.update(msg)

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self._process_frame(frame)

    def raw_image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(
                f"Erreur conversion image RAW: {exc}",
                throttle_duration_sec=5.0)
            return
        self._process_frame(frame)

    def _process_frame(self, frame):
        self._last_img_time = time.time()
        if not self._got_camera:
            self._got_camera = True
            self.get_logger().info(
                f"Caméra reçue: {frame.shape[1]}x{frame.shape[0]}")

        # Delta time
        now = time.time()
        dt  = max(now - self._last_time, 0.001)
        dt  = min(dt, 0.1)   # évite les grands sauts
        self._last_time = now

        # ── Traitement vision ──────────────────────────────────────────────
        result = self.vision.process(
            frame,
            in_roundabout        = self.fsm.in_roundabout,
            roundabout_direction = self.fsm.roundabout_direction
        )

        error     = result['error'] + self.lidar.lateral_shift
        curvature = result['curvature']

        # ── Mise à jour FSM ───────────────────────────────────────────────
        # The camera sees the curve before the robot physically reaches it.
        # Arm roundabout detection only after a minimum run time, otherwise C1
        # enters ROUNDABOUT too early and turns before the real bend.
        roundabout_armed = (
            self._run_start_time is not None and
            time.time() - self._run_start_time >= self._roundabout_min_time
        )
        curvature_for_fsm = curvature if roundabout_armed else 0.0
        self.fsm.update(curvature_for_fsm, self.lidar.obstacle_dist, dt)

        # ── Commande vitesse ──────────────────────────────────────────────
        twist = Twist()

        if self.fsm.is_running:
            linear, angular = self.ctrl.compute(
                error, curvature, dt,
                in_roundabout=self.fsm.in_roundabout
            )
            twist.linear.x  = linear
            twist.angular.z = angular
        elif self.fsm.state == State.OBSTACLE_STOP:
            # Stop complet mais publie 0 explicitement
            twist.linear.x  = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        # ── Affichage debug ───────────────────────────────────────────────
        if self._debug_mode:
            self._draw_debug(result, error, curvature, twist)

        cv2.waitKey(1)

    # ─── Clavier ──────────────────────────────────────────────────────────────
    def _handle_keys(self):
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('s'):
            self._start_robot()
            self.get_logger().info("▶ Démarrage")
        elif key == ord('q'):
            self._stop_robot()
            self.get_logger().info("■ Arrêt")
        elif key == ord('l'):
            self.fsm.set_direction('left')
            self.get_logger().info("↰ Rond-point GAUCHE")
        elif key == ord('r'):
            self.fsm.set_direction('right')
            self.get_logger().info("↱ Rond-point DROITE")
        elif key == ord('+') or key == ord('='):
            self.ctrl.linear_speed = min(self.ctrl.linear_speed + 0.01, 0.25)
        elif key == ord('-'):
            self.ctrl.linear_speed = max(self.ctrl.linear_speed - 0.01, 0.02)
        elif key == ord('d'):
            self._debug_mode = not self._debug_mode
        elif key == ord('c'):
            self._show_calibration_hint()

    def _gui_spin(self):
        self._handle_keys()
        if not self._got_camera:
            canvas = np.zeros((240, 520, 3), dtype=np.uint8)
            cv2.putText(canvas, "EN ATTENTE CAMERA", (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(canvas, "Topics essayes:", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            y = 120
            for topic in self._camera_topics[:6]:
                cv2.putText(canvas, topic, (42, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 255), 1)
                y += 18
            cv2.putText(canvas, "[s]=start [q]=stop", (30, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
            cv2.imshow("Challenge1_BEV", canvas)

    def _show_calibration_hint(self):
        """Affiche les points de calibration BEV actuels."""
        self.get_logger().info(
            "Points BEV actuels (SRC_POINTS) dans vision.py — "
            "Modifie-les selon ta vue caméra Gazebo"
        )

    # ─── Debug visuel ─────────────────────────────────────────────────────────
    def _draw_debug(self, result, error, curvature, twist):
        debug = result['debug_bev'].copy()
        state_color = COLOR.get(self.fsm.state, (255, 255, 255))

        # Barre d'état
        cv2.rectangle(debug, (0, 0), (debug.shape[1], 20), (30, 30, 30), -1)
        cv2.putText(debug, str(self.fsm), (5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, state_color, 1)

        # Métriques
        y0 = 30
        lines = [
            f"err={error:+.1f}px  curv={curvature:.5f}",
            f"v={twist.linear.x:.3f}m/s  w={twist.angular.z:+.3f}rad/s",
            f"LIDAR: {self.lidar.status_str()}",
            f"Kp={self.ctrl.kp:.4f} Kd={self.ctrl.kd:.4f}",
            f"[s]=start [q]=stop [l/r]=dir [+/-]=speed",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(debug, txt, (5, y0 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        # Barre d'erreur latérale
        bar_cx = debug.shape[1] // 2
        bar_y  = debug.shape[0] - 12
        bar_len = 100
        cv2.line(debug, (bar_cx - bar_len, bar_y),
                        (bar_cx + bar_len, bar_y), (60, 60, 60), 4)
        err_px = int(np.clip(error, -bar_len, bar_len))
        color  = (0, 200, 0) if abs(error) < 20 else \
                 (0, 165, 255) if abs(error) < 60 else (0, 0, 255)
        cv2.circle(debug, (bar_cx + err_px, bar_y), 5, color, -1)

        # Indicateur direction
        dir_txt = f"← GAUCHE" if self.fsm.roundabout_direction == 'left' else "DROITE →"
        cv2.putText(debug, dir_txt,
                    (debug.shape[1] - 80, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 165, 255) if self.fsm.in_roundabout else (150, 150, 150), 1)

        cv2.imshow("Challenge1_BEV", debug)

        # Fenêtre masques
        mask_g = result['mask_green']
        mask_r = result['mask_red']
        masks_combined = np.zeros((*mask_g.shape, 3), dtype=np.uint8)
        masks_combined[mask_g > 0] = [0, 200, 0]
        masks_combined[mask_r > 0] = [0, 0, 200]
        cv2.imshow("Challenge1_Masks", masks_combined)

    # ─── Sécurité ─────────────────────────────────────────────────────────────
    def _safety_check(self):
        """Stoppe le robot si plus d'images depuis 2s."""
        if time.time() - self._last_img_time > 2.0 and self.fsm.is_running:
            self.get_logger().warn("⚠ Pas d'image depuis 2s — arrêt sécurité")
            self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = Challenge1Node()
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
