"""
challenge3_node.py — Challenge 3 : Corridor Navigation (LIDAR uniquement)
Pour TurtleBot3 RÉEL — Version robuste.

Fonctionne avec ou sans display (SSH headless OK).

Stratégie : wall-centering PD + détection virage en bout de couloir.

Lancer :
  ros2 launch projet challenge3.launch.py
  # ou directement :
  ros2 run projet challenge3

Le robot démarre en mode STOP. Appuyer [s] dans la fenêtre OpenCV
ou publier manuellement pour démarrer.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import time
import math
from collections import deque

from .lidar_handler import LidarHandler

# OpenCV optionnel (pour SSH headless)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class CorridorNavigator(Node):

    def __init__(self):
        super().__init__('challenge3_corridor')

        # ── Paramètres ROS2 ──────────────────────────────────────────────
        self.declare_parameter('linear_speed', 0.05)
        self.declare_parameter('kp',           0.8)
        self.declare_parameter('kd',           0.3)
        self.declare_parameter('target_gap',   0.0)
        self.declare_parameter('lidar_front_index', 0)
        self.declare_parameter('auto_start', False)

        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.kp    = float(self.get_parameter('kp').value)
        self.kd    = float(self.get_parameter('kd').value)
        self.target_gap = float(self.get_parameter('target_gap').value)
        front_idx  = int(self.get_parameter('lidar_front_index').value)
        auto_start = bool(self.get_parameter('auto_start').value)

        self.running = auto_start

        # ── LIDAR handler ────────────────────────────────────────────────
        self.lidar = LidarHandler(
            stop_dist=0.20,
            avoid_dist=0.40,
            lidar_front_index=front_idx
        )

        # ── Contrôleur PD ────────────────────────────────────────────────
        self._last_error = 0.0
        self._last_time  = time.time()
        self._error_hist = deque(maxlen=5)

        # ── État du couloir ──────────────────────────────────────────────
        self._turn_state = 'straight'
        self._turn_timer = 0.0
        self._front_blocked_count = 0
        self._preferred_turn = 1.0

        # ── Log ──────────────────────────────────────────────────────────
        self._log_timer = 0.0
        self._diag_printed = False

        # ── GUI (optionnel) ──────────────────────────────────────────────
        self._has_gui = False
        if HAS_CV2:
            try:
                cv2.namedWindow("Challenge3_Corridor")
                cv2.createTrackbar("Vitesse x100", "Challenge3_Corridor",
                                   int(self.linear_speed * 100), 20,
                                   lambda v: setattr(self, 'linear_speed', v / 100.0))
                cv2.createTrackbar("Kp x10", "Challenge3_Corridor",
                                   int(self.kp * 10), 50,
                                   lambda v: setattr(self, 'kp', v / 10.0))
                cv2.createTrackbar("Kd x10", "Challenge3_Corridor",
                                   int(self.kd * 10), 30,
                                   lambda v: setattr(self, 'kd', v / 10.0))
                cv2.createTrackbar("0=STOP 1=START", "Challenge3_Corridor", 0, 1,
                                   lambda v: setattr(self, 'running', bool(v)))
                self._has_gui = True
            except Exception:
                self.get_logger().warn("Pas de display — mode headless (logs only)")

        # ── Topics ROS2 ─────────────────────────────────────────────────
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer de contrôle à 10 Hz
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            f"Challenge3 Corridor PRÊT | v={self.linear_speed} Kp={self.kp} "
            f"Kd={self.kd} front_idx={front_idx} "
            f"auto_start={auto_start} gui={self._has_gui}"
        )

    # ─── LIDAR callback ─────────────────────────────────────────────────────
    def scan_callback(self, msg):
        self.lidar.update(msg)

        # Premier message : afficher les paramètres LIDAR pour diagnostic
        if not self._diag_printed and self.lidar._msg_count >= 1:
            self._diag_printed = True
            self.get_logger().info(f"LIDAR config : {self.lidar.diag_str()}")
            self.get_logger().info(f"LIDAR brut   : {self.lidar.status_str()}")

    # ─── Boucle de contrôle (10 Hz) ─────────────────────────────────────────
    def control_loop(self):
        now = time.time()
        dt  = max(now - self._last_time, 0.001)
        dt  = min(dt, 0.2)
        self._last_time = now

        # Clavier (si GUI disponible)
        if self._has_gui:
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    self.running = True
                    self._last_error = 0.0
                    self._error_hist.clear()
                    self.get_logger().info("▶ Corridor START")
                elif key == ord('q'):
                    self.running = False
                    self.get_logger().info("■ Corridor STOP")
            except Exception:
                pass

        twist = Twist()

        if self.running:
            twist = self._compute_command(dt)

        self.cmd_pub.publish(twist)

        if self._has_gui:
            try:
                self._draw_view(twist)
            except Exception:
                pass

    # ─── Calcul commande ─────────────────────────────────────────────────────
    def _compute_command(self, dt):
        twist = Twist()

        left = self.lidar.left_dist
        right = self.lidar.right_dist
        front = self.lidar.front_narrow_dist
        front_min = self.lidar.front_narrow_min_dist
        fl = self.lidar.front_left_dist
        fr = self.lidar.front_right_dist

        # Challenge 3 = couloir parallele: no in-place turning. Keep centered.
        raw_error = (left - right) - self.target_gap
        heading_error = fl - fr
        self._error_hist.append(raw_error)
        error = float(np.mean(self._error_hist))

        d_error = float(np.clip((error - self._last_error) / dt, -1.5, 1.5))
        self._last_error = error

        angular = self.kp * error + self.kd * d_error + 0.18 * heading_error
        angular = float(np.clip(angular, -0.42, 0.42))

        if front < 0.16 and front_min < 0.14:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self._turn_state = 'blocked'
        else:
            self._turn_state = 'straight'
            front_factor = float(np.clip((front - 0.18) / 0.75, 0.45, 1.0))
            turn_factor = 1.0 - min(abs(angular) / 0.65, 0.40)
            twist.linear.x = float(self.linear_speed * front_factor * turn_factor)
            twist.angular.z = angular

        self._log_timer += dt
        if self._log_timer > 0.5:
            self._log_timer = 0.0
            self.get_logger().info(
                f"[{self._turn_state:>12s}] "
                f"L={left:.2f} FL={fl:.2f} Fn={front:.2f}/{front_min:.2f} "
                f"FR={fr:.2f} R={right:.2f} | "
                f"v={twist.linear.x:.3f} w={twist.angular.z:+.3f}")

        return twist

        left  = self.lidar.left_dist
        right = self.lidar.right_dist
        front = self.lidar.front_dist
        fl    = self.lidar.front_left_dist
        fr    = self.lidar.front_right_dist

        # ── Détection virage ─────────────────────────────────────────────
        WALL_CLOSE = 0.35
        SIDE_OPEN  = 0.60

        if self._turn_state == 'straight':
            if front < WALL_CLOSE:
                if left > right:
                    self._turn_state = 'turn_left'
                    self._turn_timer = 0.0
                    self.get_logger().info(
                        f"↰ VIRAGE GAUCHE | F={front:.2f} L={left:.2f} R={right:.2f}")
                else:
                    self._turn_state = 'turn_right'
                    self._turn_timer = 0.0
                    self.get_logger().info(
                        f"↱ VIRAGE DROITE | F={front:.2f} L={left:.2f} R={right:.2f}")

        if self._turn_state == 'turn_left':
            self._turn_timer += dt
            twist.linear.x  = 0.02
            twist.angular.z = 0.5
            if front > 0.45 and self._turn_timer > 0.5:
                self._turn_state = 'straight'
                self._error_hist.clear()
                self._last_error = 0.0
                self.get_logger().info("→ Retour STRAIGHT")

        elif self._turn_state == 'turn_right':
            self._turn_timer += dt
            twist.linear.x  = 0.02
            twist.angular.z = -0.5
            if front > 0.45 and self._turn_timer > 0.5:
                self._turn_state = 'straight'
                self._error_hist.clear()
                self._last_error = 0.0
                self.get_logger().info("→ Retour STRAIGHT")

        else:
            # ── Wall-centering PD ────────────────────────────────────────
            # error > 0 quand plus d'espace à gauche → robot trop à droite
            #   → angular.z > 0 → tourne à gauche ✓
            # error < 0 quand plus d'espace à droite → robot trop à gauche
            #   → angular.z < 0 → tourne à droite ✓
            raw_error = (left - right) - self.target_gap
            self._error_hist.append(raw_error)
            error = float(np.mean(self._error_hist))

            d_error = (error - self._last_error) / dt
            self._last_error = error

            angular = float(np.clip(
                self.kp * error + self.kd * d_error,
                -1.2, 1.2
            ))

            # Vitesse adaptative
            if front < 0.20:
                twist.linear.x  = 0.0
                twist.angular.z = 0.0
                self.get_logger().warn(
                    f"⚠ MUR — F={front:.2f}", throttle_duration_sec=1.0)
            elif front < 0.35:
                twist.linear.x  = 0.02
                twist.angular.z = angular
            else:
                sym = 1.0 - min(abs(error) / 0.4, 0.5)
                twist.linear.x  = float(self.linear_speed * sym)
                twist.angular.z = angular

        # ── Log périodique ────────────────────────────────────────────────
        self._log_timer += dt
        if self._log_timer > 0.5:
            self._log_timer = 0.0
            self.get_logger().info(
                f"[{self._turn_state:>12s}] "
                f"L={left:.2f} FL={fl:.2f} F={front:.2f} "
                f"FR={fr:.2f} R={right:.2f} | "
                f"v={twist.linear.x:.3f} w={twist.angular.z:+.3f}")

        return twist

    # ─── Visualisation ───────────────────────────────────────────────────────
    def _draw_view(self, twist):
        W, H = 400, 350
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        cx, cy = W // 2, H - 80

        # Robot
        cv2.circle(canvas, (cx, cy), 12, (0, 200, 255), -1)

        scale = 180

        def draw_ray(angle_deg, dist, color):
            d_px = int(min(dist * scale, W // 2 - 20))
            rad  = math.radians(angle_deg - 90)
            ex = cx + int(d_px * math.cos(rad))
            ey = cy + int(d_px * math.sin(rad))
            cv2.line(canvas, (cx, cy), (ex, ey), color, 2)
            cv2.circle(canvas, (ex, ey), 4, color, -1)
            lx = ex + 5 if ex > cx else ex - 55
            cv2.putText(canvas, f"{dist:.2f}", (lx, ey - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        draw_ray(90,  self.lidar.left_dist,        (0, 255, 0))
        draw_ray(270, self.lidar.right_dist,       (0, 255, 0))
        draw_ray(0,   self.lidar.front_dist,       (0, 150, 255))
        draw_ray(45,  self.lidar.front_left_dist,  (100, 200, 100))
        draw_ray(315, self.lidar.front_right_dist, (100, 200, 100))

        y = 20
        infos = [
            (f"L={self.lidar.left_dist:.2f}  R={self.lidar.right_dist:.2f}",
             (0, 255, 0)),
            (f"F={self.lidar.front_dist:.2f}  FL={self.lidar.front_left_dist:.2f}"
             f"  FR={self.lidar.front_right_dist:.2f}",
             (0, 150, 255)),
            (f"v={twist.linear.x:.3f}  w={twist.angular.z:+.3f}",
             (200, 200, 200)),
            (f"State: {self._turn_state.upper()}",
             (0, 220, 220) if 'turn' in self._turn_state else (0, 220, 0)),
            (f"Kp={self.kp:.1f}  Kd={self.kd:.1f}  v_max={self.linear_speed:.2f}",
             (150, 150, 150)),
        ]
        for txt, col in infos:
            cv2.putText(canvas, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
            y += 16

        status = "RUN" if self.running else "STOP"
        cv2.rectangle(canvas, (0, H - 22), (W, H), (30, 30, 30), -1)
        cv2.putText(canvas, f"[{status}]  [s]=start  [q]=stop",
                    (5, H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 220, 0) if self.running else (0, 0, 200), 1)

        cv2.imshow("Challenge3_Corridor", canvas)


def main(args=None):
    rclpy.init(args=args)
    node = CorridorNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
        if HAS_CV2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == '__main__':
    main()
