# """
# challenge4_node.py — Challenge 4 : Soccer — Pousser la balle dans le but
# Pour TurtleBot3 RÉEL.

# Capteurs : caméra (détection balle HSV jaune) + LIDAR (sécurité murs)

# États internes :
#   SEARCH   → tourne sur lui-même pour trouver la balle
#   APPROACH → s'aligne et avance vers la balle
#   PUSH     → pousse la balle vers le but (avec correction angulaire)
#   DONE     → arrêt

# Corrections par rapport à l'ancienne version :
#   - HSV restreint au JAUNE (30-45) pour ne pas confondre avec les lignes vertes
#   - Détection par contours + circularité au lieu de HoughCircles (plus fiable)
#   - Filtrage temporel (EMA) de la position balle
#   - Approche continue (pas d'arrêt à 40cm)
#   - Push avec correction angulaire (suit la balle pendant le push)
#   - Sécurité LIDAR pendant le push
#   - Confirmation sur 3+ frames avant SEARCH→APPROACH

# Commandes clavier :
#   [s] Démarrer  [q] Arrêter
# """

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage, Image, LaserScan
# from geometry_msgs.msg import Twist
# from rclpy.qos import qos_profile_sensor_data
# import numpy as np
# import cv2
# import time
# from enum import Enum, auto
# from collections import deque

# #from .lidar_handler import LidarHandler
# # Par ceci :
# import sys, os
# sys.path.insert(0, os.path.dirname(__file__))
# from lidar_handler import LidarHandler

# class SoccerState(Enum):
#     IDLE     = auto()
#     SEARCH   = auto()
#     APPROACH = auto()
#     PUSH     = auto()
#     DONE     = auto()


# class BallDetector:
#     """
#     Détecte une balle de tennis (jaune vif) dans l'image.

#     Stratégie pour robot réel :
#     1. CLAHE pour normaliser la luminosité
#     2. Masque HSV centré sur JAUNE (Hue 25-45), pas vert !
#        → Évite de confondre avec les lignes vertes de la piste (Hue ~60)
#     3. Détection par contours + test de circularité
#        → Plus robuste que HoughCircles sur images bruitées
#     4. Filtrage temporel (EMA) de la position
#     """

#     def __init__(self, seuil=12):
#         self.seuil = seuil   # ±seuil autour du centre Hue
#         self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

#         # Filtrage temporel
#         self._ema_cx = None
#         self._ema_cy = None
#         self._ema_r  = None
#         self._alpha  = 0.5   # réactivité EMA (0.3=lent, 0.7=rapide)

#         # Compteur de détection consécutive
#         self.consecutive_detections = 0
#         self.consecutive_misses     = 0

#     def detect(self, frame):
#         """
#         Retourne (cx, cy, radius, mask) ou None.
#         cx, cy  : centroïde filtré en pixels
#         radius  : rayon estimé (proxy de distance)
#         mask    : masque binaire pour debug
#         """
#         # Prétraitement CLAHE
#         lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)
#         l = self._clahe.apply(l)
#         frame_eq = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

#         hsv = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2HSV)
#         s = max(5, self.seuil)

#         # ── Masque HSV : JAUNE uniquement ────────────────────────────────
#         # Balle de tennis : Hue ≈ 25-45 (jaune-vert clair)
#         # ON EXCLUT le vert pur (Hue 50-80) pour ne pas capter les lignes
#         # Saturation et Value minimums plus élevés car la balle est VIVE
#         h_center = 35   # centre du jaune
#         mask = cv2.inRange(hsv,
#                            np.array([max(0, h_center - s), 80, 80]),
#                            np.array([min(65, h_center + s), 255, 255]))

#         # Morphologie
#         k = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

#         # ── Détection par contours ───────────────────────────────────────
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
#                                         cv2.CHAIN_APPROX_SIMPLE)

#         best = None
#         best_area = 0

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area < 200:   # trop petit = bruit
#                 continue

#             # Test de circularité : périmètre² / (4π × aire)
#             # Cercle parfait = 1.0, carré ≈ 1.27
#             perimeter = cv2.arcLength(cnt, True)
#             if perimeter == 0:
#                 continue
#             circularity = 4.0 * np.pi * area / (perimeter * perimeter)

#             if circularity < 0.4:   # pas assez circulaire
#                 continue

#             if area > best_area:
#                 best_area = area
#                 best = cnt

#         if best is not None:
#             # Centroïde et rayon
#             M = cv2.moments(best)
#             if M['m00'] > 0:
#                 raw_cx = int(M['m10'] / M['m00'])
#                 raw_cy = int(M['m01'] / M['m00'])
#                 raw_r  = int(np.sqrt(best_area / np.pi))

#                 # Filtrage EMA
#                 if self._ema_cx is None:
#                     self._ema_cx = float(raw_cx)
#                     self._ema_cy = float(raw_cy)
#                     self._ema_r  = float(raw_r)
#                 else:
#                     self._ema_cx = self._alpha * raw_cx + (1 - self._alpha) * self._ema_cx
#                     self._ema_cy = self._alpha * raw_cy + (1 - self._alpha) * self._ema_cy
#                     self._ema_r  = self._alpha * raw_r  + (1 - self._alpha) * self._ema_r

#                 self.consecutive_detections += 1
#                 self.consecutive_misses = 0

#                 return (int(self._ema_cx), int(self._ema_cy),
#                         int(self._ema_r), mask)

#         # Pas de détection
#         self.consecutive_misses += 1
#         self.consecutive_detections = 0
#         return None

#     def reset(self):
#         self._ema_cx = None
#         self._ema_cy = None
#         self._ema_r  = None
#         self.consecutive_detections = 0
#         self.consecutive_misses = 0


# class Challenge4Node(Node):

#     # Rayon approximatif d'une balle de tennis à 0.3m
#     # À calibrer : place la balle à 30cm, note le rayon affiché
#     RADIUS_AT_30CM = 35

#     def __init__(self):
#         super().__init__('challenge4_soccer')

#         # ── Paramètres ROS2 ──────────────────────────────────────────────
#         self.declare_parameter('linear_speed',      0.08)
#         self.declare_parameter('push_speed',        0.15)
#         self.declare_parameter('seuil_balle',       12)
#         self.declare_parameter('push_duration',     3.0)
#         self.declare_parameter('lidar_front_index', 0)

#         self.linear_speed = self.get_parameter('linear_speed').value
#         self.push_speed   = self.get_parameter('push_speed').value
#         seuil             = self.get_parameter('seuil_balle').value
#         self.push_duration = self.get_parameter('push_duration').value
#         front_idx         = self.get_parameter('lidar_front_index').value

#         # ── Modules ──────────────────────────────────────────────────────
#         self.detector = BallDetector(seuil=seuil)
#         self.lidar    = LidarHandler(
#             stop_dist=0.20,
#             avoid_dist=0.40,
#             lidar_front_index=front_idx
#         )
#         self.state = SoccerState.IDLE

#         # ── Timers ───────────────────────────────────────────────────────
#         self._last_time      = time.time()
#         self._search_timer   = 0.0
#         self._approach_timer = 0.0
#         self._push_timer     = 0.0
#         self._search_dir     = 1.0
#         self._kp_ball        = 0.004
#         self._log_timer      = 0.0

#         # Confirmation de détection
#         self._detect_confirm = 3   # frames consécutives avant APPROACH

#         # ── GUI ──────────────────────────────────────────────────────────
#         cv2.namedWindow("Challenge4_Soccer")
#         cv2.createTrackbar("Vitesse x100", "Challenge4_Soccer",
#                            int(self.linear_speed * 100), 25,
#                            lambda v: setattr(self, 'linear_speed', v / 100.0))
#         cv2.createTrackbar("Push speed x100", "Challenge4_Soccer",
#                            int(self.push_speed * 100), 30,
#                            lambda v: setattr(self, 'push_speed', v / 100.0))
#         cv2.createTrackbar("Seuil balle", "Challenge4_Soccer", seuil, 40,
#                            lambda s: setattr(self.detector, 'seuil', max(s, 5)))
#         cv2.createTrackbar("Kp x1000", "Challenge4_Soccer",
#                            int(self._kp_ball * 1000), 15,
#                            lambda v: setattr(self, '_kp_ball', v / 1000.0))
#         cv2.createTrackbar("Push dur x10", "Challenge4_Soccer",
#                            int(self.push_duration * 10), 60,
#                            lambda v: setattr(self, 'push_duration', v / 10.0))
#         cv2.createTrackbar("0=STOP 1=START", "Challenge4_Soccer", 0, 1,
#                            lambda v: self._start() if v == 1 else self._stop())

#         # ── Topics ROS2 ─────────────────────────────────────────────────
#         # Auto-souscription à plusieurs topics caméra possibles
#         from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
#         from cv_bridge import CvBridge
#         self._bridge = CvBridge()
#         self._last_frame = None

#         qos_cam = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             history=HistoryPolicy.KEEP_LAST, depth=1)

#         for topic in ['/camera/image_raw/compressed',
#                       '/image_raw/compressed',
#                       '/camera/image/compressed']:
#             self.create_subscription(
#                 CompressedImage, topic, self._compressed_cb, qos_cam)

#         for topic in ['/camera/image_raw', '/image_raw']:
#             self.create_subscription(
#                 Image, topic, self._raw_cb, qos_cam)

#         self.scan_sub = self.create_subscription(
#             LaserScan, '/scan', self.scan_callback, 10)
#         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

#         self.get_logger().info(
#             f"Challenge4 Soccer PRÊT | v={self.linear_speed} push={self.push_speed} "
#             f"seuil={seuil} | [s]=start [q]=stop"
#         )

#         # Timer de contrôle 10 Hz
#         self._control_timer = self.create_timer(0.1, self._control_loop)

#     def _start(self):
#         self.state = SoccerState.SEARCH
#         self._search_timer = 0.0
#         self.detector.reset()
#         self.get_logger().info("⚽ Soccer : SEARCH")

#     def _stop(self):
#         self.state = SoccerState.IDLE
#         self.cmd_pub.publish(Twist())

#     # ─── LIDAR ────────────────────────────────────────────────────────────────
#     def scan_callback(self, msg):
#         self.lidar.update(msg)

#     # ─── Callbacks caméra (stocke la dernière frame) ─────────────────────────
#     def _compressed_cb(self, msg):
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         if frame is not None:
#             self._last_frame = frame

#     def _raw_cb(self, msg):
#         try:
#             self._last_frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
#         except Exception:
#             pass

#     # ─── Boucle de contrôle ──────────────────────────────────────────────────
#     def _control_loop(self):
#         # Clavier (marche toujours)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('s'):
#             self._start()
#         elif key == ord('q'):
#             self._stop()

#         frame = self._last_frame
#         if frame is None:
#             # Écran d'attente
#             canvas = np.zeros((200, 400, 3), dtype=np.uint8)
#             cv2.putText(canvas, "ATTENTE CAMERA...", (50, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             cv2.putText(canvas, f"LIDAR: F={self.lidar.front_dist:.2f}m",
#                         (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
#             cv2.imshow("Challenge4_Soccer", canvas)
#             return

#         self._process_frame(frame)

#     # ─── Traitement image ────────────────────────────────────────────────────
#     def _process_frame(self, frame):
#         h, w = frame.shape[:2]
#         now  = time.time()
#         dt   = max(now - self._last_time, 0.001)
#         dt   = min(dt, 0.1)
#         self._last_time = now

#         # Clavier supprimé ici, géré dans _control_loop

#         twist = Twist()
#         detection = self.detector.detect(frame)
#         debug = frame.copy()

#         # ── IDLE ─────────────────────────────────────────────────────────
#         if self.state == SoccerState.IDLE:
#             pass

#         # ── SEARCH ───────────────────────────────────────────────────────
#         elif self.state == SoccerState.SEARCH:
#             self._search_timer += dt

#             if (detection is not None and
#                     self.detector.consecutive_detections >= self._detect_confirm):
#                 # Balle confirmée sur N frames consécutives → APPROACH
#                 self.state = SoccerState.APPROACH
#                 self._approach_timer = 0.0
#                 self.get_logger().info("⚽ APPROACH — balle confirmée")
#             else:
#                 # Rotation lente pour chercher
#                 twist.angular.z = self._search_dir * 0.35
#                 # Inverse sens si cherché trop longtemps
#                 if self._search_timer > 6.0:
#                     self._search_dir *= -1
#                     self._search_timer = 0.0
#                     self.get_logger().info("⚽ SEARCH — inversion sens de rotation")

#         # ── APPROACH ─────────────────────────────────────────────────────
#         elif self.state == SoccerState.APPROACH:
#             self._approach_timer += dt

#             if detection is None:
#                 # Balle perdue
#                 if self.detector.consecutive_misses > 20:
#                     # Perdue depuis longtemps → retour SEARCH
#                     self.state = SoccerState.SEARCH
#                     self._search_timer = 0.0
#                     self.detector.reset()
#                     self.get_logger().info("⚽ Balle perdue → SEARCH")
#                 else:
#                     # Perdue temporairement → continue dans la dernière direction
#                     twist.linear.x = float(self.linear_speed * 0.3)
#             else:
#                 bx, by, br, ball_mask = detection
#                 error_x = bx - w // 2

#                 # Commande angulaire proportionnelle
#                 twist.angular.z = float(np.clip(
#                     -self._kp_ball * error_x, -1.0, 1.0))

#                 # Distance estimée par rayon
#                 if br > 5:
#                     dist_est = self.RADIUS_AT_30CM / br * 0.30
#                 else:
#                     # Fallback : utiliser la position Y dans l'image
#                     # Plus la balle est en bas de l'image, plus elle est proche
#                     dist_est = max(0.15, 1.0 - (by / h))

#                 # ── Vitesse d'approche CONTINUE ──────────────────────────
#                 # Ancien bug : linear.x = 0 quand dist < 0.40 → le robot s'arrêtait
#                 # avant d'atteindre la balle. Maintenant il avance toujours.
#                 if dist_est > 0.60:
#                     twist.linear.x = float(self.linear_speed * 0.7)
#                 elif dist_est > 0.30:
#                     twist.linear.x = float(self.linear_speed * 0.4)
#                 else:
#                     twist.linear.x = float(self.linear_speed * 0.2)

#                 # Ralentir si mal aligné
#                 if abs(error_x) > 60:
#                     twist.linear.x *= 0.3

#                 # ── Transition vers PUSH ─────────────────────────────────
#                 # Condition : bien aligné ET proche
#                 # Utilise aussi la position Y (balle en bas de l'image = proche)
#                 close_by_radius = dist_est < 0.35
#                 close_by_position = by > h * 0.70
#                 aligned = abs(error_x) < 50

#                 if aligned and (close_by_radius or close_by_position):
#                     self.state = SoccerState.PUSH
#                     self._push_timer = 0.0
#                     self.get_logger().info(
#                         f"⚽ PUSH ! dist≈{dist_est:.2f}m by={by}/{h} err={error_x}")

#                 # Debug
#                 cv2.circle(debug, (bx, by), max(br, 10), (0, 255, 255), 2)
#                 cv2.putText(debug, f"d={dist_est:.2f}m r={br}px",
#                             (bx - 40, by - 15),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

#                 # Log
#                 self._log_timer += dt
#                 if self._log_timer > 0.5:
#                     self._log_timer = 0.0
#                     self.get_logger().info(
#                         f"⚽ err={error_x:+4d}px d≈{dist_est:.2f}m r={br} "
#                         f"by={by}/{h} F={self.lidar.front_dist:.2f}m")

#         # ── PUSH ─────────────────────────────────────────────────────────
#         elif self.state == SoccerState.PUSH:
#             self._push_timer += dt

#             # Vitesse de push
#             twist.linear.x = float(self.push_speed)

#             # ── Correction angulaire pendant le push ─────────────────────
#             # Si on voit encore la balle, on corrige la direction
#             # (ancien bug : angular.z = 0 fixe → rate si la balle dévie)
#             if detection is not None:
#                 bx, by, br, _ = detection
#                 error_x = bx - w // 2
#                 twist.angular.z = float(np.clip(
#                     -self._kp_ball * error_x * 0.5, -0.5, 0.5))

#             # ── Sécurité LIDAR : arrêt si mur trop proche ───────────────
#             if self.lidar.front_dist < 0.15:
#                 twist.linear.x = 0.0
#                 twist.angular.z = 0.0
#                 self.state = SoccerState.DONE
#                 self.get_logger().info("⚽ STOP — mur trop proche")

#             # Fin du push par timer
#             if self._push_timer > self.push_duration:
#                 self.state = SoccerState.DONE
#                 self.get_logger().info("⚽ DONE !")

#         # ── DONE ─────────────────────────────────────────────────────────
#         elif self.state == SoccerState.DONE:
#             twist.linear.x = 0.0
#             cv2.putText(debug, "GOAL !!!!", (w // 2 - 80, h // 2),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 220, 0), 3)

#         self.cmd_pub.publish(twist)

#         # ── Overlay debug ────────────────────────────────────────────────
#         state_colors = {
#             SoccerState.IDLE:     (128, 128, 128),
#             SoccerState.SEARCH:   (255, 165, 0),
#             SoccerState.APPROACH: (0, 200, 0),
#             SoccerState.PUSH:     (0, 0, 255),
#             SoccerState.DONE:     (0, 220, 0),
#         }
#         col = state_colors.get(self.state, (255, 255, 255))

#         # Barre d'état
#         cv2.rectangle(debug, (0, 0), (w, 22), (20, 20, 20), -1)
#         cv2.putText(debug, f"SOCCER: {self.state.name}", (5, 16),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

#         # Infos LIDAR
#         cv2.putText(debug,
#                     f"F={self.lidar.front_dist:.2f}m  "
#                     f"L={self.lidar.left_dist:.2f}  R={self.lidar.right_dist:.2f}",
#                     (5, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
#         cv2.putText(debug, f"v={twist.linear.x:.2f} w={twist.angular.z:+.2f}  "
#                     f"[s]=start [q]=stop",
#                     (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

#         # Afficher le masque en petit dans le coin
#         if detection is not None:
#             _, _, _, ball_mask = detection
#             mask_small = cv2.resize(ball_mask, (w // 4, h // 4))
#             mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
#             mask_color[mask_small > 0] = [0, 255, 255]
#             debug[0:h // 4, w - w // 4:w] = mask_color

#         cv2.imshow("Challenge4_Soccer", debug)


# def main(args=None):
#     rclpy.init(args=args)
#     node = Challenge4Node()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.cmd_pub.publish(Twist())
#         node.destroy_node()
#         rclpy.shutdown()
#         cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()


"""
challenge4_node.py — Challenge 4 : Soccer — Pousser la balle dans le but
Pour TurtleBot3 RÉEL.

Capteurs : caméra (détection balle HSV jaune) + LIDAR (sécurité murs)

États internes :
  SEARCH      → tourne sur lui-même pour trouver la balle
  APPROACH    → s'aligne et avance vers la balle
  ALIGN_GOAL  → se positionne dans l'axe balle→but (poteaux rouges)
  PUSH        → pousse la balle vers le but (avec correction angulaire)
  DONE        → arrêt

Commandes clavier :
  [s] Démarrer  [q] Arrêter
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import cv2
import time
from enum import Enum, auto
from collections import deque

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_handler import LidarHandler


class SoccerState(Enum):
    IDLE       = auto()
    SEARCH     = auto()
    APPROACH   = auto()
    ALIGN_GOAL = auto()   # ← NOUVEAU : vise les poteaux rouges
    PUSH       = auto()
    DONE       = auto()


# ─────────────────────────────────────────────────────────────────────────────
class BallDetector:
    """Détecte une balle de tennis (jaune vif) dans l'image."""

    def __init__(self, seuil=12):
        self.seuil = seuil
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._ema_cx = None
        self._ema_cy = None
        self._ema_r  = None
        self._alpha  = 0.5
        self.consecutive_detections = 0
        self.consecutive_misses     = 0

    def detect(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        frame_eq = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2HSV)
        s = max(5, self.seuil)
        h_center = 35
        mask = cv2.inRange(hsv,
                           np.array([max(0, h_center - s), 80, 80]),
                           np.array([min(65, h_center + s), 255, 255]))
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.4:
                continue
            if area > best_area:
                best_area = area
                best = cnt

        if best is not None:
            M = cv2.moments(best)
            if M['m00'] > 0:
                raw_cx = int(M['m10'] / M['m00'])
                raw_cy = int(M['m01'] / M['m00'])
                raw_r  = int(np.sqrt(best_area / np.pi))
                if self._ema_cx is None:
                    self._ema_cx = float(raw_cx)
                    self._ema_cy = float(raw_cy)
                    self._ema_r  = float(raw_r)
                else:
                    self._ema_cx = self._alpha * raw_cx + (1 - self._alpha) * self._ema_cx
                    self._ema_cy = self._alpha * raw_cy + (1 - self._alpha) * self._ema_cy
                    self._ema_r  = self._alpha * raw_r  + (1 - self._alpha) * self._ema_r
                self.consecutive_detections += 1
                self.consecutive_misses = 0
                return (int(self._ema_cx), int(self._ema_cy),
                        int(self._ema_r), mask)

        self.consecutive_misses += 1
        self.consecutive_detections = 0
        return None

    def reset(self):
        self._ema_cx = None
        self._ema_cy = None
        self._ema_r  = None
        self.consecutive_detections = 0
        self.consecutive_misses = 0


# ─────────────────────────────────────────────────────────────────────────────
class GoalDetector:
    """
    Détecte les deux poteaux ROUGES du but.

    Retourne :
      - goal_cx  : centre horizontal du but en pixels (milieu entre les 2 poteaux)
      - post_left, post_right : positions X des poteaux
      - mask     : masque binaire pour debug
    ou None si but non visible.
    """

    def __init__(self):
        # HSV rouge : deux plages car le rouge est à cheval sur 0/180
        self.red_lo1 = np.array([0,   120, 80])
        self.red_hi1 = np.array([10,  255, 255])
        self.red_lo2 = np.array([165, 120, 80])
        self.red_hi2 = np.array([180, 255, 255])

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Masque rouge (deux plages HSV)
        mask1 = cv2.inRange(hsv, self.red_lo1, self.red_hi1)
        mask2 = cv2.inRange(hsv, self.red_lo2, self.red_hi2)
        mask  = cv2.bitwise_or(mask1, mask2)

        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Garder les 2 plus grands contours (les 2 poteaux)
        valid = [c for c in contours if cv2.contourArea(c) > 150]
        if len(valid) < 1:
            return None

        # Trier par taille décroissante, garder les 2 meilleurs
        valid.sort(key=cv2.contourArea, reverse=True)
        posts = valid[:2]

        # Centroïdes des poteaux
        cx_list = []
        for cnt in posts:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx_list.append(int(M['m10'] / M['m00']))

        if len(cx_list) == 0:
            return None

        if len(cx_list) == 1:
            # Un seul poteau visible → centre estimé = ce poteau
            goal_cx = cx_list[0]
            return {
                'goal_cx':    goal_cx,
                'post_left':  None,
                'post_right': None,
                'n_posts':    1,
                'mask':       mask,
            }

        # Deux poteaux visibles
        post_left  = min(cx_list)
        post_right = max(cx_list)
        goal_cx    = (post_left + post_right) // 2

        return {
            'goal_cx':    goal_cx,
            'post_left':  post_left,
            'post_right': post_right,
            'n_posts':    2,
            'mask':       mask,
        }


# ─────────────────────────────────────────────────────────────────────────────
class Challenge4Node(Node):

    RADIUS_AT_30CM = 35

    def __init__(self):
        super().__init__('challenge4_soccer')

        # ── Paramètres ROS2 ──────────────────────────────────────────────
        self.declare_parameter('linear_speed',      0.08)
        self.declare_parameter('push_speed',        0.15)
        self.declare_parameter('seuil_balle',       12)
        self.declare_parameter('push_duration',     3.0)
        self.declare_parameter('lidar_front_index', 0)
        self.declare_parameter('align_timeout',     8.0)   # timeout ALIGN_GOAL

        self.linear_speed  = self.get_parameter('linear_speed').value
        self.push_speed    = self.get_parameter('push_speed').value
        seuil              = self.get_parameter('seuil_balle').value
        self.push_duration = self.get_parameter('push_duration').value
        front_idx          = self.get_parameter('lidar_front_index').value
        self.align_timeout = self.get_parameter('align_timeout').value

        # ── Modules ──────────────────────────────────────────────────────
        self.detector      = BallDetector(seuil=seuil)
        self.goal_detector = GoalDetector()
        self.lidar         = LidarHandler(
            stop_dist=0.20,
            avoid_dist=0.40,
            lidar_front_index=front_idx
        )
        self.state = SoccerState.IDLE

        # ── Timers internes ───────────────────────────────────────────────
        self._last_time      = time.time()
        self._search_timer   = 0.0
        self._approach_timer = 0.0
        self._align_timer    = 0.0
        self._push_timer     = 0.0
        self._search_dir     = 1.0
        self._kp_ball        = 0.004
        self._kp_goal        = 0.003   # gain proportionnel alignement but
        self._log_timer      = 0.0
        self._detect_confirm = 3

        # ── GUI ──────────────────────────────────────────────────────────
        cv2.namedWindow("Challenge4_Soccer")
        cv2.createTrackbar("Vitesse x100", "Challenge4_Soccer",
                           int(self.linear_speed * 100), 25,
                           lambda v: setattr(self, 'linear_speed', v / 100.0))
        cv2.createTrackbar("Push speed x100", "Challenge4_Soccer",
                           int(self.push_speed * 100), 30,
                           lambda v: setattr(self, 'push_speed', v / 100.0))
        cv2.createTrackbar("Seuil balle", "Challenge4_Soccer", seuil, 40,
                           lambda s: setattr(self.detector, 'seuil', max(s, 5)))
        cv2.createTrackbar("Kp ball x1000", "Challenge4_Soccer",
                           int(self._kp_ball * 1000), 15,
                           lambda v: setattr(self, '_kp_ball', v / 1000.0))
        cv2.createTrackbar("Kp goal x1000", "Challenge4_Soccer",
                           int(self._kp_goal * 1000), 15,
                           lambda v: setattr(self, '_kp_goal', v / 1000.0))
        cv2.createTrackbar("Push dur x10", "Challenge4_Soccer",
                           int(self.push_duration * 10), 60,
                           lambda v: setattr(self, 'push_duration', v / 10.0))
        cv2.createTrackbar("0=STOP 1=START", "Challenge4_Soccer", 0, 1,
                           lambda v: self._start() if v == 1 else self._stop())

        # ── Topics ROS2 ──────────────────────────────────────────────────
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from cv_bridge import CvBridge
        self._bridge     = CvBridge()
        self._last_frame = None

        qos_cam = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        for topic in ['/camera/image_raw/compressed',
                      '/image_raw/compressed',
                      '/camera/image/compressed']:
            self.create_subscription(
                CompressedImage, topic, self._compressed_cb, qos_cam)

        for topic in ['/camera/image_raw', '/image_raw']:
            self.create_subscription(
                Image, topic, self._raw_cb, qos_cam)

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info(
            f"Challenge4 Soccer PRÊT | v={self.linear_speed} "
            f"push={self.push_speed} seuil={seuil} | [s]=start [q]=stop"
        )

        self._control_timer = self.create_timer(0.1, self._control_loop)

    # ─── Start / Stop ─────────────────────────────────────────────────────────
    def _start(self):
        self.state = SoccerState.SEARCH
        self._search_timer = 0.0
        self.detector.reset()
        self.get_logger().info("⚽ Soccer : SEARCH")

    def _stop(self):
        self.state = SoccerState.IDLE
        self.cmd_pub.publish(Twist())

    # ─── LIDAR ────────────────────────────────────────────────────────────────
    def scan_callback(self, msg):
        self.lidar.update(msg)

    # ─── Caméra ───────────────────────────────────────────────────────────────
    def _compressed_cb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self._last_frame = frame

    def _raw_cb(self, msg):
        try:
            self._last_frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass

    # ─── Boucle de contrôle ──────────────────────────────────────────────────
    def _control_loop(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            self._start()
        elif key == ord('q'):
            self._stop()

        frame = self._last_frame
        if frame is None:
            canvas = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(canvas, "ATTENTE CAMERA...", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(canvas, f"LIDAR: F={self.lidar.front_dist:.2f}m",
                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.imshow("Challenge4_Soccer", canvas)
            return

        self._process_frame(frame)

    # ─── Traitement principal ─────────────────────────────────────────────────
    def _process_frame(self, frame):
        h, w  = frame.shape[:2]
        now   = time.time()
        dt    = max(now - self._last_time, 0.001)
        dt    = min(dt, 0.1)
        self._last_time = now

        twist     = Twist()
        detection = self.detector.detect(frame)
        goal_info = self.goal_detector.detect(frame)
        debug     = frame.copy()

        # ══════════════════════════════════════════════════════════════════
        # IDLE
        # ══════════════════════════════════════════════════════════════════
        if self.state == SoccerState.IDLE:
            pass

        # ══════════════════════════════════════════════════════════════════
        # SEARCH — tourne pour trouver la balle
        # ══════════════════════════════════════════════════════════════════
        elif self.state == SoccerState.SEARCH:
            self._search_timer += dt

            if (detection is not None and
                    self.detector.consecutive_detections >= self._detect_confirm):
                self.state = SoccerState.APPROACH
                self._approach_timer = 0.0
                self.get_logger().info("⚽ APPROACH — balle confirmée")
            else:
                twist.angular.z = self._search_dir * 0.35
                if self._search_timer > 6.0:
                    self._search_dir  *= -1
                    self._search_timer = 0.0
                    self.get_logger().info("⚽ SEARCH — inversion sens")

        # ══════════════════════════════════════════════════════════════════
        # APPROACH — s'aligne sur la balle et avance
        # ══════════════════════════════════════════════════════════════════
        elif self.state == SoccerState.APPROACH:
            self._approach_timer += dt

            if detection is None:
                if self.detector.consecutive_misses > 20:
                    self.state = SoccerState.SEARCH
                    self._search_timer = 0.0
                    self.detector.reset()
                    self.get_logger().info("⚽ Balle perdue → SEARCH")
                else:
                    twist.linear.x = float(self.linear_speed * 0.3)
            else:
                bx, by, br, ball_mask = detection
                error_x = bx - w // 2

                twist.angular.z = float(np.clip(
                    -self._kp_ball * error_x, -1.0, 1.0))

                if br > 5:
                    dist_est = self.RADIUS_AT_30CM / br * 0.30
                else:
                    dist_est = max(0.15, 1.0 - (by / h))

                if dist_est > 0.60:
                    twist.linear.x = float(self.linear_speed * 0.7)
                elif dist_est > 0.30:
                    twist.linear.x = float(self.linear_speed * 0.4)
                else:
                    twist.linear.x = float(self.linear_speed * 0.2)

                if abs(error_x) > 60:
                    twist.linear.x *= 0.3

                close_by_radius   = dist_est < 0.35
                close_by_position = by > h * 0.70
                aligned           = abs(error_x) < 50

                # ── Transition : balle proche → ALIGN_GOAL ───────────────
                if aligned and (close_by_radius or close_by_position):
                    self.state       = SoccerState.ALIGN_GOAL
                    self._align_timer = 0.0
                    self.get_logger().info(
                        f"⚽ ALIGN_GOAL ! dist≈{dist_est:.2f}m")

                # Debug balle
                cv2.circle(debug, (bx, by), max(br, 10), (0, 255, 255), 2)
                cv2.putText(debug, f"d={dist_est:.2f}m",
                            (bx - 30, by - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # ══════════════════════════════════════════════════════════════════
        # ALIGN_GOAL — tourne pour mettre le but dans l'axe
        # ══════════════════════════════════════════════════════════════════
        elif self.state == SoccerState.ALIGN_GOAL:
            self._align_timer += dt

            # Timeout → on pousse quand même (cas où le but est hors champ)
            if self._align_timer > self.align_timeout:
                self.state      = SoccerState.PUSH
                self._push_timer = 0.0
                self.get_logger().info("⚽ ALIGN timeout → PUSH direct")

            elif goal_info is not None:
                goal_cx = goal_info['goal_cx']
                error_goal = goal_cx - w // 2   # erreur but par rapport au centre

                # ── Logique d'alignement ──────────────────────────────────
                # On veut : but au centre de l'image (error_goal ≈ 0)
                # ET balle encore visible (pas perdue pendant la rotation)
                #
                # Le robot tourne SUR PLACE pour amener le but au centre
                # sans trop s'éloigner de la balle

                if abs(error_goal) < 40:
                    # But bien centré → PUSH !
                    self.state       = SoccerState.PUSH
                    self._push_timer = 0.0
                    self.get_logger().info(
                        f"⚽ But centré (err={error_goal}px) → PUSH !")
                else:
                    # Tourne doucement vers le but
                    twist.angular.z = float(np.clip(
                        -self._kp_goal * error_goal, -0.4, 0.4))
                    # Avance très lentement pour garder la balle proche
                    twist.linear.x = float(self.linear_speed * 0.1)

                # Debug but
                cv2.line(debug, (goal_cx, 0), (goal_cx, h), (0, 0, 255), 2)
                cv2.putText(debug,
                            f"BUT err={error_goal:+d}px "
                            f"posts={goal_info['n_posts']}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if goal_info['post_left'] is not None:
                    cv2.line(debug,
                             (goal_info['post_left'], 0),
                             (goal_info['post_left'], h),
                             (0, 80, 255), 1)
                    cv2.line(debug,
                             (goal_info['post_right'], 0),
                             (goal_info['post_right'], h),
                             (0, 80, 255), 1)
            else:
                # But pas visible → tourne lentement pour le chercher
                # tout en restant près de la balle
                twist.angular.z = 0.25
                cv2.putText(debug, "CHERCHE BUT...", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

            # Debug balle pendant ALIGN
            if detection is not None:
                bx, by, br, _ = detection
                cv2.circle(debug, (bx, by), max(br, 10), (0, 255, 255), 2)

        # ══════════════════════════════════════════════════════════════════
        # PUSH — fonce vers le but avec correction angulaire
        # ══════════════════════════════════════════════════════════════════
        elif self.state == SoccerState.PUSH:
            self._push_timer += dt
            twist.linear.x = float(self.push_speed)

            # Correction angulaire : garde le but au centre si visible
            if goal_info is not None:
                goal_cx    = goal_info['goal_cx']
                error_goal = goal_cx - w // 2
                twist.angular.z = float(np.clip(
                    -self._kp_goal * error_goal * 0.5, -0.4, 0.4))
            elif detection is not None:
                # Sinon suit la balle
                bx, by, br, _ = detection
                error_x = bx - w // 2
                twist.angular.z = float(np.clip(
                    -self._kp_ball * error_x * 0.5, -0.5, 0.5))

            # Sécurité LIDAR
            if self.lidar.front_dist < 0.15:
                twist.linear.x  = 0.0
                twist.angular.z = 0.0
                self.state = SoccerState.DONE
                self.get_logger().info("⚽ STOP mur → DONE")

            if self._push_timer > self.push_duration:
                self.state = SoccerState.DONE
                self.get_logger().info("⚽ DONE !")

        # ══════════════════════════════════════════════════════════════════
        # DONE
        # ══════════════════════════════════════════════════════════════════
        elif self.state == SoccerState.DONE:
            twist.linear.x = 0.0
            cv2.putText(debug, "GOAL !!!!", (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 220, 0), 3)

        self.cmd_pub.publish(twist)

        # ── Overlay debug ────────────────────────────────────────────────
        state_colors = {
            SoccerState.IDLE:       (128, 128, 128),
            SoccerState.SEARCH:     (255, 165, 0),
            SoccerState.APPROACH:   (0,   200, 0),
            SoccerState.ALIGN_GOAL: (0,   0,   255),
            SoccerState.PUSH:       (0,   100, 255),
            SoccerState.DONE:       (0,   220, 0),
        }
        col = state_colors.get(self.state, (255, 255, 255))

        cv2.rectangle(debug, (0, 0), (w, 22), (20, 20, 20), -1)
        cv2.putText(debug, f"SOCCER: {self.state.name}", (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        cv2.putText(debug,
                    f"F={self.lidar.front_dist:.2f}m  "
                    f"L={self.lidar.left_dist:.2f}  R={self.lidar.right_dist:.2f}",
                    (5, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(debug,
                    f"v={twist.linear.x:.2f} w={twist.angular.z:+.2f}  "
                    f"[s]=start [q]=stop",
                    (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Masque balle (coin haut droit)
        if detection is not None:
            _, _, _, ball_mask = detection
            mask_small = cv2.resize(ball_mask, (w // 4, h // 4))
            mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            mask_color[mask_small > 0] = [0, 255, 255]
            debug[0:h // 4, w - w // 4:w] = mask_color

        # Masque but (coin haut gauche)
        if goal_info is not None:
            goal_mask_small = cv2.resize(goal_info['mask'], (w // 4, h // 4))
            goal_color = cv2.cvtColor(goal_mask_small, cv2.COLOR_GRAY2BGR)
            goal_color[goal_mask_small > 0] = [0, 0, 255]
            debug[0:h // 4, 0:w // 4] = goal_color

        cv2.imshow("Challenge4_Soccer", debug)


def main(args=None):
    rclpy.init(args=args)
    node = Challenge4Node()
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
