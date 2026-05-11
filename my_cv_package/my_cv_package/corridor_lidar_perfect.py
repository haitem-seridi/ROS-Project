# #!/usr/bin/env python3
# """
# Challenge 3 — LIDAR-only wall-centered corridor / long U-turn navigation.

# This version does NOT spin in place.

# Behavior:
# - Detects LEFT wall and RIGHT wall using /scan.
# - Tries to stay in the middle between the two walls.
# - Looks ahead with front-left / front-right / open-angle detection.
# - Moves forward continuously.
# - Turns smoothly little by little through the long U-turn.

# Use this only for the wall corridor part.
# Do not run line_follower.py at the same time unless you use a cmd_vel mux.
# """

# import math
# import statistics

# import rclpy
# from rclpy.node import Node

# from sensor_msgs.msg import LaserScan
# from geometry_msgs.msg import Twist


# SCAN_TOPIC = "/scan"
# CMD_TOPIC = "/cmd_vel"


# # ═══════════════════════════════════════════════════════════════════
# #  MAIN TUNING
# # ═══════════════════════════════════════════════════════════════════

# # Start with auto.
# # If it chooses wrong side in your U-turn, force "left" or "right".
# U_TURN_DIRECTION = "auto"   # "auto", "left", "right"

# # If angular direction is reversed in your sim, set True.
# INVERT_ANGULAR = False

# # Forward speeds
# SPEED_NORMAL = 0.075
# SPEED_CURVE = 0.060
# SPEED_DANGER = 0.035

# # LIDAR wall centering
# DESIRED_WALL_DIST = 0.30

# # Valid wall distance limits
# SIDE_MIN_VALID = 0.06
# SIDE_MAX_VALID = 1.40

# # If front is closer than this, reduce speed and turn more
# FRONT_SLOW_DIST = 0.65

# # If front is very close, still move forward but slowly
# FRONT_DANGER_DIST = 0.22

# # Wall-centering gain:
# # left - right.
# # If left is smaller than right, angular becomes negative, robot turns right.
# KP_CENTER = 0.95

# # Lookahead gain:
# # front_left - front_right.
# # If front_left is more open, angular becomes positive, robot turns left.
# KP_LOOKAHEAD = 0.65

# # Open corridor direction gain:
# # best open angle from laser scan.
# KP_OPEN_ANGLE = 0.95

# # One-wall fallback gain
# KP_ONE_WALL = 0.85

# # Maximum angular command
# ANG_MAX = 0.32

# # Smooth angular command
# ANG_SMOOTH = 0.70

# # Emergency side protection
# SIDE_DANGER_DIST = 0.13
# SIDE_DANGER_PUSH = 0.16

# # Open direction search
# OPEN_SEARCH_MIN_DEG = -105
# OPEN_SEARCH_MAX_DEG = 105
# OPEN_SEARCH_STEP_DEG = 5
# OPEN_SECTOR_WIDTH_DEG = 8

# # Prefer not turning too much unless needed
# OPEN_FORWARD_BONUS = 0.18
# OPEN_SIDE_PENALTY = 0.0018

# # Debug
# DEBUG_EVERY_N_FRAMES = 8


# class CorridorLidar(Node):
#     def __init__(self):
#         super().__init__("corridor_lidar")

#         self.sub = self.create_subscription(
#             LaserScan,
#             SCAN_TOPIC,
#             self.scan_callback,
#             10
#         )

#         self.pub = self.create_publisher(
#             Twist,
#             CMD_TOPIC,
#             10
#         )

#         self.frame_count = 0
#         self.filtered_angular = 0.0

#         # Used only in auto mode to avoid rapid left/right flipping
#         self.preferred_turn_sign = 0.0

#         self.get_logger().info("LIDAR wall-centered corridor follower ready")
#         self.get_logger().info(f"Subscribing to {SCAN_TOPIC}")
#         self.get_logger().info(f"Publishing to {CMD_TOPIC}")
#         self.get_logger().info("This node uses /scan only.")

#     # ───────────────────────────────────────────────────────────────
#     def clamp(self, value, low, high):
#         return max(low, min(high, value))

#     def normalize_angle(self, angle):
#         while angle > math.pi:
#             angle -= 2.0 * math.pi
#         while angle < -math.pi:
#             angle += 2.0 * math.pi
#         return angle

#     def angle_to_index(self, msg, angle_rad):
#         angle = self.normalize_angle(angle_rad)

#         if angle < msg.angle_min:
#             angle += 2.0 * math.pi

#         if angle > msg.angle_max:
#             angle -= 2.0 * math.pi

#         idx = int(round((angle - msg.angle_min) / msg.angle_increment))
#         idx = max(0, min(idx, len(msg.ranges) - 1))
#         return idx

#     def valid_range(self, value, msg):
#         if value is None:
#             return False
#         if math.isnan(value) or math.isinf(value):
#             return False
#         if value < max(msg.range_min, 0.03):
#             return False
#         if value > msg.range_max:
#             return False
#         return True

#     def get_sector_values(self, msg, center_deg, width_deg):
#         values = []
#         half = width_deg // 2

#         for deg in range(center_deg - half, center_deg + half + 1):
#             idx = self.angle_to_index(msg, math.radians(deg))
#             value = msg.ranges[idx]

#             if self.valid_range(value, msg):
#                 values.append(float(value))

#         return values

#     def get_sector_median(self, msg, center_deg, width_deg):
#         values = self.get_sector_values(msg, center_deg, width_deg)

#         if not values:
#             return None

#         return statistics.median(values)

#     def get_sector_percentile(self, msg, center_deg, width_deg, percentile=35):
#         """
#         Percentile is better than pure median for wall distance.
#         It catches the closer wall surface without being too noisy.
#         """
#         values = self.get_sector_values(msg, center_deg, width_deg)

#         if not values:
#             return None

#         values.sort()
#         index = int((percentile / 100.0) * (len(values) - 1))
#         return values[index]

#     def is_side_valid(self, value):
#         if value is None:
#             return False
#         return SIDE_MIN_VALID <= value <= SIDE_MAX_VALID

#     def apply_angular_direction(self, angular):
#         return -angular if INVERT_ANGULAR else angular

#     # ───────────────────────────────────────────────────────────────
#     def get_left_wall_distance(self, msg):
#         """
#         Wide left sector, not only 90 degrees.
#         This helps in the curved U-turn where walls are not perfectly side-on.
#         """
#         candidates = []

#         for angle in [55, 70, 85, 100, 115]:
#             d = self.get_sector_percentile(msg, angle, 16, percentile=35)
#             if self.is_side_valid(d):
#                 candidates.append(d)

#         if not candidates:
#             return None

#         return statistics.median(candidates)

#     def get_right_wall_distance(self, msg):
#         """
#         Wide right sector, not only -90 degrees.
#         """
#         candidates = []

#         for angle in [-55, -70, -85, -100, -115]:
#             d = self.get_sector_percentile(msg, angle, 16, percentile=35)
#             if self.is_side_valid(d):
#                 candidates.append(d)

#         if not candidates:
#             return None

#         return statistics.median(candidates)

#     def get_best_open_angle(self, msg):
#         """
#         Finds the open direction in front of the robot.

#         Positive angle = open space is left.
#         Negative angle = open space is right.
#         """
#         best_angle = 0.0
#         best_score = -999.0

#         if U_TURN_DIRECTION == "left":
#             angle_range = range(0, OPEN_SEARCH_MAX_DEG + 1, OPEN_SEARCH_STEP_DEG)
#         elif U_TURN_DIRECTION == "right":
#             angle_range = range(OPEN_SEARCH_MIN_DEG, 1, OPEN_SEARCH_STEP_DEG)
#         else:
#             angle_range = range(
#                 OPEN_SEARCH_MIN_DEG,
#                 OPEN_SEARCH_MAX_DEG + 1,
#                 OPEN_SEARCH_STEP_DEG
#             )

#         for deg in angle_range:
#             d = self.get_sector_median(msg, deg, OPEN_SECTOR_WIDTH_DEG)

#             if d is None:
#                 continue

#             # Prefer open distance, but still prefer forward if distances are similar.
#             forward_bonus = OPEN_FORWARD_BONUS * math.cos(math.radians(deg))
#             side_penalty = OPEN_SIDE_PENALTY * abs(deg)

#             score = d + forward_bonus - side_penalty

#             if score > best_score:
#                 best_score = score
#                 best_angle = float(deg)

#         return math.radians(best_angle), best_angle, best_score

#     def update_preferred_turn(self, open_angle_deg, front_left, front_right):
#         """
#         Prevents auto mode from flipping left/right every scan.
#         """
#         if U_TURN_DIRECTION == "left":
#             self.preferred_turn_sign = +1.0
#             return

#         if U_TURN_DIRECTION == "right":
#             self.preferred_turn_sign = -1.0
#             return

#         # In auto mode, update preference only when direction is clear.
#         if abs(open_angle_deg) > 18:
#             self.preferred_turn_sign = +1.0 if open_angle_deg > 0 else -1.0
#             return

#         if front_left is not None and front_right is not None:
#             diff = front_left - front_right
#             if abs(diff) > 0.12:
#                 self.preferred_turn_sign = +1.0 if diff > 0 else -1.0

#     # ───────────────────────────────────────────────────────────────
#     def compute_wall_center_angular(self, left_wall, right_wall):
#         """
#         Tries to stay exactly between two walls.
#         """
#         left_valid = self.is_side_valid(left_wall)
#         right_valid = self.is_side_valid(right_wall)

#         if left_valid and right_valid:
#             center_error = left_wall - right_wall
#             return KP_CENTER * center_error, "both"

#         if left_valid and not right_valid:
#             # Too close to left -> negative angular -> steer right.
#             error = left_wall - DESIRED_WALL_DIST
#             return KP_ONE_WALL * error, "left_only"

#         if right_valid and not left_valid:
#             # Too close to right -> positive angular -> steer left.
#             error = DESIRED_WALL_DIST - right_wall
#             return KP_ONE_WALL * error, "right_only"

#         return 0.0, "none"

#     def compute_speed(self, front, angular):
#         """
#         Always moves forward, but slows down in tight curve/front wall.
#         """
#         if front is not None and front < FRONT_DANGER_DIST:
#             return SPEED_DANGER

#         if front is not None and front < FRONT_SLOW_DIST:
#             return SPEED_CURVE

#         if abs(angular) > 0.22:
#             return SPEED_CURVE

#         return SPEED_NORMAL

#     # ───────────────────────────────────────────────────────────────
#     def scan_callback(self, msg):
#         cmd = Twist()

#         # Core distances
#         front = self.get_sector_median(msg, 0, 20)
#         front_left = self.get_sector_median(msg, 35, 24)
#         front_right = self.get_sector_median(msg, -35, 24)

#         left_wall = self.get_left_wall_distance(msg)
#         right_wall = self.get_right_wall_distance(msg)

#         open_angle_rad, open_angle_deg, open_score = self.get_best_open_angle(msg)
#         self.update_preferred_turn(open_angle_deg, front_left, front_right)

#         # 1) Stay between walls
#         angular_center, wall_mode = self.compute_wall_center_angular(
#             left_wall,
#             right_wall
#         )

#         # 2) Look ahead: turn toward the more open part of the corridor
#         angular_lookahead = 0.0

#         if front_left is not None and front_right is not None:
#             angular_lookahead = KP_LOOKAHEAD * (front_left - front_right)

#         # 3) Open-angle steering becomes stronger when front is close
#         open_gain = KP_OPEN_ANGLE

#         if front is not None and front > FRONT_SLOW_DIST:
#             open_gain *= 0.35

#         angular_open = open_gain * open_angle_rad

#         # 4) If front is close and auto has a preferred U-turn side,
#         # add a small bias to keep turning consistently.
#         angular_bias = 0.0

#         if front is not None and front < FRONT_SLOW_DIST:
#             if self.preferred_turn_sign != 0.0:
#                 angular_bias = 0.07 * self.preferred_turn_sign

#         # Raw command
#         raw_angular = (
#             angular_center
#             + angular_lookahead
#             + angular_open
#             + angular_bias
#         )

#         # Gentle side emergency protection
#         if left_wall is not None and left_wall < SIDE_DANGER_DIST:
#             raw_angular -= SIDE_DANGER_PUSH

#         if right_wall is not None and right_wall < SIDE_DANGER_DIST:
#             raw_angular += SIDE_DANGER_PUSH

#         raw_angular = self.clamp(raw_angular, -ANG_MAX, ANG_MAX)

#         # Smooth angular
#         self.filtered_angular = (
#             ANG_SMOOTH * self.filtered_angular
#             + (1.0 - ANG_SMOOTH) * raw_angular
#         )

#         cmd.angular.z = self.apply_angular_direction(self.filtered_angular)
#         cmd.linear.x = self.compute_speed(front, cmd.angular.z)

#         # If scan is completely useless, stop gently
#         if front is None and left_wall is None and right_wall is None:
#             cmd.linear.x = 0.0
#             cmd.angular.z = 0.0

#         self.pub.publish(cmd)

#         self.print_debug(
#             front,
#             front_left,
#             front_right,
#             left_wall,
#             right_wall,
#             open_angle_deg,
#             wall_mode,
#             cmd
#         )

#     # ───────────────────────────────────────────────────────────────
#     def print_debug(
#         self,
#         front,
#         front_left,
#         front_right,
#         left_wall,
#         right_wall,
#         open_angle_deg,
#         wall_mode,
#         cmd
#     ):
#         self.frame_count += 1

#         if self.frame_count % DEBUG_EVERY_N_FRAMES != 0:
#             return

#         pref = "NONE"
#         if self.preferred_turn_sign > 0:
#             pref = "LEFT"
#         elif self.preferred_turn_sign < 0:
#             pref = "RIGHT"

#         self.get_logger().info(
#             "front={} fl={} fr={} left={} right={} wall={} open={:.0f} pref={} cmd=({:.3f}, {:.3f})".format(
#                 self.fmt(front),
#                 self.fmt(front_left),
#                 self.fmt(front_right),
#                 self.fmt(left_wall),
#                 self.fmt(right_wall),
#                 wall_mode,
#                 open_angle_deg,
#                 pref,
#                 cmd.linear.x,
#                 cmd.angular.z
#             )
#         )

#     def fmt(self, value):
#         if value is None:
#             return "None"
#         return f"{value:.2f}"


# def main(args=None):
#     rclpy.init(args=args)

#     node = CorridorLidar()

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass

#     stop = Twist()

#     for _ in range(10):
#         node.pub.publish(stop)
#         rclpy.spin_once(node, timeout_sec=0.02)

#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
Challenge 3 — LIDAR-only wall-centered corridor navigation.

CORRECTIONS par rapport à l'ancienne version :
  1. WARM-UP : les 15 premières trames sont ignorées → plus de démarrage brutal
  2. Vitesse progressive : rampe de 0 → SPEED_NORMAL sur 2 secondes
  3. ANG_SMOOTH réduit à 0.45 → moins d'inertie dans les virages serrés
  4. SIDE_MAX_VALID réduit à 0.80 → ignore les gros obstacles lointains
  5. open_angle désactivé quand les DEUX murs sont bien visibles (mode "both")
     → évite que le 3e gain contredise le centrage dans le tunnel droit
"""

import math
import statistics
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


SCAN_TOPIC = "/scan"
CMD_TOPIC  = "/cmd_vel"

# ═══════════════════════════════════════════════════════════════════
#  PARAMÈTRES — MODIFIÉS PAR RAPPORT À L'ANCIENNE VERSION
# ═══════════════════════════════════════════════════════════════════

U_TURN_DIRECTION = "auto"   # "auto", "left", "right"
INVERT_ANGULAR   = False

# Vitesses
SPEED_NORMAL = 0.075
SPEED_CURVE  = 0.060
SPEED_DANGER = 0.035

# Distances murs
DESIRED_WALL_DIST = 0.30

# ── FIX 4 : SIDE_MAX_VALID 1.40 → 0.80 ─────────────────────────────
# Avant : 1.40m → les grands obstacles latéraux étaient traités comme murs
# Maintenant : 0.80m → seules les surfaces vraiment proches comptent
SIDE_MIN_VALID = 0.06
SIDE_MAX_VALID = 0.80   # ← MODIFIÉ (était 1.40)

FRONT_SLOW_DIST   = 0.65
FRONT_DANGER_DIST = 0.22

# Gains de commande
KP_CENTER     = 0.95
KP_LOOKAHEAD  = 0.65
KP_OPEN_ANGLE = 0.95
KP_ONE_WALL   = 0.85

ANG_MAX = 0.32

# ── FIX 3 : ANG_SMOOTH 0.70 → 0.45 ─────────────────────────────────
# Avant : 0.70 → trop d'inertie → oscillations dans le tunnel
# Maintenant : 0.45 → plus réactif, se corrige plus vite
ANG_SMOOTH = 0.45   # ← MODIFIÉ (était 0.70)

SIDE_DANGER_DIST = 0.13
SIDE_DANGER_PUSH = 0.16

OPEN_SEARCH_MIN_DEG    = -105
OPEN_SEARCH_MAX_DEG    = 105
OPEN_SEARCH_STEP_DEG   = 5
OPEN_SECTOR_WIDTH_DEG  = 8
OPEN_FORWARD_BONUS     = 0.18
OPEN_SIDE_PENALTY      = 0.0018

# ── FIX 1 & 2 : WARM-UP ─────────────────────────────────────────────
# Nombre de trames ignorées au démarrage (évite le départ brutal)
WARMUP_FRAMES = 15   # ← NOUVEAU
# Durée de la rampe de vitesse (secondes)
RAMP_DURATION = 2.0  # ← NOUVEAU

DEBUG_EVERY_N_FRAMES = 8


class CorridorLidar(Node):
    def __init__(self):
        super().__init__("corridor_lidar")

        self.sub = self.create_subscription(
            LaserScan, SCAN_TOPIC, self.scan_callback, 10)
        self.pub = self.create_publisher(
            Twist, CMD_TOPIC, 10)

        self.frame_count       = 0
        self.filtered_angular  = 0.0
        self.preferred_turn_sign = 0.0

        # ── FIX 1 & 2 : état warm-up ────────────────────────────────
        self._warmup_done  = False
        self._start_time   = None   # mis à jour après le warm-up

        self.get_logger().info("LIDAR corridor follower (version corrigée) ready")
        self.get_logger().info(
            f"Warm-up sur {WARMUP_FRAMES} trames puis rampe {RAMP_DURATION}s")

    # ───────────────────────────────────────────────────────────────
    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def angle_to_index(self, msg, angle_rad):
        angle = self.normalize_angle(angle_rad)
        if angle < msg.angle_min:
            angle += 2.0 * math.pi
        if angle > msg.angle_max:
            angle -= 2.0 * math.pi
        idx = int(round((angle - msg.angle_min) / msg.angle_increment))
        return max(0, min(idx, len(msg.ranges) - 1))

    def valid_range(self, value, msg):
        if value is None:
            return False
        if math.isnan(value) or math.isinf(value):
            return False
        if value < max(msg.range_min, 0.03):
            return False
        if value > msg.range_max:
            return False
        return True

    def get_sector_values(self, msg, center_deg, width_deg):
        values = []
        half = width_deg // 2
        for deg in range(center_deg - half, center_deg + half + 1):
            idx   = self.angle_to_index(msg, math.radians(deg))
            value = msg.ranges[idx]
            if self.valid_range(value, msg):
                values.append(float(value))
        return values

    def get_sector_median(self, msg, center_deg, width_deg):
        values = self.get_sector_values(msg, center_deg, width_deg)
        return statistics.median(values) if values else None

    def get_sector_percentile(self, msg, center_deg, width_deg, percentile=35):
        values = self.get_sector_values(msg, center_deg, width_deg)
        if not values:
            return None
        values.sort()
        index = int((percentile / 100.0) * (len(values) - 1))
        return values[index]

    def is_side_valid(self, value):
        if value is None:
            return False
        return SIDE_MIN_VALID <= value <= SIDE_MAX_VALID

    def apply_angular_direction(self, angular):
        return -angular if INVERT_ANGULAR else angular

    # ───────────────────────────────────────────────────────────────
    def get_left_wall_distance(self, msg):
        candidates = []
        for angle in [55, 70, 85, 100, 115]:
            d = self.get_sector_percentile(msg, angle, 16, percentile=35)
            if self.is_side_valid(d):
                candidates.append(d)
        return statistics.median(candidates) if candidates else None

    def get_right_wall_distance(self, msg):
        candidates = []
        for angle in [-55, -70, -85, -100, -115]:
            d = self.get_sector_percentile(msg, angle, 16, percentile=35)
            if self.is_side_valid(d):
                candidates.append(d)
        return statistics.median(candidates) if candidates else None

    def get_best_open_angle(self, msg):
        best_angle = 0.0
        best_score = -999.0

        if U_TURN_DIRECTION == "left":
            angle_range = range(0, OPEN_SEARCH_MAX_DEG + 1, OPEN_SEARCH_STEP_DEG)
        elif U_TURN_DIRECTION == "right":
            angle_range = range(OPEN_SEARCH_MIN_DEG, 1, OPEN_SEARCH_STEP_DEG)
        else:
            angle_range = range(
                OPEN_SEARCH_MIN_DEG,
                OPEN_SEARCH_MAX_DEG + 1,
                OPEN_SEARCH_STEP_DEG)

        for deg in angle_range:
            d = self.get_sector_median(msg, deg, OPEN_SECTOR_WIDTH_DEG)
            if d is None:
                continue
            forward_bonus = OPEN_FORWARD_BONUS * math.cos(math.radians(deg))
            side_penalty  = OPEN_SIDE_PENALTY * abs(deg)
            score = d + forward_bonus - side_penalty
            if score > best_score:
                best_score = score
                best_angle = float(deg)

        return math.radians(best_angle), best_angle, best_score

    def update_preferred_turn(self, open_angle_deg, front_left, front_right):
        if U_TURN_DIRECTION == "left":
            self.preferred_turn_sign = +1.0
            return
        if U_TURN_DIRECTION == "right":
            self.preferred_turn_sign = -1.0
            return
        if abs(open_angle_deg) > 18:
            self.preferred_turn_sign = +1.0 if open_angle_deg > 0 else -1.0
            return
        if front_left is not None and front_right is not None:
            diff = front_left - front_right
            if abs(diff) > 0.12:
                self.preferred_turn_sign = +1.0 if diff > 0 else -1.0

    # ───────────────────────────────────────────────────────────────
    def compute_wall_center_angular(self, left_wall, right_wall):
        left_valid  = self.is_side_valid(left_wall)
        right_valid = self.is_side_valid(right_wall)

        if left_valid and right_valid:
            center_error = left_wall - right_wall
            return KP_CENTER * center_error, "both"

        if left_valid and not right_valid:
            error = left_wall - DESIRED_WALL_DIST
            return KP_ONE_WALL * error, "left_only"

        if right_valid and not left_valid:
            error = DESIRED_WALL_DIST - right_wall
            return KP_ONE_WALL * error, "right_only"

        return 0.0, "none"

    def compute_speed(self, front, angular):
        """
        Vitesse de base × facteur de rampe.
        La rampe monte de 0 → 1.0 sur RAMP_DURATION secondes.
        """
        if front is not None and front < FRONT_DANGER_DIST:
            base = SPEED_DANGER
        elif front is not None and front < FRONT_SLOW_DIST:
            base = SPEED_CURVE
        elif abs(angular) > 0.22:
            base = SPEED_CURVE
        else:
            base = SPEED_NORMAL

        # ── FIX 2 : rampe de vitesse ─────────────────────────────────
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            ramp    = min(1.0, elapsed / RAMP_DURATION)
        else:
            ramp = 1.0

        return base * ramp

    # ───────────────────────────────────────────────────────────────
    def scan_callback(self, msg):
        self.frame_count += 1

        # ── FIX 1 : warm-up — ignorer les premières trames ───────────
        if not self._warmup_done:
            if self.frame_count < WARMUP_FRAMES:
                # Publier stop pendant le warm-up
                self.pub.publish(Twist())
                return
            # Warm-up terminé : noter l'heure de départ réel
            self._warmup_done = True
            self._start_time  = time.time()
            self.get_logger().info(
                f"Warm-up terminé ({WARMUP_FRAMES} trames) → démarrage !")

        cmd = Twist()

        front       = self.get_sector_median(msg, 0, 20)
        front_left  = self.get_sector_median(msg, 35, 24)
        front_right = self.get_sector_median(msg, -35, 24)
        left_wall   = self.get_left_wall_distance(msg)
        right_wall  = self.get_right_wall_distance(msg)

        open_angle_rad, open_angle_deg, open_score = self.get_best_open_angle(msg)
        self.update_preferred_turn(open_angle_deg, front_left, front_right)

        # 1) Centrage entre les murs
        angular_center, wall_mode = self.compute_wall_center_angular(
            left_wall, right_wall)

        # 2) Lookahead
        angular_lookahead = 0.0
        if front_left is not None and front_right is not None:
            angular_lookahead = KP_LOOKAHEAD * (front_left - front_right)

        # ── FIX 5 : open_angle désactivé si les deux murs sont visibles ─
        # Avant : les 3 gains (center + lookahead + open) s'additionnaient
        #         et se contredisaient dans un tunnel droit
        # Maintenant : si wall_mode == "both", open_angle = 0
        #              → seul le centrage murs pilote le robot dans le tunnel
        if wall_mode == "both":
            angular_open = 0.0
        else:
            open_gain = KP_OPEN_ANGLE
            if front is not None and front > FRONT_SLOW_DIST:
                open_gain *= 0.35
            angular_open = open_gain * open_angle_rad

        # 4) Biais U-turn
        angular_bias = 0.0
        if front is not None and front < FRONT_SLOW_DIST:
            if self.preferred_turn_sign != 0.0:
                angular_bias = 0.07 * self.preferred_turn_sign

        raw_angular = (
            angular_center
            + angular_lookahead
            + angular_open
            + angular_bias
        )

        # Urgence latérale
        if left_wall is not None and left_wall < SIDE_DANGER_DIST:
            raw_angular -= SIDE_DANGER_PUSH
        if right_wall is not None and right_wall < SIDE_DANGER_DIST:
            raw_angular += SIDE_DANGER_PUSH

        raw_angular = self.clamp(raw_angular, -ANG_MAX, ANG_MAX)

        # ── FIX 3 : ANG_SMOOTH réduit à 0.45 ────────────────────────
        self.filtered_angular = (
            ANG_SMOOTH * self.filtered_angular
            + (1.0 - ANG_SMOOTH) * raw_angular
        )

        cmd.angular.z = self.apply_angular_direction(self.filtered_angular)
        cmd.linear.x  = self.compute_speed(front, cmd.angular.z)

        if front is None and left_wall is None and right_wall is None:
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0

        self.pub.publish(cmd)

        self.print_debug(
            front, front_left, front_right,
            left_wall, right_wall,
            open_angle_deg, wall_mode, cmd)

    # ───────────────────────────────────────────────────────────────
    def print_debug(self, front, front_left, front_right,
                    left_wall, right_wall,
                    open_angle_deg, wall_mode, cmd):
        if self.frame_count % DEBUG_EVERY_N_FRAMES != 0:
            return

        pref = "NONE"
        if self.preferred_turn_sign > 0:
            pref = "LEFT"
        elif self.preferred_turn_sign < 0:
            pref = "RIGHT"

        self.get_logger().info(
            "front={} fl={} fr={} left={} right={} wall={} open={:.0f} "
            "pref={} cmd=({:.3f}, {:.3f})".format(
                self.fmt(front), self.fmt(front_left), self.fmt(front_right),
                self.fmt(left_wall), self.fmt(right_wall),
                wall_mode, open_angle_deg, pref,
                cmd.linear.x, cmd.angular.z))

    def fmt(self, value):
        return "None" if value is None else f"{value:.2f}"


def main(args=None):
    rclpy.init(args=args)
    node = CorridorLidar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    stop = Twist()
    for _ in range(10):
        node.pub.publish(stop)
        rclpy.spin_once(node, timeout_sec=0.02)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
