"""
lidar_handler.py — Traitement LIDAR robuste pour TurtleBot3 réel
Utilise angle_min / angle_increment du LaserScan pour calculer les indices
(ne suppose PAS que N=360 ni que 1 index = 1 degré).

Conventions TurtleBot3 LDS-01/LDS-02 :
  0°   = DEVANT
  90°  = GAUCHE  (en convention ROS : counter-clockwise)
  180° = ARRIÈRE
  270° = DROITE
"""

import numpy as np
import math
from collections import deque


class LidarHandler:
    """
    Gère le scan LIDAR 360° du TurtleBot3.
    Adapté pour un robot RÉEL avec gestion complète du bruit.
    """

    def __init__(self,
                 stop_dist: float  = 0.35,
                 avoid_dist: float = 0.55,
                 corridor_target_dist: float = 0.25,
                 lidar_front_index: int = 0):
        self.stop_dist            = stop_dist
        self.avoid_dist           = avoid_dist
        self.corridor_target_dist = corridor_target_dist
        self.lidar_front_index    = int(lidar_front_index)  # force int

        # ── Sorties ──────────────────────────────────────────────────────
        self.obstacle_dist   = 5.0
        self.obstacle_close  = False
        self.lateral_shift   = 0.0

        self.front_dist      = 5.0
        self.front_narrow_dist = 5.0
        self.left_dist       = 5.0
        self.right_dist      = 5.0
        self.front_left_dist  = 5.0
        self.front_right_dist = 5.0

        self.front_min_dist       = 5.0
        self.front_narrow_min_dist = 5.0
        self.left_min_dist        = 5.0
        self.right_min_dist       = 5.0
        self.front_left_min_dist  = 5.0
        self.front_right_min_dist = 5.0

        # ── Lissage EMA ──────────────────────────────────────────────────
        self._alpha = 0.4
        self._shift_hist = deque(maxlen=5)
        self._has_measurement = False

        # ── Range valide ─────────────────────────────────────────────────
        self.R_MIN = 0.10   # LDS-01=0.12, LDS-02=0.05 → on prend 0.10 pour être safe
        self.R_MAX = 3.5

        # ── Diagnostic ───────────────────────────────────────────────────
        self._msg_count = 0
        self._N = 0
        self._angle_min = 0.0
        self._angle_inc = 0.0

    # ─── Conversion angle (radians) → index dans le tableau ranges ───────────
    def _angle_to_index(self, angle_rad, N):
        """
        Convertit un angle en radians vers l'index correspondant dans ranges[].
        Utilise angle_min et angle_increment du dernier message.
        """
        if self._angle_inc == 0:
            # Fallback si pas encore reçu de message
            return int(round(math.degrees(angle_rad))) % N

        idx = int(round((angle_rad - self._angle_min) / self._angle_inc))
        return idx % N

    # ─── Extraction robuste d'une zone angulaire ─────────────────────────────
    def _safe_zone(self, ranges, center_rad, half_width_rad, N,
                   use_median=True):
        """
        Extrait une mesure fiable dans un arc angulaire.
        center_rad     : centre de l'arc en RADIANS
        half_width_rad : demi-largeur en RADIANS
        """
        if N == 0:
            return self.R_MAX

        lo_angle = center_rad - half_width_rad
        hi_angle = center_rad + half_width_rad

        lo = self._angle_to_index(lo_angle, N)
        hi = self._angle_to_index(hi_angle, N)

        # Extraire le segment (gère le wraparound)
        if lo <= hi:
            seg = ranges[lo:hi+1]
        else:
            seg = np.concatenate([ranges[lo:], ranges[:hi+1]])

        if len(seg) == 0:
            return self.R_MAX

        # Filtrer les valeurs invalides (0, inf, nan, hors range)
        valid = seg[np.isfinite(seg) & (seg > self.R_MIN) & (seg < self.R_MAX)]

        if len(valid) == 0:
            return self.R_MAX

        if use_median:
            return float(np.median(valid))
        else:
            return float(np.min(valid))

    # ─── Lissage EMA ─────────────────────────────────────────────────────────
    def _ema(self, old_val, new_val):
        if not self._has_measurement:
            return new_val
        return self._alpha * new_val + (1.0 - self._alpha) * old_val

    # ─── Mise à jour principale ──────────────────────────────────────────────
    def update(self, msg):
        """Appelé dans scan_callback avec le message LaserScan."""
        ranges = np.array(msg.ranges, dtype=np.float64)
        N = len(ranges)
        if N == 0:
            return

        # Stocker les paramètres du message pour la conversion angle→index
        self._angle_min = msg.angle_min
        self._angle_inc = msg.angle_increment
        self._N = N

        # Diagnostic (premier message)
        self._msg_count += 1

        # ── Angles en radians ────────────────────────────────────────────
        # Convention ROS : 0 = devant, π/2 = gauche, π = arrière, 3π/2 = droite
        # Si lidar_front_index != 0, on ajoute un offset
        offset = math.radians(self.lidar_front_index)

        FRONT  = 0.0 + offset
        LEFT   = math.pi / 2.0 + offset
        RIGHT  = -math.pi / 2.0 + offset
        FRONT_LEFT  = math.pi / 4.0 + offset
        FRONT_RIGHT = -math.pi / 4.0 + offset

        # Demi-largeurs des arcs
        FRONT_HW = math.radians(20)   # ±20°
        FRONT_NARROW_HW = math.radians(6)
        SIDE_HW  = math.radians(20)   # ±20°
        DIAG_HW  = math.radians(15)   # ±15°

        # ── Distances directionnelles ────────────────────────────────────
        d_front = self._safe_zone(ranges, FRONT, FRONT_HW, N, use_median=False)
        self.front_min_dist = d_front
        self.front_dist = self._ema(self.front_dist, d_front)

        d_front_narrow = self._safe_zone(ranges, FRONT, FRONT_NARROW_HW, N, use_median=True)
        self.front_narrow_min_dist = self._safe_zone(
            ranges, FRONT, FRONT_NARROW_HW, N, use_median=False)
        self.front_narrow_dist = self._ema(self.front_narrow_dist, d_front_narrow)

        d_left = self._safe_zone(ranges, LEFT, SIDE_HW, N, use_median=True)
        self.left_min_dist = self._safe_zone(ranges, LEFT, SIDE_HW, N, use_median=False)
        self.left_dist = self._ema(self.left_dist, d_left)

        d_right = self._safe_zone(ranges, RIGHT, SIDE_HW, N, use_median=True)
        self.right_min_dist = self._safe_zone(ranges, RIGHT, SIDE_HW, N, use_median=False)
        self.right_dist = self._ema(self.right_dist, d_right)

        d_fl = self._safe_zone(ranges, FRONT_LEFT, DIAG_HW, N, use_median=True)
        self.front_left_min_dist = self._safe_zone(ranges, FRONT_LEFT, DIAG_HW, N, use_median=False)
        self.front_left_dist = self._ema(self.front_left_dist, d_fl)

        d_fr = self._safe_zone(ranges, FRONT_RIGHT, DIAG_HW, N, use_median=True)
        self.front_right_min_dist = self._safe_zone(ranges, FRONT_RIGHT, DIAG_HW, N, use_median=False)
        self.front_right_dist = self._ema(self.front_right_dist, d_fr)

        # ── Obstacle ─────────────────────────────────────────────────────
        self.obstacle_dist  = self.front_dist
        self.obstacle_close = self.front_dist < self.stop_dist

        # ── Shift latéral pour Challenge 2 ───────────────────────────────
        shift = 0.0
        if self.front_dist < self.avoid_dist:
            if self.front_right_dist < self.front_left_dist:
                intensity = (self.avoid_dist - self.front_right_dist) / self.avoid_dist
                shift = -intensity * 200.0
            else:
                intensity = (self.avoid_dist - self.front_left_dist) / self.avoid_dist
                shift = intensity * 200.0

        self._shift_hist.append(shift)
        self.lateral_shift = self._ema(self.lateral_shift, shift)
        self._has_measurement = True

    # ─── Diagnostic pour debug ───────────────────────────────────────────────
    def diag_str(self) -> str:
        """Retourne une string de diagnostic détaillée."""
        return (f"N={self._N} angle_min={math.degrees(self._angle_min):.1f}° "
                f"angle_inc={math.degrees(self._angle_inc):.3f}° "
                f"msgs={self._msg_count}")

    def status_str(self) -> str:
        return (f"L={self.left_dist:.2f}m "
                f"FL={self.front_left_dist:.2f}m "
                f"F={self.front_dist:.2f}m Fn={self.front_narrow_dist:.2f}m "
                f"FR={self.front_right_dist:.2f}m "
                f"R={self.right_dist:.2f}m "
                f"shift={self.lateral_shift:+.0f}px")
