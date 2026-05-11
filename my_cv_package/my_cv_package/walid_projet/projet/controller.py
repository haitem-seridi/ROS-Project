"""
controller.py — Contrôleur PD pour TurtleBot3 RÉEL
Gains calibrés pour un robot physique (pas simulation).
"""

import numpy as np
from collections import deque


class LaneController:
    """
    Contrôleur PD doux pour robot réel.
    Gains volontairement BAS pour éviter les oscillations.
    """

    def __init__(self,
                 kp: float = 0.002,
                 kd: float = 0.001,
                 kff: float = 5.0,
                 linear_speed: float = 0.06,
                 max_angular: float = 0.8):

        self.kp            = kp
        self.kd            = kd
        self.kff           = kff
        self.linear_speed  = linear_speed
        self.max_angular   = max_angular

        self._last_error   = 0.0
        self._deriv_hist   = deque(maxlen=5)   # lissage dérivée sur 5 frames

    def compute(self,
                error: float,
                curvature: float,
                dt: float,
                in_roundabout: bool = False) -> tuple:
        dt = max(dt, 0.001)

        # Proportionnel
        p_term = self.kp * error

        # Dérivé (très lissé pour éviter le bruit)
        raw_deriv = (error - self._last_error) / dt
        self._deriv_hist.append(raw_deriv)
        d_term = self.kd * float(np.mean(self._deriv_hist))

        # Feedforward courbure (doux)
        smooth_sign = float(np.tanh(error / 30.0))
        ff_term = self.kff * curvature * smooth_sign

        angular = -(p_term + d_term + ff_term)
        angular = float(np.clip(angular, -self.max_angular, self.max_angular))

        # Vitesse adaptative
        error_factor  = 1.0 - min(abs(error) / 200.0, 0.5)
        curv_factor   = 1.0 - min(curvature / 0.010, 0.4)
        speed_factor  = 0.7 if in_roundabout else 1.0
        linear = float(self.linear_speed * error_factor * curv_factor * speed_factor)
        linear = max(linear, 0.02)

        self._last_error = error
        return linear, angular

    def reset(self):
        self._last_error = 0.0
        self._deriv_hist.clear()
