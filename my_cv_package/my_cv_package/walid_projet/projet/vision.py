"""
vision.py — Traitement image SIMPLE et ROBUSTE pour TurtleBot3 réel
PAS DE BEV / PAS DE HOMOGRAPHIE / PAS DE CALIBRATION NÉCESSAIRE

Approche directe :
  1. Prendre le bas de l'image caméra (la route proche du robot)
  2. Masques HSV pour ligne verte (gauche) et rouge (droite)
  3. Position X moyenne de chaque ligne (pondérée vers le bas = proche)
  4. Erreur = milieu des deux lignes - centre image
  5. Courbure estimée par différence haut/bas

Cette approche marche sur N'IMPORTE QUELLE caméra, résolution, angle de montage.
"""

import cv2
import numpy as np
from collections import deque


class LaneVision:
    """
    Pipeline vision SIMPLE sans Bird's Eye View.
    Fonctionne directement sur l'image caméra brute.
    """

    def __init__(self, img_width=640, img_height=480, seuil_hsv=30):
        self.seuil_hsv = seuil_hsv
        self.img_w = img_width
        self.img_h = img_height

        # CLAHE pour normaliser la luminosité
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Lissage de l'erreur
        self._error_history = deque(maxlen=8)
        self._curvature_history = deque(maxlen=12)

        # Dernières positions valides des lignes
        self._last_green_x = None
        self._last_red_x = None

        # Compteur pour log
        self._frame_count = 0

    # ─── Prétraitement ────────────────────────────────────────────────────────
    def preprocess(self, frame):
        """CLAHE sur la luminance pour robustesse à la lumière."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # ─── Masques HSV ──────────────────────────────────────────────────────────
    def get_masks(self, frame_region):
        """Crée les masques HSV pour les lignes verte et rouge."""
        hsv = cv2.cvtColor(frame_region, cv2.COLOR_BGR2HSV)
        s = max(10, self.seuil_hsv)

        # Vert (ligne gauche) : Hue ~60, tolérance large
        mask_g = cv2.inRange(hsv,
            np.array([max(0, 60 - s), 30, 30]),
            np.array([min(179, 60 + s), 255, 255]))

        # Rouge (ligne droite) : Hue ~0 et ~180 (circulaire)
        s_r = max(10, s // 2)
        mask_r = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,        40, 40]), np.array([s_r,  255, 255])),
            cv2.inRange(hsv, np.array([180 - s_r, 40, 40]), np.array([180,  255, 255]))
        )

        # Morphologie : nettoyage
        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        for mask in [mask_g, mask_r]:
            cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, dst=mask)
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, dst=mask)

        return mask_g, mask_r

    # ─── Trouver la position X d'une ligne ────────────────────────────────────
    def _find_line_x(self, mask, fallback_x):
        """
        Trouve la position X moyenne d'une ligne dans le masque.
        Pondère les pixels vers le BAS de l'image (plus proches du robot).
        """
        ys, xs = np.where(mask > 0)
        if len(xs) < 30:
            return fallback_x, False

        # Pondération : les pixels en bas (grand y) comptent plus
        h = mask.shape[0]
        weights = (ys.astype(float) / h) ** 2  # carré pour accentuer le bas
        weights_sum = weights.sum()
        if weights_sum < 1e-6:
            return fallback_x, False

        x_avg = float(np.sum(xs * weights) / weights_sum)
        return x_avg, True

    # ─── Pipeline principal ───────────────────────────────────────────────────
    def process(self, frame, in_roundabout=False, roundabout_direction='right'):
        """
        Traite l'image caméra et retourne l'erreur latérale.

        Retourne dict :
          error     : erreur latérale en pixels (>0 → robot décalé à gauche)
          curvature : estimation de courbure (pour détection rond-point)
          debug_bev : image debug annotée (même nom pour compatibilité)
          mask_green, mask_red : masques binaires
        """
        self._frame_count += 1
        h, w = frame.shape[:2]

        # 1. Prétraitement
        frame_eq = self.preprocess(frame)

        # 2. Régions d'intérêt
        # CLOSE : bas 30% de l'image (route juste devant le robot)
        # FAR   : milieu 20-50% (route un peu plus loin, pour anticipation)
        y_close_start = int(h * 0.70)
        y_far_start   = int(h * 0.45)
        y_far_end     = int(h * 0.70)

        roi_close = frame_eq[y_close_start:, :]
        roi_far   = frame_eq[y_far_start:y_far_end, :]

        # 3. Masques sur la zone proche
        mask_g_close, mask_r_close = self.get_masks(roi_close)

        # 4. Masques sur la zone loin (pour courbure)
        mask_g_far, mask_r_far = self.get_masks(roi_far)

        # 5. Position X des lignes (zone proche)
        green_x, green_found = self._find_line_x(mask_g_close, w * 0.25)
        red_x,   red_found   = self._find_line_x(mask_r_close, w * 0.75)

        # Utiliser les dernières valeurs valides si une ligne est perdue
        if green_found:
            self._last_green_x = green_x
        elif self._last_green_x is not None:
            green_x = self._last_green_x

        if red_found:
            self._last_red_x = red_x
        elif self._last_red_x is not None:
            red_x = self._last_red_x

        # 6. Position X des lignes (zone loin, pour courbure)
        green_x_far, _ = self._find_line_x(mask_g_far, green_x)
        red_x_far,   _ = self._find_line_x(mask_r_far, red_x)

        # 7. Centre de la voie
        if in_roundabout:
            # Rond-point : suivre UNE SEULE ligne
            if roundabout_direction == 'right':
                # Suivre la rouge (extérieur), se décaler à gauche
                lane_width_est = abs(red_x - green_x) if (green_found and red_found) else w * 0.3
                center_x = red_x - lane_width_est * 0.5
            else:
                lane_width_est = abs(red_x - green_x) if (green_found and red_found) else w * 0.3
                center_x = green_x + lane_width_est * 0.5
            curvature = 0.0
        else:
            # Normal : milieu entre les deux lignes
            center_x = (green_x + red_x) / 2.0

            # Courbure : différence de position entre proche et loin
            center_x_far = (green_x_far + red_x_far) / 2.0
            curvature = abs(center_x - center_x_far) / w

        # 8. Erreur
        # Pondération 85% proche + 15% anticipation
        if not in_roundabout:
            center_x_far_full = (green_x_far + red_x_far) / 2.0
            target_x = 0.85 * center_x + 0.15 * center_x_far_full
        else:
            target_x = center_x

        error = target_x - w / 2.0

        # 9. Lissage
        self._error_history.append(error)
        smooth_error = float(np.mean(self._error_history))

        self._curvature_history.append(curvature)
        avg_curvature = float(np.mean(self._curvature_history))

        # ── Debug visuel ──────────────────────────────────────────────────
        debug = frame.copy()

        # Dessiner les ROIs
        cv2.line(debug, (0, y_close_start), (w, y_close_start), (100, 100, 100), 1)
        cv2.line(debug, (0, y_far_start), (w, y_far_start), (60, 60, 60), 1)

        # Colorier les masques en surimpression
        overlay_close = debug[y_close_start:, :].copy()
        overlay_close[mask_g_close > 0] = [0, 200, 0]
        overlay_close[mask_r_close > 0] = [0, 0, 200]
        cv2.addWeighted(overlay_close, 0.4, debug[y_close_start:, :], 0.6, 0,
                        debug[y_close_start:, :])

        # Points des lignes
        y_mark = y_close_start + (h - y_close_start) // 2
        if green_found:
            cv2.circle(debug, (int(green_x), y_mark), 8, (0, 255, 0), -1)
        if red_found:
            cv2.circle(debug, (int(red_x), y_mark), 8, (0, 0, 255), -1)

        # Point cible (centre voie)
        cv2.circle(debug, (int(target_x), y_mark), 10, (0, 255, 255), -1)

        # Ligne centrale image
        cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

        # Barre d'erreur
        cv2.putText(debug, f"err={smooth_error:+.1f}px", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.putText(debug, f"curv={avg_curvature:.4f}", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.putText(debug, f"G={'OK' if green_found else 'LOST'} "
                    f"R={'OK' if red_found else 'LOST'}", (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if in_roundabout:
            cv2.putText(debug, f"ROND-POINT [{roundabout_direction.upper()}]",
                        (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Masques combinés pour affichage séparé
        full_mask_g = np.zeros((h, w), dtype=np.uint8)
        full_mask_r = np.zeros((h, w), dtype=np.uint8)
        full_mask_g[y_close_start:, :] = mask_g_close
        full_mask_r[y_close_start:, :] = mask_r_close

        return {
            'error':      smooth_error,
            'curvature':  avg_curvature,
            'raw_error':  error,
            'target_x':   target_x,
            'green_x':    green_x,
            'red_x':      red_x,
            'green_found': green_found,
            'red_found':   red_found,
            'lane_left_x':  min(green_x, red_x),
            'lane_right_x': max(green_x, red_x),
            'lane_width':   abs(red_x - green_x),
            'image_width':  w,
            'debug_bev':  debug,       # même clé pour compatibilité
            'mask_green': full_mask_g,
            'mask_red':   full_mask_r,
        }

    def reset_memory(self):
        """Réinitialise la mémoire."""
        self._error_history.clear()
        self._curvature_history.clear()
        self._last_green_x = None
        self._last_red_x = None
