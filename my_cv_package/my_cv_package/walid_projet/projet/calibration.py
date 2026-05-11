"""
calibration.py — Outil de calibration interactif pour les points BEV
Lance ce nœud pour visualiser les points de transformation homographique
et les ajuster à ta scène Gazebo.

Usage :
  ros2 run projet calibration

Instructions :
  - Clique sur l'image pour voir les coordonnées pixel
  - Ajuste SRC_POINTS dans vision.py selon les 4 coins de la route
  - [s] Sauvegarder les points dans un fichier calibration.yaml
  - [q] Quitter
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import cv2
import os
import yaml

#from .vision import SRC_POINTS, DST_POINTS, DST_W, DST_H


# --- Constantes pour la calibration BEV ---
SRC_POINTS = np.float32([[13.0, 290.0], [636.0, 276.0], [514.0, 159.0], [145.0, 151.0]])
#SRC_POINTS = np.float32([[1, 229], [312, 233], [317, 169], [1, 169]])
#SRC_POINTS = np.float32([[4, 233], [310, 234], [314, 27], [5, 16]])


DST_W, DST_H = 400, 400
DST_POINTS = np.float32([[0, DST_H], [DST_W, DST_H], [DST_W, 0], [0, 0]])
class CalibrationNode(Node):

    def __init__(self):
        super().__init__('calibration')

        self.latest_frame = None
        self._click_pts   = []
        self._src_pts     = SRC_POINTS.tolist()

        # Copie modifiable
        self._M = cv2.getPerspectiveTransform(
            np.float32(self._src_pts), DST_POINTS)

        cv2.namedWindow("Calibration_Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Calibration_BEV",      cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibration_Original", self._on_click)

        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed',
            self.image_callback, qos_profile_sensor_data)

        self.get_logger().info(
            "Calibration — Clique sur l'image pour voir coords "
            "| [s]=sauvegarder | [r]=recalculer | [q]=quitter"
        )
        self.get_logger().info(f"Points SRC actuels : {self._src_pts}")

    def _on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_pts.append([x, y])
            self.get_logger().info(f"Clic : ({x}, {y})")
            if len(self._click_pts) == 4:
                self._src_pts = self._click_pts.copy()
                self._M = cv2.getPerspectiveTransform(
                    np.float32(self._src_pts), DST_POINTS)
                self.get_logger().info(f"4 points sélectionnés : {self._src_pts}")
                self.get_logger().info(
                    "Copie ce bloc dans vision.py :\n"
                    f"SRC_POINTS = np.float32({self._src_pts})"
                )
                self._click_pts = []

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        self.latest_frame = frame.copy()

        # ── Image originale avec points BEV ──────────────────────────────
        orig_debug = frame.copy()
        colors     = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)]
        labels     = ['BG', 'BD', 'HD', 'HG']

        for i, (pt, col, lbl) in enumerate(zip(self._src_pts, colors, labels)):
            cv2.circle(orig_debug, (int(pt[0]), int(pt[1])), 8, col, -1)
            cv2.putText(orig_debug, lbl, (int(pt[0])+10, int(pt[1])-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # Tracer le trapèze de calibration
        pts_np = np.array(self._src_pts, dtype=np.int32)
        cv2.polylines(orig_debug, [pts_np.reshape(-1,1,2)], True, (0,200,200), 2)

        # Crosshair sur derniers clics
        for pt in self._click_pts:
            cv2.circle(orig_debug, pt, 5, (0, 200, 200), -1)

        cv2.putText(orig_debug,
                    f"Clique 4 coins de la route (BG, BD, HD, HG) | pts={len(self._click_pts)}/4",
                    (5, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        cv2.imshow("Calibration_Original", orig_debug)

        # ── Vue BEV ───────────────────────────────────────────────────────
        bev = cv2.warpPerspective(frame, self._M, (DST_W, DST_H))

        # Grille de référence sur BEV
        bev_debug = bev.copy()
        for x in range(0, DST_W, 50):
            cv2.line(bev_debug, (x, 0), (x, DST_H), (40,40,40), 1)
        for y in range(0, DST_H, 50):
            cv2.line(bev_debug, (0, y), (DST_W, y), (40,40,40), 1)

        # Centre
        cv2.line(bev_debug, (DST_W//2, 0), (DST_W//2, DST_H), (100,100,100), 1)

        cv2.putText(bev_debug, "BEV — Les lignes doivent etre verticales !",
                    (5, DST_H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

        cv2.imshow("Calibration_BEV", bev_debug)

        # ── Touches ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            self._save_calibration()
        elif key == ord('r'):
            try:
                self._M = cv2.getPerspectiveTransform(
                    np.float32(self._src_pts), DST_POINTS)
                self.get_logger().info("Homographie recalculée")
            except Exception as e:
                self.get_logger().error(f"Erreur recalcul : {e}")
        elif key == ord('q'):
            rclpy.shutdown()

    def _save_calibration(self):
        data = {
            'src_points': self._src_pts,
            'dst_w':      int(DST_W),
            'dst_h':      int(DST_H),
        }
        path = os.path.expanduser('~/ros2_ws/src/projet/config/bev_calibration.yaml')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f)
        self.get_logger().info(f"✓ Calibration sauvegardée → {path}")
        self.get_logger().info(
            f"Copie dans vision.py :\n"
            f"SRC_POINTS = np.float32({self._src_pts})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
