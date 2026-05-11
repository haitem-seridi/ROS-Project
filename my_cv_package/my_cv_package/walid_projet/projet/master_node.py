"""
master_node.py — Nœud maître : enchaîne tous les challenges automatiquement
Challenge 1 → Challenge 2 → Challenge 3 → Challenge 4

Transition automatique basée sur :
  - C1→C2 : timer (durée fixe) ou détection zone
  - C2→C3 : détection couloir (murs proches des deux côtés)
  - C3→C4 : sortie couloir (plus de murs)
  - C4    : fin

Commandes :
  [s] Démarrer
  [q] Arrêter
  [n] Forcer passage au challenge suivant (manuel)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import cv2
import time
from enum import Enum, auto

from .vision        import LaneVision
from .fsm           import ChallengeFSM, State
from .controller    import LaneController
from .lidar_handler import LidarHandler


class MasterState(Enum):
    IDLE        = auto()
    CHALLENGE1  = auto()
    CHALLENGE2  = auto()
    CHALLENGE3  = auto()
    CHALLENGE4  = auto()
    FINISHED    = auto()


class MasterNode(Node):

    def __init__(self):
        super().__init__('master_node')

        # ── Paramètres ────────────────────────────────────────────────────
        self.declare_parameter('roundabout_direction', 'right')
        self.declare_parameter('linear_speed',         0.08)
        self.declare_parameter('seuil_hsv',            30)
        self.declare_parameter('c1_duration',          60.0)   # secondes max C1
        self.declare_parameter('c2_duration',          60.0)   # secondes max C2

        self.roundabout_direction = self.get_parameter('roundabout_direction').value
        linear_speed  = self.get_parameter('linear_speed').value
        seuil_hsv     = self.get_parameter('seuil_hsv').value
        self.c1_duration = self.get_parameter('c1_duration').value
        self.c2_duration = self.get_parameter('c2_duration').value

        # ── Modules partagés ──────────────────────────────────────────────
        self.vision  = LaneVision(seuil_hsv=seuil_hsv)
        self.ctrl    = LaneController(linear_speed=linear_speed)
        self.lidar   = LidarHandler()

        # FSM Challenge 1/2
        self.fsm = ChallengeFSM(
            roundabout_direction=self.roundabout_direction,
            challenge=1
        )

        # ── État maître ───────────────────────────────────────────────────
        self.master_state     = MasterState.IDLE
        self._state_start     = time.time()
        self._last_time       = time.time()
        self._last_img_time   = time.time()

        # Corridor
        self._corridor_confirm = 0
        self._exit_confirm     = 0

        # Soccer
        self._search_timer  = 0.0
        self._push_timer    = 0.0
        self._soccer_state  = 'search'   # search → approach → push → done
        self._kp_ball       = 0.003

        # ── GUI ───────────────────────────────────────────────────────────
        cv2.namedWindow("MASTER", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MASTER", 420, 60)
        cv2.createTrackbar("Vitesse×100", "MASTER",
                           int(linear_speed * 100), 25,
                           lambda v: setattr(self.ctrl, 'linear_speed', v/100.0))
        cv2.createTrackbar("Seuil HSV", "MASTER", seuil_hsv, 80,
                           lambda s: setattr(self.vision, 'seuil_hsv', s))
        cv2.createTrackbar("0=STOP 1=START", "MASTER", 0, 1,
                           lambda v: (self._transition(MasterState.CHALLENGE1), self.fsm.start(), self.ctrl.reset()) if v == 1 else (self._transition(MasterState.IDLE), self.fsm.stop()))
        cv2.createTrackbar("0=RIGHT 1=LEFT", "MASTER", 0, 1,
                           lambda v: (setattr(self, 'roundabout_direction', 'left' if v == 1 else 'right'), self.fsm.set_direction('left' if v == 1 else 'right')))
        # ── Topics ROS2 ───────────────────────────────────────────────────
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed',
            self.image_callback, qos_profile_sensor_data)
        self.scan_sub  = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub   = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info(
            f"Master prêt | direction={self.roundabout_direction} | "
            "[s]=start [q]=stop [n]=next challenge"
        )

    # ─── LIDAR ────────────────────────────────────────────────────────────────
    def scan_callback(self, msg):
        self.lidar.update(msg)

        # Détection corridor : murs proches des deux côtés
        if self.master_state == MasterState.CHALLENGE2:
            if self.lidar.left_dist < 0.6 and self.lidar.right_dist < 0.6:
                self._corridor_confirm += 1
                if self._corridor_confirm > 15:
                    self._transition(MasterState.CHALLENGE3)
            else:
                self._corridor_confirm = max(0, self._corridor_confirm - 1)

        # Détection sortie corridor
        if self.master_state == MasterState.CHALLENGE3:
            if self.lidar.left_dist > 1.0 and self.lidar.right_dist > 1.0:
                self._exit_confirm += 1
                if self._exit_confirm > 20:
                    self._transition(MasterState.CHALLENGE4)
            else:
                self._exit_confirm = max(0, self._exit_confirm - 1)

    # ─── CAMÉRA ───────────────────────────────────────────────────────────────
    def image_callback(self, msg):
        self._last_img_time = time.time()
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        now = time.time()
        dt  = max(now - self._last_time, 0.001)
        dt  = min(dt, 0.1)
        self._last_time = now
        t_in_state = now - self._state_start

        # ── Clavier ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and self.master_state == MasterState.IDLE:
            self._transition(MasterState.CHALLENGE1)
            self.fsm.start()
            self.ctrl.reset()
        elif key == ord('q'):
            self._transition(MasterState.IDLE)
            self.fsm.stop()
        elif key == ord('n'):
            self._next_challenge()
        elif key == ord('l'):
            self.roundabout_direction = 'left'
            self.fsm.set_direction('left')
        elif key == ord('r'):
            self.roundabout_direction = 'right'
            self.fsm.set_direction('right')

        twist = Twist()

        # ── Challenge 1 ───────────────────────────────────────────────────
        if self.master_state == MasterState.CHALLENGE1:
            self.fsm.challenge = 1
            result = self.vision.process(
                frame,
                in_roundabout=self.fsm.in_roundabout,
                roundabout_direction=self.fsm.roundabout_direction
            )
            error = result['error'] + self.lidar.lateral_shift
            self.fsm.update(result['curvature'], self.lidar.obstacle_dist, dt)

            if self.fsm.is_running:
                lin, ang = self.ctrl.compute(
                    error, result['curvature'], dt,
                    in_roundabout=self.fsm.in_roundabout)
                twist.linear.x  = lin
                twist.angular.z = ang

            # Transition auto après durée max
            if t_in_state > self.c1_duration:
                self._transition(MasterState.CHALLENGE2)

            self._draw_lane_debug(result, error, frame)

        # ── Challenge 2 ───────────────────────────────────────────────────
        elif self.master_state == MasterState.CHALLENGE2:
            self.fsm.challenge = 2
            result = self.vision.process(
                frame,
                in_roundabout=self.fsm.in_roundabout,
                roundabout_direction=self.fsm.roundabout_direction
            )
            shift = self.lidar.lateral_shift * 2.0
            error = result['error'] + shift
            self.fsm.update(result['curvature'], self.lidar.obstacle_dist, dt)

            if self.fsm.is_running:
                extra = 0.0
                if self.fsm.is_avoiding:
                    extra = +0.4 if self.lidar.right_dist < self.lidar.left_dist else -0.4
                lin, ang = self.ctrl.compute(
                    error, result['curvature'], dt,
                    in_roundabout=self.fsm.in_roundabout)
                twist.linear.x  = lin * (0.6 if self.fsm.is_avoiding else 1.0)
                twist.angular.z = ang + extra

            if t_in_state > self.c2_duration:
                self._transition(MasterState.CHALLENGE3)

            self._draw_lane_debug(result, error, frame)

        # ── Challenge 3 : Corridor ─────────────────────────────────────────
        elif self.master_state == MasterState.CHALLENGE3:
            lin, ang = self.lidar.corridor_control()
            twist.linear.x  = lin
            twist.angular.z = ang
            self._draw_corridor_debug(frame)

        # ── Challenge 4 : Soccer ───────────────────────────────────────────
        elif self.master_state == MasterState.CHALLENGE4:
            twist = self._soccer_step(frame, dt)
            self._draw_soccer_debug(frame)

        # ── IDLE / FINISHED ───────────────────────────────────────────────
        elif self.master_state in (MasterState.IDLE, MasterState.FINISHED):
            pass  # twist = 0,0

        self.cmd_pub.publish(twist)
        self._draw_master_hud(frame, t_in_state)

    # ─── Soccer ───────────────────────────────────────────────────────────────
    def _soccer_step(self, frame, dt):
        twist = Twist()
        h, w  = frame.shape[:2]

        # Détection balle HSV jaune-vert
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([25, 80, 80]), np.array([65, 255, 255]))
        k    = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        M    = cv2.moments(mask)
        ball = None
        if M['m00'] > 500:
            bx = int(M['m10'] / M['m00'])
            by = int(M['m01'] / M['m00'])
            ball = (bx, by)

        if self._soccer_state == 'search':
            self._search_timer += dt
            if ball:
                self._soccer_state = 'approach'
                self._search_timer = 0.0
            else:
                twist.angular.z = 0.4
                if self._search_timer > 6.0:
                    twist.angular.z = -0.4
                    if self._search_timer > 12.0:
                        self._search_timer = 0.0

        elif self._soccer_state == 'approach':
            if ball is None:
                self._soccer_state = 'search'
            else:
                bx, by = ball
                error_x = bx - w // 2
                twist.angular.z = float(np.clip(-self._kp_ball * error_x * 3, -1.2, 1.2))
                if by > h * 0.75:
                    self._soccer_state = 'push'
                    self._push_timer = 0.0
                else:
                    twist.linear.x = 0.07

        elif self._soccer_state == 'push':
            self._push_timer += dt
            twist.linear.x = 0.18
            if self._push_timer > 2.5:
                self._soccer_state = 'done'
                self._transition(MasterState.FINISHED)

        return twist

    # ─── Transitions ──────────────────────────────────────────────────────────
    def _transition(self, new_state: MasterState):
        old = self.master_state.name
        self.master_state  = new_state
        self._state_start  = time.time()
        self._corridor_confirm = 0
        self._exit_confirm     = 0

        if new_state == MasterState.CHALLENGE1:
            self.fsm = ChallengeFSM(
                roundabout_direction=self.roundabout_direction,
                challenge=1)
            self.fsm.start()
            self.ctrl.reset()
            self.vision.reset_memory()

        elif new_state == MasterState.CHALLENGE2:
            self.fsm.challenge = 2
            self.vision.reset_memory()

        self.get_logger().info(f"★ {old} → {new_state.name}")

    def _next_challenge(self):
        order = [MasterState.IDLE, MasterState.CHALLENGE1,
                 MasterState.CHALLENGE2, MasterState.CHALLENGE3,
                 MasterState.CHALLENGE4, MasterState.FINISHED]
        idx = order.index(self.master_state)
        if idx < len(order) - 1:
            self._transition(order[idx + 1])
            if order[idx + 1] == MasterState.CHALLENGE1:
                self.fsm.start()

    # ─── Debug visuel ─────────────────────────────────────────────────────────
    def _draw_lane_debug(self, result, error, frame):
        debug = result['debug_bev'].copy()
        cv2.imshow("BEV", debug)

    def _draw_corridor_debug(self, frame):
        canvas = np.zeros((200, 300, 3), dtype=np.uint8)
        cx = 150
        cv2.circle(canvas, (cx, 160), 10, (0, 200, 255), -1)
        lp = int(min(self.lidar.left_dist  * 150, 130))
        rp = int(min(self.lidar.right_dist * 150, 130))
        fp = int(min(self.lidar.front_dist * 150, 150))
        cv2.line(canvas, (cx,160), (cx-lp,160), (0,255,0), 2)
        cv2.line(canvas, (cx,160), (cx+rp,160), (0,255,0), 2)
        cv2.line(canvas, (cx,160), (cx,160-fp), (0,150,255), 2)
        cv2.putText(canvas, f"L={self.lidar.left_dist:.2f} R={self.lidar.right_dist:.2f}",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        cv2.imshow("BEV", canvas)

    def _draw_soccer_debug(self, frame):
        cv2.putText(frame, f"SOCCER: {self._soccer_state.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,0), 2)
        cv2.imshow("BEV", frame)

    def _draw_master_hud(self, frame, t_in_state):
        colors = {
            MasterState.IDLE:       (128,128,128),
            MasterState.CHALLENGE1: (0,220,0),
            MasterState.CHALLENGE2: (0,200,200),
            MasterState.CHALLENGE3: (0,165,255),
            MasterState.CHALLENGE4: (255,165,0),
            MasterState.FINISHED:   (0,255,0),
        }
        hud = np.zeros((60, 420, 3), dtype=np.uint8)
        col = colors.get(self.master_state, (255,255,255))
        cv2.rectangle(hud, (0,0), (420,60), (30,30,30), -1)
        cv2.putText(hud, f"★ {self.master_state.name}  t={t_in_state:.0f}s",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.putText(hud,
                    f"[s]=start [q]=stop [n]=next  dir={self.roundabout_direction}  "
                    f"[l/r]=dir",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
        cv2.imshow("MASTER", hud)


def main(args=None):
    rclpy.init(args=args)
    node = MasterNode()
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
