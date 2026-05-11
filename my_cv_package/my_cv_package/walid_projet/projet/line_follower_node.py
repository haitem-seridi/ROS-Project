import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
#from cv_bridge import CvBridge

import cv2
import numpy as np


class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('line_follower_node')

        #self.bridge = CvBridge()

        self.declare_parameter('image_topic', '/image_raw')
        image_topic = self.get_parameter('image_topic').value

        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )

        # Publie sur /cmd_vel_line — lu par state_machine_node
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.linear_speed = 0.06
        self.kp = 0.003
        self.offset = 150

        self.search_direction = 1   # 1 = gauche, -1 = droite
        self.search_counter = 0
        self.search_step = 15       # frames avant d'inverser la direction

        cv2.namedWindow("Line Follower Debug", cv2.WINDOW_NORMAL)
        self.logged_once = False

        self.get_logger().info(
            f'Line follower node started, listening to {image_topic}'
        )

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
            if msg.encoding == 'yuv422_yuy2':
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
            elif msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                pass
            elif len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if not self.logged_once:
                self.get_logger().info(f"encoding: {msg.encoding}")
                self.get_logger().info(f"shape: {frame.shape}")
                self.get_logger().info(f"dtype: {frame.dtype}")
                self.logged_once = True

            debug_frame = frame.copy()
            height, width = frame.shape[:2]

            roi_y_start = int(height * 0.4)
            roi = frame[roi_y_start:height, :]

            cv2.rectangle(debug_frame, (0, roi_y_start), (width, height), (255, 255, 0), 2)

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            lower_green = np.array([35, 40, 40])
            upper_green = np.array([90, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            contours_green, _ = cv2.findContours(
                mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours_red, _ = cv2.findContours(
                mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            green_center = None
            red_center = None

            if contours_green:
                largest = max(contours_green, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    green_center = (cx, cy)
                    shifted = largest.copy()
                    shifted[:, 0, 1] += roi_y_start
                    cv2.drawContours(debug_frame, [shifted], -1, (0, 255, 0), 2)
                    cv2.circle(debug_frame, (cx, cy + roi_y_start), 8, (0, 255, 0), -1)

            if contours_red:
                largest = max(contours_red, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    red_center = (cx, cy)
                    shifted = largest.copy()
                    shifted[:, 0, 1] += roi_y_start
                    cv2.drawContours(debug_frame, [shifted], -1, (0, 0, 255), 2)
                    cv2.circle(debug_frame, (cx, cy + roi_y_start), 8, (0, 0, 255), -1)

            cmd = Twist()
            image_center_x = width // 2
            image_center_y = roi_y_start + roi.shape[0] // 2
            target_x = None
            target_y = None
            mode_text = "STOP"
            lane_width = 0

            if green_center is not None and red_center is not None:
                target_x = int((green_center[0] + red_center[0]) / 2)
                target_y = int((green_center[1] + red_center[1]) / 2)
                mode_text = "BOTH LINES"
                lane_width = abs(red_center[0] - green_center[0])
            elif red_center is not None:
                target_x = red_center[0] - self.offset
                target_y = red_center[1]
                mode_text = "RED ONLY"
            elif green_center is not None:
                target_x = green_center[0] + self.offset
                target_y = green_center[1]
                mode_text = "GREEN ONLY"
            else:
                # Aucune ligne détectée → osciller gauche/droite pour chercher
                self.search_counter += 1
                if self.search_counter >= self.search_step:
                    self.search_direction *= -1
                    self.search_counter = 0
                cmd.linear.x = 0.0
                cmd.angular.z = 0.2 * self.search_direction
                self.cmd_pub.publish(cmd)
                cv2.putText(debug_frame, "NO LINE -> SEARCHING", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 165, 255) if self.search_direction > 0 else (255, 165, 0), 2)
                cv2.imshow("Line Follower Debug", debug_frame)
                cv2.waitKey(1)
                return

            error = target_x - image_center_x
            cmd.linear.x = self.linear_speed
            cmd.angular.z = max(-0.5, min(0.5, -self.kp * error))
            self.cmd_pub.publish(cmd)

            cv2.circle(debug_frame, (target_x, target_y + roi_y_start), 10, (255, 0, 0), -1)
            cv2.line(debug_frame, (image_center_x, image_center_y),
                     (target_x, target_y + roi_y_start), (255, 255, 0), 2)
            cv2.putText(debug_frame, f'Error: {error}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(debug_frame, mode_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if lane_width > 0:
                cv2.putText(debug_frame, f'Lane: {lane_width}px', (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Line Follower Debug", debug_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Line following failed: {e}')

    def destroy(self):
        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#  """

#  """
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from geometry_msgs.msg import Twist
# import cv2
# import numpy as np
# import time

# class LineFollowerNode(Node):
#     def __init__(self):
#         super().__init__('line_follower_node')

#         # --- PARAMÈTRES ROS ---
#         self.declare_parameter('image_topic', '/camera/image_raw/compressed')
#         image_topic = self.get_parameter('image_topic').value

#         self.image_sub = self.create_subscription(CompressedImage, image_topic, self.image_callback, 10)
#         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

#         # --- VARIABLES DE CONTRÔLE ---
#         self.linear_speed = 0.05
#         self.kp = 0.003
#         self.is_running = False  
        
#         # --- NOUVEAU : LE BUFFER DE RETARD (PURE PURSUIT) ---
#         self.cmd_buffer = []      # La mémoire des commandes
#         self.delay_frames = 10    # Le nombre d'images de retard (à ajuster !)

#         # --- VARIABLES ROND-POINT ---
#         self.lane_width = 300
#         self.roundabout_dir = 'right'  
#         self.seuil_rond_point = 450    
        
#         self.etape_rond_point = 0       
#         self.timer_action = 0.0         
#         self.duree_arret = 2.0          
#         self.duree_traversee = 5.0      

#         # --- INTERFACE GRAPHIQUE ---
#         cv2.namedWindow("Line Follower Debug", cv2.WINDOW_NORMAL)
#         cv2.resizeWindow("Line Follower Debug", 640, 480)

#         cv2.createTrackbar("Vitesse x100", "Line Follower Debug", int(self.linear_speed * 100), 15, 
#                            lambda v: setattr(self, 'linear_speed', v / 100.0))
#         cv2.createTrackbar("Kp x1000", "Line Follower Debug", int(self.kp * 1000), 20, 
#                            lambda v: setattr(self, 'kp', v / 1000.0))
#         cv2.createTrackbar("Seuil R-Point", "Line Follower Debug", self.seuil_rond_point, 600, 
#                            lambda v: setattr(self, 'seuil_rond_point', v))
#         # LE CURSEUR MAGIQUE POUR LE VIRAGE :
#         cv2.createTrackbar("Retard Virage", "Line Follower Debug", self.delay_frames, 30, 
#                            lambda v: setattr(self, 'delay_frames', v))
        
#         self.get_logger().info('✅ Nœud prêt ! Appuie sur [s] pour démarrer.')

#     def image_callback(self, msg):
#         try:
#             # --- DÉCOMPRESSION IMAGE ---
#             np_arr = np.frombuffer(msg.data, np.uint8)
#             frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
#             debug_frame = frame.copy()
#             height, width = frame.shape[:2]

#             # ROI : On laisse la caméra regarder LOIN (40% du haut coupés seulement)
#             roi_y_start = int(height * 0.4)
#             roi = frame[roi_y_start:height, :]
#             cv2.rectangle(debug_frame, (0, roi_y_start), (width, height), (255, 255, 0), 2)

#             # --- DÉTECTION ---
#             hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#             mask_green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))
#             mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
#             mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
#             mask_red = cv2.bitwise_or(mask_red1, mask_red2)

#             contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             green_center = None
#             red_center = None

#             if contours_green:
#                 largest = max(contours_green, key=cv2.contourArea)
#                 if cv2.contourArea(largest) > 100: 
#                     M = cv2.moments(largest)
#                     if M["m00"] > 0:
#                         green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#                         cv2.circle(debug_frame, (green_center[0], green_center[1] + roi_y_start), 8, (0, 255, 0), -1)

#             if contours_red:
#                 largest = max(contours_red, key=cv2.contourArea)
#                 if cv2.contourArea(largest) > 100: 
#                     M = cv2.moments(largest)
#                     if M["m00"] > 0:
#                         red_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#                         cv2.circle(debug_frame, (red_center[0], red_center[1] + roi_y_start), 8, (0, 0, 255), -1)

#             # --- CLAVIER ---
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('s'):
#                 self.is_running = True
#             elif key == ord('q'):
#                 self.is_running = False
#                 self.cmd_buffer.clear() # On vide la mémoire à l'arrêt

#             cmd = Twist()
#             image_center_x = width // 2
            
#             # CE QUE LA CAMÉRA VOIT DANS LE FUTUR :
#             future_target_x = image_center_x 
#             mode_text = "STOP"

#             if not self.is_running:
#                 pass
#             else:
#                 if self.etape_rond_point == 0:
#                     if green_center is not None and red_center is not None:
#                         current_width = red_center[0] - green_center[0]
#                         if current_width > self.seuil_rond_point:
#                             self.etape_rond_point = 1
#                             self.timer_action = time.time()
#                             self.get_logger().info("🛑 Rond-point détecté ! Arrêt...")
#                             self.cmd_buffer.clear() # On vide le buffer pour s'arrêter net
#                         else:
#                             self.lane_width = current_width
#                             future_target_x = int((green_center[0] + red_center[0]) / 2)
#                             mode_text = "CENTRE PARFAIT"

#                     elif red_center is not None:
#                         future_target_x = red_center[0] - (self.lane_width // 2)
#                         mode_text = "1 LIGNE : SUIT ROUGE"
#                     elif green_center is not None:
#                         future_target_x = green_center[0] + (self.lane_width // 2)
#                         mode_text = "1 LIGNE : SUIT VERT"
#                     else:
#                         mode_text = "PERDU"

#                 elif self.etape_rond_point == 1:
#                     mode_text = f"CALCUL DIR : {self.roundabout_dir.upper()}"
#                     if time.time() - self.timer_action > self.duree_arret:
#                         self.etape_rond_point = 2
#                         self.timer_action = time.time()

#                 elif self.etape_rond_point == 2:
#                     mode_text = f"TRAVERSEE -> {self.roundabout_dir.upper()}"
#                     if self.roundabout_dir == 'right':
#                         future_target_x = red_center[0] - (self.lane_width // 2) if red_center else image_center_x + 50
#                     else:
#                         future_target_x = green_center[0] + (self.lane_width // 2) if green_center else image_center_x - 50

#                     if time.time() - self.timer_action > self.duree_traversee:
#                         self.etape_rond_point = 0

#                 # -------------------------------------------------------------
#                 # 🧠 L'INTELLIGENCE DU BUFFER (LE RETARDATEUR)
#                 # -------------------------------------------------------------
#                 if mode_text != "PERDU" and self.etape_rond_point != 1:
#                     # 1. On stocke l'objectif vu par la caméra
#                     self.cmd_buffer.append(future_target_x)
                    
#                     # 2. On attend que le buffer soit plein pour dépiler
#                     delayed_target_x = image_center_x # Par défaut tout droit
                    
#                     while len(self.cmd_buffer) > self.delay_frames:
#                         delayed_target_x = self.cmd_buffer.pop(0) # On prend la plus vieille commande
                    
#                     # 3. Les roues utilisent la vieille commande !
#                     error = delayed_target_x - image_center_x
#                     cmd.linear.x = self.linear_speed
#                     cmd.angular.z = max(-0.8, min(0.8, -self.kp * error))

#                     # Affichage du point de visée ACTUEL des roues (en cyan)
#                     cv2.circle(debug_frame, (delayed_target_x, int(height*0.8)), 12, (255, 255, 0), -1)

#             self.cmd_pub.publish(cmd)

#             # --- AFFICHAGE ---
#             # Point que la caméra regarde pour le futur (en violet)
#             cv2.circle(debug_frame, (future_target_x, int(height*0.5)), 8, (255, 0, 255), -1)
            
#             etat_txt = "RUNNING" if self.is_running else "STOPPED"
#             color_etat = (0, 255, 0) if self.is_running else (0, 0, 255)
#             cv2.putText(debug_frame, f"ETAT: {etat_txt}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_etat, 2)
#             cv2.putText(debug_frame, mode_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#             cv2.putText(debug_frame, f"Retard: {self.delay_frames} frames", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#             cv2.imshow("Line Follower Debug", debug_frame)

#         except Exception as e:
#             pass

# def main(args=None):
#     rclpy.init(args=args)
#     node = LineFollowerNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.cmd_pub.publish(Twist()) 
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()