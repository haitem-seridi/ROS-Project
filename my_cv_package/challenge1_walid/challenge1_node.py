import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import CompressedImage, LaserScan 
from geometry_msgs.msg import Twist 
import numpy as np 
import cv2 

# ═══════════════════════════════════════════ 
# CONSTANTES OPTIMISÉES 
# ═══════════════════════════════════════════ 
VITESSE_MAX       = 0.08  
VITESSE_MIN       = 0.02  

# PID Controller 
KP = 0.0035 
KI = 0.0001 
KD = 0.0090 

class LineFollowerAdvanced(Node): 
    def __init__(self): 
        super().__init__('line_follower_advanced') 

        self.declare_parameter('roundabout_direction', 'right')
        
        self.image_sub = self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, 10) 
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10) 
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10) 

        self.twist = Twist() 
        self.lidar_pixel_shift = 0.0 
        self.last_error = 0.0 
        self.integral_error = 0.0 
        self.widths = {'bottom': 450, 'mid': 300, 'top': 150} 

    def scan_callback(self, msg): 
        """ 
        CORRECTION LIDAR : Angle plus large (45°) et plus loin (50cm) 
        pour ne plus rater les obstacles sur les côtés !
        """ 
        valid_left = [r for r in msg.ranges[0:45] if 0.02 < r < 0.50]
        valid_right = [r for r in msg.ranges[315:360] if 0.02 < r < 0.50]

        min_left = min(valid_left) if valid_left else 99.0
        min_right = min(valid_right) if valid_right else 99.0

        cible_shift = 0.0

        if min_right < 0.50 and min_right <= min_left:
            # Obstacle à droite -> On pousse le centre à DROITE (+) pour forcer un virage à GAUCHE
            force = (0.50 - min_right) / 0.50
            cible_shift = force * 280.0  

        elif min_left < 0.50 and min_left < min_right:
            # Obstacle à gauche -> On pousse le centre à GAUCHE (-) pour forcer un virage à DROITE
            force = (0.50 - min_left) / 0.50
            cible_shift = -force * 280.0

        self.lidar_pixel_shift = 0.85 * self.lidar_pixel_shift + 0.15 * cible_shift 

    def clean_mask(self, mask): 
        kernel = np.ones((5, 5), np.uint8) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
        return mask 

    def get_contour_center(self, mask, y_start, y_end, preferred_side): 
        slice_mask = mask[y_start:y_end, :] 
        contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        valid_contours = [c for c in contours if cv2.moments(c)['m00'] > 50]
        
        if not valid_contours:
            return None

        if len(valid_contours) == 1:
            M = cv2.moments(valid_contours[0])
            return int(M['m10'] / M['m00'])

        centers = [int(cv2.moments(c)['m10'] / cv2.moments(c)['m00']) for c in valid_contours]
        if preferred_side == 'left': return min(centers)
        elif preferred_side == 'right': return max(centers)
        else:
            c = max(valid_contours, key=cv2.contourArea)
            return int(cv2.moments(c)['m10'] / cv2.moments(c)['m00'])

    def get_target_point(self, hsv, y_ratio, width_key, img_h, img_w, direction): 
        y_center = int(img_h * y_ratio) 
        y_start = y_center - 10 
        y_end = y_center + 10 

        mask_g = self.clean_mask(cv2.inRange(hsv, np.array([40, 50, 50]), np.array([85, 255, 255]))) 
        mask_r = self.clean_mask(cv2.bitwise_or( 
            cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])), 
            cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255])) 
        )) 

        cx_g = self.get_contour_center(mask_g, y_start, y_end, direction) 
        cx_r = self.get_contour_center(mask_r, y_start, y_end, direction) 

        target_x = None 

        if cx_g is not None and cx_r is not None: 
            current_width = cx_r - cx_g 
            if current_width > 20: 
                self.widths[width_key] = int(0.9 * self.widths[width_key] + 0.1 * current_width) 
            target_x = (cx_g + cx_r) // 2 
        elif cx_g is not None: 
            target_x = cx_g + self.widths[width_key] // 2 
        elif cx_r is not None: 
            target_x = cx_r - self.widths[width_key] // 2 

        return target_x, cx_g, cx_r, y_center 

    def image_callback(self, msg): 
        np_arr = np.frombuffer(msg.data, np.uint8) 
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
        if frame is None: return 

        h, w, _ = frame.shape 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        img_center_x = w // 2 

        direction = self.get_parameter('roundabout_direction').value

        virtual_center_x = img_center_x + self.lidar_pixel_shift

        t_top, g_top, r_top, y_top = self.get_target_point(hsv, 0.45, 'top', h, w, direction) 
        t_mid, g_mid, r_mid, y_mid = self.get_target_point(hsv, 0.65, 'mid', h, w, direction) 
        t_bot, g_bot, r_bot, y_bot = self.get_target_point(hsv, 0.85, 'bottom', h, w, direction) 

        targets = [] 
        weights = [] 
        
        # ---------------------------------------------------------
        # CORRECTION DU CENTRAGE : C'est la ligne magique !
        # En donnant 80% de poids au BAS de l'image, ton robot 
        # va RESTER AU MILIEU et arrêter de couper les virages.
        # ---------------------------------------------------------
        if t_bot is not None: targets.append(t_bot); weights.append(0.80) 
        if t_mid is not None: targets.append(t_mid); weights.append(0.15) 
        if t_top is not None: targets.append(t_top); weights.append(0.05) 

        if len(targets) > 0: 
            target_x = sum(t * w for t, w in zip(targets, weights)) / sum(weights) 
            error = target_x - virtual_center_x 
        else: 
            error = self.last_error 

        # --- RÉGULATEUR PID ---
        self.integral_error += error 
        self.integral_error = max(min(self.integral_error, 1000), -1000)  
         
        delta_error = error - self.last_error 
        self.last_error = error 

        angular_z = -(KP * error + KI * self.integral_error + KD * delta_error) 
        angular_z = max(min(angular_z, 1.8), -1.8)  

        # --- VITESSE ADAPTATIVE ---
        speed_penalty = abs(error) / (w / 2) 
        linear_x = VITESSE_MAX - (speed_penalty * (VITESSE_MAX - VITESSE_MIN)) 
        linear_x = max(VITESSE_MIN, linear_x) 

        self.twist.linear.x = float(linear_x) 
        self.twist.angular.z = float(angular_z) 
        self.cmd_pub.publish(self.twist) 

        # ══════════════════════════════════════════════════ 
        # AFFICHAGE DEBUG
        # ══════════════════════════════════════════════════ 
        debug = frame.copy() 
        cv2.line(debug, (img_center_x, 0), (img_center_x, h), (255,255,255), 1, cv2.LINE_AA) 

        if abs(self.lidar_pixel_shift) > 5:
            cv2.line(debug, (int(virtual_center_x), 0), (int(virtual_center_x), h), (0,0,255), 3, cv2.LINE_AA)
            cv2.putText(debug, f"<< ESQUIVE >>", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        def draw_scanline(y, tg, cg, cr, label): 
            cv2.line(debug, (0, y), (w, y), (50, 50, 50), 1) 
            if cg: cv2.circle(debug, (cg, y), 6, (0, 255, 0), -1) 
            if cr: cv2.circle(debug, (cr, y), 6, (0, 0, 255), -1) 
            if tg: cv2.circle(debug, (int(tg), y), 8, (0, 255, 255), 2) 

        draw_scanline(y_top, t_top, g_top, r_top, "TOP") 
        draw_scanline(y_mid, t_mid, g_mid, r_mid, "MID") 
        draw_scanline(y_bot, t_bot, g_bot, r_bot, "BOT") 

        if len(targets) > 0: 
            cv2.line(debug, (int(virtual_center_x), h), (int(target_x), h//2), (255, 0, 255), 2, cv2.LINE_AA) 
            cv2.circle(debug, (int(target_x), h//2), 10, (255, 0, 255), -1) 

        cv2.putText(debug, f"PID Err: {error:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 
        cv2.putText(debug, f"Shift: {self.lidar_pixel_shift:.1f} px", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 

        cv2.imshow("Vision & Lidar", debug) 
        cv2.waitKey(1) 

def main(args=None): 
    rclpy.init(args=args) 
    node = LineFollowerAdvanced() 
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