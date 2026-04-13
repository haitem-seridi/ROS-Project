import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

class Calibration(Node):
    def __init__(self):
        super().__init__('calibration')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.callback,
            10
        )

    def callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Masque vert
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Masque rouge (2 plages car rouge wrap autour de 0)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

        # Affiche les masques
        cv2.imshow("Original", frame)
        cv2.imshow("Masque Vert", mask_green)
        cv2.imshow("Masque Rouge", mask_red)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = Calibration()
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